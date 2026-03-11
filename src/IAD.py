import wandb
import os
# import sys
import yaml
import shutil
import fiftyone as fo
import fiftyone.core.dataset as fod
# import fiftyone.zoo as foz # zoo datasets and models
# import cv2
import datetime
import logging
from logging.config import dictConfig
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
# import argparse

# from numpy.typing import NDArray 
# from jsonargparse import ArgumentParser, Namespace
# from cv2.typing import MatLike
from enum import IntFlag, auto
# from logging import Logger
# from contextlib import contextmanager
from fiftyone import ViewField as F # helper for defining views
from pathlib import Path
# from copy import deepcopy
# from functools import partial
from typing import Any, List, Tuple, Optional


# FIFTYONE
import fiftyone.core.dataset as fod

# ANOMALIB

from anomalib.deploy import ExportType
from anomalib.callbacks import LoadModelCallback
from anomalib.models.components import AnomalibModule
from anomalib.loggers import AnomalibWandbLogger
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.engine import Engine
from anomalib.data.utils import Split
# from lightning.pytorch.core import LightningModule

# PYTROCH LIGHTNING
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import TQDMProgressBar

# OWN FILES
from AnomalyDataset import importDataset, exportDataset, FODataModule, importPredictDataset, FODataset
from setup import mapNameToModule, create_model
from visualisation import clipEmbedding
from settings import DATASETS, CATEGORIES, MODELS, DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG, ENGINE_PARAMS, DATAMODULE_PARAMS
from Training import run_inference
# from cameraProcessor import CameraProcessor
from tiling.tiled_ensemble import TrainTiledEnsemble, EvalTiledEnsemble
from utils import find_first_file, exclude_from_logger, loadConfig
from anomalib.visualization import ImageVisualizer

os.environ["TRUST_REMOTE_CODE"] = "1"

class modelFlags(IntFlag):
    default = auto()
    hasModel = auto()
    hasTrainingData = auto()
    hasValidationData = auto()
    hasEmbedding = auto()

# Create the general logger
logger = logging.getLogger(__name__)

class IAD():
    """Class managing the training and validation of the Industrial Anomaly Detection
    """

    def __init__(self, logFileName:str="general.log", outputPath:Path=Path("results"), configDir:Path=Path("configs"), datasetDir:Path=Path("datasets")) -> None:
        self.state:modelFlags = modelFlags.default
        
        self.dataset:FODataset|None = None
        self.datamodule:FODataModule|None = None
        self.datasetName:str|None = None
        self.datasetDir = datasetDir
        self.datasetPath:Path|None = None
        self.category:str|None = None
        self.categories:List[str]|None = None

        self.modelConfigPath:Path|None = None
        self.preProcessorPath:Path|None = None
        self.postProcessorPath:Path|None = None
        self.evaluatorPath:Path|None = None
        self.visualizerPath:Path|None = None
        self.modelConfig:dict[str, Any]|None = None
        self.modelName:str|None = None
        self.model:AnomalibModule|None = None
        self.modelTrained:bool = False
        self.modelCopyPath:Path|None = None


        self.engine:Engine|None = None
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M")

        # general out folder like "results"; probably does not change often
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        self.baseOutpath:Path = outputPath
        self.outputPath:Path = outputPath

        self.configDir:Path = configDir
        self.tilingConfigPath:Path|None = None
        self.isTilingSetup = False

        self.logDir:Path = self.outputPath / "logs"
        # self.logFileNameGeneral = logFileName + f"_{self.now}"
        # self.logPathGeneral = self.logDir / self.logFileNameGeneral

        self.version:int = 0
        self.versionName:str = "version"
        self.runDir:Path|None = None
        self.ckptDir:Path|None = None
        self.ckptFileName:str = "best"
        self.ckptPath:Path|None = None
        self.ckptSuffix:str = ".ckpt"

        self.weightsPath:Path|None = None
        
        self.callbacks:dict[str, Callback]
        self.loadModelCallback:LoadModelCallback|None = None
        self.modelLogger:AnomalibWandbLogger|None = None
        self.runLogger:AnomalibWandbLogger|None = None

        self.shutdown:bool = False        
        self.setupLogging()


    def setupLogging(self):
        """Read the logging.yaml config file from disk and setup the logger accordingly
        """
        self.logDir:Path = self.outputPath / "logs"

        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)
        else:
            shutil.rmtree(self.logDir) # TODO Make this safer
            os.makedirs(self.logDir)

        config_file = self.configDir / "logging.yaml"
        with open(config_file) as file:
            config:dict[str,Any] = yaml.safe_load(file)
        handlers:dict[str, dict[str,Any]]|None = config.get("handlers", None)
        if handlers is not None:
            for handlerName in handlers.keys():
                filename = handlers[handlerName].get("filename", None)
                if filename is not None:
                    config["handlers"][handlerName]["filename"] = self.logDir / filename
        dictConfig(config=config)

    def adjustOutputPath(self) -> Path:
        """This function should be called depending on the current values of dataset, model and the category for which the model is trained
        The idea is the following:

        Models are trained either for an entire dataset or a category of that dataset.
        -> If the dataset is added to the IAD object adjust path to include the dataset.                    outpath/dataset/
        -> If the model is trained for the entire dataset include the model in the path at this stage.      outpath/dataset/model/
        -> If a specific category is trained interject the category.                                        outpath/dataset/category/model
        """
        outputPath:Path = Path()

        if self.datasetName is not None:
            outputPath = self.baseOutpath / self.datasetName
        else:
            logger.info("Set dataset before adjusting path!")
            return self.outputPath
        if self.category is not None:
            outputPath = outputPath / self.category
        if self.modelName is not None:
            outputPath = outputPath / self.modelName
            if self.isTilingSetup:
                outputPath /= "tiled"
            if self.model is not None:
                if isinstance(self.model.visualizer, ImageVisualizer):
                    self.model.visualizer.output_dir = outputPath / "images"
            else:
                logger.info("No model set; Cant adjust path for visualiser (saving result images to disk)")


        self.ckptDir = outputPath / "checkpoints"
        self.ckptPath = self.ckptDir / (self.ckptFileName + self.ckptSuffix)

        self.outputPath = outputPath
        return outputPath

    def generateModel(self, configModelName:str="Padim.yaml", configsDir:Optional[Path] = None) -> None:
        """Generate a model based on a config file always looks under the self.configsDir directory

        Keyword Arguments:
            configModelName -- Name of the config file (default: {"Padim.yaml"})
        """
        if configsDir is not None:
            _configsDir = configsDir
        else:
            _configsDir = self.configDir

        if not configModelName.lower().endswith('.yaml'):
            configModelName = configModelName + '.yaml'
        
        modelConfigPath = _configsDir / configModelName
        if not os.path.exists(modelConfigPath):
            FileNotFoundError(f"Error: Config file {modelConfigPath} not found.")
        self.modelConfigPath = modelConfigPath

        self.modelConfig = loadConfig(self.modelConfigPath, copyPath=None)

        model:dict[str,str|None]|None = self.modelConfig.get("model", None)
        # if isinstance(model, dict):
        if model is not None:
            classpath:str|None = model.get("class_path", None)
            if classpath is not None:
                self.modelName = classpath
            else:
                raise KeyError("class_path not found in modelconfig")
        else:
            raise KeyError("model not found in modelconfig")

        path:str|None = self.modelConfig["model"]["init_args"].pop("post_processor_path",None)
        if path is not None:
            self.postProcessorPath = _configsDir / path
        path:str|None = self.modelConfig["model"]["init_args"].pop("pre_processor_path",None)
        if path is not None:
            self.preProcessorPath = _configsDir / path
        path:str|None = self.modelConfig["model"]["init_args"].pop("evaluator_path",None)
        if path is not None:
            self.evaluatorPath = _configsDir / path

        logger.info(f"Model {self.modelName} loaded: {self.modelConfig}")
        self.model = create_model(self.modelName, self.modelConfig["model"]["init_args"])
        logger.info(f"Model {self.model} created.")

    def loadTrainedModel(self, folder:Path) -> None:
        """DEPRECATED
        Load a trained model from a model.pt from a folder; Especially look for a model.pt file there. Then search for a config .yaml-file
        in that folder

        Arguments:
            folderPath -- Path to the folder of the trained model
        """
        # Find the model weights (the actual model) at the given folderPath
        modelPath:Path|None = find_first_file(folder, "model.pt")
        if modelPath is None:
            raise FileNotFoundError(f"model.pt not found in {folder}!")
        else:
            logger.info(f"Found model weights at {modelPath}")
            self.modelPath = modelPath
        
        # Find the config to the model.pt at the given folderPath
        modelConfigPath = None
        for model in MODELS:
            modelConfigPath = find_first_file(folder, f"{model}.yaml")
        if modelConfigPath is None:
            FileNotFoundError(f"No .yaml config file found in {folder}!")
        else:
            logger.info(f"Found model config at {modelConfigPath}")
            self.modelConfigPath = modelConfigPath
        
        self.generateModel(str(self.modelConfigPath))
        loadModelCallback = LoadModelCallback(weights_path=str(self.modelPath))
        self.callbacks["loadModel"] = loadModelCallback
        self.state |= modelFlags.hasModel
        logger.info("Model loaded.")
        self.adjustOutputPath()

    def loadCheckpoint(self, path:Path, modelName:str):
        """Load model from a lightning Checkpoint file

        Arguments:
            path -- Path to the checkpoint file.
        """
        # TODO: Adapt for other model types; check path or ask for model type
        # self.model = Padim.load_from_checkpoint(path)
        if modelName not in MODELS:
            KeyError(f"Model: {modelName} not known")
        modelInstance = mapNameToModule(modelName)
        self.model = modelInstance.load_from_checkpoint(path)

    def loadDatasetFromDatabase(self, datasetName:str):
        """Load a dataset from the voxel51 MongoDB

        Arguments:
            datasetName -- Name of the dataset as it was saved into the database
        """
        if fo.dataset_exists(datasetName):
            # logger.info(f"Dataset '{datasetName}' exists in database")
            self._dataset = fo.load_dataset(datasetName)
            if Split.TRAIN in list(self._dataset.tags):
                self.state |= modelFlags.hasTrainingData
            if Split.TEST in list(self._dataset.tags):
                self.state |= modelFlags.hasValidationData
            self.datasetName = datasetName
            self.categories = self._dataset.distinct("category.label")
            logger.info(f"Loaded dataset '{datasetName}' from database!")
        else:
            logger.info(f"Dataset '{datasetName}' does not exist in database")
        self.dataset = self._dataset
        
    def loadDatasetFromDisk(self, datasetPath: Path, datasetName:str = "", overwrite:bool=True, merge:bool=False, split:Tuple[str,...] = ("train", "test")):
        """Load a dataset from a given path on the disk and give it a name for the Voxel51 MongoDB.
        Existing dataset in the database can be overwritten or merged with.
        

        Arguments:
            datasetPath -- directory where to find the dataset

        Keyword Arguments:
            datasetName -- Name of the dataset for the database (default: {""})
            overwrite -- overwrite a potential dataset in the database with the same name? (default: {True})
            merge -- merge with a potential dataset in the database with the same name? (default: {False})
            split -- Does the data contain training, testing both or prediction data (default: {("train", "test")})
        """
        if "pred" in split:
            self._dataset, self.anomalibPredDataset = importPredictDataset(datasetPath, name=datasetName)
        else:
            self.anomalibPredDataset = None
            if overwrite and merge:
                logger.info("Overwrite and merge should not both be true. Overwrite is ignored...")
                overwrite = False
            elif not datasetPath.exists():
                FileNotFoundError(f"Dataset {datasetPath} does not exit")
            elif fo.dataset_exists(datasetName) and not overwrite and not merge:
                logger.info(f"Dataset '{datasetName}' already exists in database")
                logger.info("Loading from database")
                self.loadDatasetFromDatabase(datasetName)            
            elif fo.dataset_exists(datasetName) and overwrite:
                logger.info(f"Dataset '{datasetName}' already exists in database")
                logger.info("Overwriting")
                dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName,
                    overwrite=overwrite,
                    split=split
                )
                self._dataset = dataset
                self.datasetName = datasetName
            elif fo.dataset_exists(datasetName) and merge:
                logger.info(f"Dataset '{datasetName}' already exists in database")
                logger.info("Merging")
                dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName+"_"+str(self.now),
                    overwrite=False,
                    split=split
                )
                self._dataset = fo.load_dataset(datasetName)
                logger.info(f"Loading {datasetName} from database for merging")
                self._dataset.merge_samples(dataset)
                self.datasetName = datasetName
            else:
                logger.info(f"Loading {datasetName} from disk")
                dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName,
                    overwrite=overwrite,
                    split=split
                )
                self._dataset = dataset
                self.datasetName = datasetName

        if Split.TRAIN in list(self._dataset.tags):
            self.state |= modelFlags.hasTrainingData
        if Split.TEST in list(self._dataset.tags):
            self.state |= modelFlags.hasValidationData
        self.categories = self._dataset.distinct("category.label")
        self.dataset = self._dataset
        logger.info(f"There are {len(self.dataset)} images in the {datasetName} dataset")

    def _setupDatamodule(self, datamoduleParams: dict[str, Any]) -> FODataModule:
        """Setup a datamodule from the dataset for running a model on the data

        Arguments:
            datamoduleParams -- _description_

        Returns:
            _description_
        """
        if self.dataset is not None:
            if self.datasetName is None:
                self.datasetName = "unnamedDataset"
            datamodule = FODataModule(name=self.datasetName, samples=self.dataset, root=self.outputPath, **datamoduleParams)
            datamodule.setup()
            self.datamodule = datamodule
        else:
            logger.info("No dataset available")
            exit(1)
        if self.model is None:
            logger.info("No model available")
            exit(1)
        return self.datamodule

    def generateEmbedding(self) -> None:
        """Generate an embedding of the dataset into a 2d space to visually inspect the data. Opens a voxel51 session

        Raises:
            AttributeError: Needs a dataset to be set
        """

        if self.dataset is None:
            raise AttributeError("No dataset loaded. Please load or create a dataset first.")
        else:
            if not modelFlags.hasEmbedding in self.state:
                with exclude_from_logger():
                    clipEmbedding(self.dataset)
                logger.info("Finished embedding computation.")
                logger.info("Please reload the FiftyOne app to see the new visualizations.")
            else:
                logger.info("Data already has embedding")
                logger.info("You find the visualizations by clicking the '+' next to Samples and choosing Embeddings.")
            self.state |= modelFlags.hasEmbedding
        
    def copyFilesToPath(self, path:Path):
        """To make it easier to understand how a model was trained copy all relevant files to the output folder.
        """
        # If the configs should be copied create the folder and copy the general config    
        if not path.exists():
            path.mkdir(parents=True)

        # Copy the modelConfig file if possible
        if self.modelConfigPath is not None:
            _, configFileName = os.path.split(self.modelConfigPath)
            shutil.copy2(self.modelConfigPath, path / configFileName)
        else:
            raise AttributeError(f"Model config path is empty. Load model before calling this function!")
        
        if self.preProcessorPath is not None:
            _, fileName = os.path.split(self.preProcessorPath)
            shutil.copy2(self.preProcessorPath, path / fileName)
        if self.postProcessorPath is not None:
            _, fileName = os.path.split(self.postProcessorPath)
            shutil.copy2(self.postProcessorPath, path / fileName)
        if self.evaluatorPath is not None:
            _, fileName = os.path.split(self.evaluatorPath)
            shutil.copy2(self.evaluatorPath, path / fileName)
        if self.visualizerPath is not None:
            _, fileName = os.path.split(self.visualizerPath)
            shutil.copy2(self.visualizerPath, path / fileName)

    def copyFilesToOutputPath(self):
        self.copyFilesToPath(self.outputPath)

    def selectCategory(self, category:str):
        """Select a category from the dataset for which the model is trained.

        Keyword Arguments:
            category -- Choosen category; None -> all categories (default: {None})

        Returns:
            returns the category if it is available in the dataset
            returns None if category not found or dataset does not contain category
        """
        if category == "all":
            self.category = None
            logger.info("Selecting all as category selects all categories")
            self.dataset = self._dataset
            logger.info(f"There are {len(self.dataset)} images in the {self.datasetName} dataset")

        else:
            if (self.categories == None) or len(self.categories) == 0:
                raise AttributeError("There are no categories to select from")
            elif category not in self.categories:
                AttributeError(f"Category {category} not found! Available categories are: {self.categories}")
            else:
                logger.info(f"Category found in categories for this dataset:")
                self.category = category
            self.dataset = self._dataset.filter_labels("category", F("label").is_in([category]))
            logger.info(f"There are {len(self.dataset)} images in the selected category {category} of the {self.datasetName} dataset")

    #####
    # Logging
    #####

    # def setupRunLogging(self, runName:str, runDir:Path, logFileName:str="run.log") -> Logger:
    #     pass
    #     logger = logging.getLogger(f"general.{runName}")
    #     logger.setLevel(logging.INFO)
    #     fileHandler = logging.FileHandler(os.path.join(runDir, logFileName))
    #     fileHandler.setLevel(logging.INFO)
    #     logFormater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     fileHandler.setFormatter(logFormater)
    #     logger.addHandler(fileHandler)
    #     return logger

    def setupCallbacks(self):
        """Setup the standard callbacks for running a model
        Checkpointing to a file based on a performance monitor
        Timing the process
        creating a visual progress bar


        Arguments:
            dir -- _description_
        """
        # self.runDir:Path = Path(os.path.join(self.outputPath, runName, f"version_{version}"))
        # self.ckptDir = path / "checkpoints"
        if self.ckptDir is not None:
            if not self.ckptDir.exists():
                self.ckptDir.mkdir(parents=True)
            self.ckptPath = self.ckptDir / (self.ckptFileName + self.ckptSuffix)
        else:
            AttributeError(f"ckptDir is both None")

        checkpointCallback = ModelCheckpoint(
            dirpath=self.ckptDir,
            filename=self.ckptFileName,
            monitor="image_F1AdaptiveThreshold",  # val_loss not found?
            verbose=True,
            save_top_k=1,  # Save only the best model
            mode="min",  # Save the model with the minimum training loss,
            enable_version_counter=False
        )
        
        # graphCallback = GraphLogger()
        timerCallback = TimerCallback()
        progressBar = TQDMProgressBar(refresh_rate=50)

        self.callbacks = {
            "progress_bar": progressBar,
            "checkpoint": checkpointCallback,
            # "graph": graphCallback,
            "timer": timerCallback
        }

    def setupWandBLogger(self, runName:str, runDir:Path, version:int|str):
        """Setup a weights and biases logger that can be review from the browser

        Arguments:
            runName -- _description_
            runDir -- _description_
            version -- _description_
        """
        self.runLogger = AnomalibWandbLogger(
            name=runName,
            save_dir=runDir,
            version=str(version),
            project="Glas 4.0",
            offline=False,
            entity="daniel-pommer-technische-hochschule-n-rnberg-georg-simon-ohm",
        )

    #####
    # Training
    #####

    def setupTiling(self, tilingConfigPath:Path):
        """Setup the training on tiles by reading a config file.

        Arguments:
            tilingConfigPath -- Path to the config file. Usually in configs folder

        Raises:
            FileNotFoundError: If Path not found
        """
        if (tilingConfigPath.suffix == ".yaml") & tilingConfigPath.exists():
            with open(tilingConfigPath, "r") as file:
                self.tilingConfigDict:dict[str,Any] = yaml.safe_load(file)
            rootDir:str|None = self.tilingConfigDict.get("rootDir", None)
            if rootDir is None:
                self.tilingConfigDict["rootDir"] = self.outputPath
            else:
                if rootDir == "":
                    self.tilingConfigDict["rootDir"] = self.outputPath

            # ckptPath:str|None = self.tilingConfigDict.get("ckptPath", None)
            # if ckptPath is None:
            #     self.tilingConfigDict["ckptPath"] = self.ckptPath
            # else:
            #     if ckptPath == "":
            #         self.tilingConfigDict["ckptPath"] = self.ckptPath
        else:
            # logger.info()
            raise FileNotFoundError(f"{tilingConfigPath} file not found.")
        self.tilingConfigPath = tilingConfigPath
        self.isTilingSetup = True
        return self.isTilingSetup

    def _checkBeforeTraining(self, model:bool=True, dataset:bool=True) -> bool:
        """Before the training can start check if a model and a dataset exist.

        Raises:
            AttributeError: model not found
            AttributeError: dataset not found

        Returns:
            True if both exist; otherwise False
        """
        if model and self.model is None:
            raise AttributeError("Expected model attribute to be set")
        if dataset and self.dataset is None:
            raise AttributeError("Expected data attribute to be set")
        return True

    def train(self, trainingConfigPath:Path, tiling:bool=False) -> None:
        """The the current model on the current dataset for the given category. Train the model on all categories.
        Loads a .yaml-file that describes the training from the configs folder. it sho

        Arguments:
            trainingConfigName -- Path to the training yaml file.

        Keyword Arguments:
            category -- Category in the dataset to train the model on. If None train on all categories (default: {None})
        """

        if not trainingConfigPath.suffix == ".yaml":
            raise FileNotFoundError("Config file needs to have .yaml suffix")
        if not trainingConfigPath.exists():
            raise FileNotFoundError("Error: .yaml config file not found in configs folder.")
        if tiling and not self.isTilingSetup:
            if self.tilingConfigPath is not None:
                self.setupTiling(self.tilingConfigPath)
            else:
                raise ValueError(f"tilingConfigPath is not set")

        with open(trainingConfigPath, 'r') as f:
            self.trainingConfig = yaml.safe_load(f) 
        if self.category is not None:
            self.runName = f"{self.modelName}-{self.datasetName}-{self.category}"
        else:
            self.runName =f"{self.modelName}-{self.datasetName}"
            # self.runDir = self.outputPath / self.runName
            # if not self.runDir.exists():
            #     self.runDir.mkdir(parents=True)

            # self.modelLogger, self.callbacks, self.runDir, self.ckptPath = setupWandBLoggingAndCallbacks(
            #     logDir=str(self.logDir),
            #     runName=self.runName,
            #     version=self.version,
            #     ckptFileName=self.ckptFileName
            # )

            # self.runLogger = self.setupRunLogging(runName=self.runName, runDir=self.runDir)
        self.adjustOutputPath()
        self.setupLogging()
        self.setupCallbacks()
        self.setupWandBLogger(self.runName, self.outputPath, self.version)
        if tiling:
            self._trainTiledModel(self.trainingConfig)
        else:
            self._trainSingleModel(self.trainingConfig)

    def _trainSingleModel(self, config:dict[str, Any]):
        """Train a singular model on the dataset based on a training config file.

        Arguments:
            trainingConfig -- _description_
        """
        self._checkBeforeTraining()
        self.engineParams = {key: config[key] for key in ENGINE_PARAMS if key in config}
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule = self._setupDatamodule(self.datamoduleParams)
        self.engine = Engine(callbacks=list(self.callbacks.values()), logger=self.runLogger, **self.engineParams)
        self.engine.fit(model=self.model, datamodule=self.datamodule)

        logger.info("Running inference on dataset...")
        with exclude_from_logger():
            run_inference(self.dataset, self.engine, self.model, self.modelName)

        self.currentSession = fo.launch_app(self.dataset)

    def _trainTiledModel(self, config:dict[str, Any], modelConfig:dict[str,Any]|None=None):
        """Train a tiled model on the dataset based on a training config file.

        Arguments:
            config -- Config dictionary

        Returns:
            the pipeline for the training jobs
        """
        if modelConfig is not None:
            self.modelConfig = modelConfig

        self._checkBeforeTraining(dataset=True, model=False)

        # Setup datamodule for the tiling 
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule = self._setupDatamodule(self.datamoduleParams)

        self.adjustOutputPath()
        if self.tilingConfigPath is not None:
            self.setupTiling(self.tilingConfigPath)
        else:
            ValueError(f"self.tilingConfigPath is not set")

        trainPipeline = TrainTiledEnsemble(rootDir=self.outputPath, datamodule=self.datamodule, dataset=self.dataset)
        # trainPipeline.setDatamodule(datamodule=self.datamodule) # Split the dataset into different dataloaders for training, validation and test
        # trainPipeline.setFODataset(dataset=self.dataset) # The entire dataset with all samples. samples are tagged with split (train, val, test)
        if self.modelConfig is not None:
            self.tilingConfigDict["TrainModels"] = self.modelConfig

        else:
            raise ValueError("modelConfig missing")
        trainPipeline.run(self.tilingConfigDict, "")
        # trainPipeline.run(args)
        self.trainPipeline = trainPipeline
        return self.trainPipeline
    
    def predict(self, config:Path, tiling:bool=False, ckptPath:Path|None=None):
        """predictuate the current model on the dataset

        Arguments: 
            config -- path to the config

        Keyword Arguments:
            tiling -- _description_ (default: {False})
        """
        # if not self._checkBeforeTraining():
        #     exit(1)
        if not config.suffix == ".yaml":
            FileNotFoundError("Config file needs to have .yaml suffix")
        if not config.exists():
            FileNotFoundError("Error: .yaml config file not found in configs folder.")
        if tiling and not self.isTilingSetup:
            AttributeError("tiling is not setup: Call 'setupTiling()")

        with open(config, 'r') as f:
            self.trainingConfig = yaml.safe_load(f) 
        if self.category is not None:
            self.runName = f"{self.modelName}-{self.datasetName}-{self.category}"
        else:
            self.runName =f"{self.modelName}-{self.datasetName}"

        ckptPath = self.ckptPath.parent.resolve()

        self.adjustOutputPath()
        self.setupLogging()
        self.setupCallbacks()
        self.setupWandBLogger(self.runName, self.outputPath, self.version)
        if tiling:
            self._predictTiledModel(self.trainingConfig, ckptPath=ckptPath)
        else:
            self._evalSingleModel(self.trainingConfig)
        
    def eval(self, config:Path, tiling:bool=False, ckptDir:Path|None=None):
        """predictuate the current model on the dataset

        Arguments: 
            config -- path to the config

        Keyword Arguments:
            tiling -- _description_ (default: {False})
        """
        # if not self._checkBeforeTraining():
        #     exit(1)
        if not config.suffix == ".yaml":
            FileNotFoundError("Config file needs to have .yaml suffix")
        if not config.exists():
            FileNotFoundError("Error: .yaml config file not found in configs folder.")
        if tiling and not self.isTilingSetup:
            AttributeError("tiling is not setup: Call 'setupTiling()")

        with open(config, 'r') as f:
            self.trainingConfig = yaml.safe_load(f) 
        if self.category is not None:
            self.runName = f"{self.modelName}-{self.datasetName}-{self.category}"
        else:
            self.runName =f"{self.modelName}-{self.datasetName}"

        self.adjustOutputPath()
        self.setupLogging()
        self.setupCallbacks()
        self.setupWandBLogger(self.runName, self.outputPath, self.version)
        if tiling:
            self._evalTiledModel(self.trainingConfig)
        else:
            self._evalSingleModel(self.trainingConfig)

    def _evalTiledModel(self, config:dict[str, Any], ckptDir:Path|None=None):
        """Evaluate a tiled model on the dataset

        Arguments:
            config -- config dictionary
        """
        self._checkBeforeTraining(dataset=True, model=False)
        # Setup datamodule for the tiling 
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule = self._setupDatamodule(self.datamoduleParams)

        self.adjustOutputPath()

        logger.info("Running tiled ensemble test pipeline.")
        # pass the root dir from train run to load checkpoints
        if self.tilingConfigPath is not None:
            self.setupTiling(self.tilingConfigPath)
        else:
            ValueError(f"self.tilingConfigPath is not set")
        test_pipeline = EvalTiledEnsemble(self.tilingConfigDict["rootDir"], dataset=self.dataset)
        test_pipeline.setDatamodule(datamodule=self.datamodule)
        if self.model:
            self.tilingConfigDict["TrainModels"] = self.modelConfig
        self.tilingConfigDict["ckptPath"] = ckptDir
        test_pipeline.run(self.tilingConfigDict, "")

    def _evalSingleModel(self, config:dict[str, Any]):
        """Evaluate a singular model on the dataset

        Arguments:
            config -- config dictionary
        """
        self._checkBeforeTraining()
        self.engineParams = {key: config[key] for key in ENGINE_PARAMS if key in config}
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule = self._setupDatamodule(self.datamoduleParams)
        self.engine = Engine(callbacks=list(self.callbacks.values()), logger=self.runLogger, **self.engineParams)
        # prediction = self.engine.predict(model=self.model, datamodule=self.datamodule) # The result of this function is kinda useless

        # logger.info("Running inference on dataset...")
        with exclude_from_logger():
            if self.model is not None and self.modelName is not None:
                run_inference(self.dataset, self.engine, self.model, self.modelName)
            else:
                AttributeError("Need self.model and self.modelName")

        self.currentSession = fo.launch_app(self.dataset)

    def _predictTiledModel(self, config:dict[str, Any], ckptPath:Path|None):
        """Evaluate a tiled model on the dataset

        Arguments:
            config -- config dictionary
        """
        self._checkBeforeTraining(dataset=True, model=False)
        # Setup datamodule for the tiling 
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule = self._setupDatamodule(self.datamoduleParams)

        self.adjustOutputPath()

        logger.info("Running tiled ensemble pred pipeline.")
        # pass the root dir from train run to load checkpoints
        if self.tilingConfigPath is not None:
            self.setupTiling(self.tilingConfigPath)
        else:
            ValueError(f"self.tilingConfigPath is not set")
        test_pipeline = EvalTiledEnsemble(self.tilingConfigDict["rootDir"], dataset=self.dataset)
        test_pipeline.setDatamodule(datamodule=self.datamodule)
        test_pipeline.dataset = self.anomalibPredDataset
        if self.model:
            self.tilingConfigDict["TrainModels"] = self.modelConfig
        
        self.tilingConfigDict["ckptPath"] = ckptPath
        logger.info(f"ckptPath: {ckptPath}")
        test_pipeline.run(self.tilingConfigDict, "")

    # def parseTilingConfig(self, path:Path):
    #     with Path(path).open(encoding="utf-8") as file:
    #         tilingConfigDict = yaml.safe_load(file)
        
    #     rootDir:str|None = tilingConfigDict.get("rootDir", None)
    #     if rootDir is None:
    #         tilingConfigDict["rootDir"] = self.outputPath
    #     else:
    #         if rootDir == "":
    #             tilingConfigDict["rootDir"] = self.outputPath

    #     ckptPath:str|None = tilingConfigDict.get("ckptPath", None)
    #     if ckptPath is None:
    #         tilingConfigDict["ckptPath"] = self.ckptPath
    #     else:
    #         if ckptPath == "":
    #             tilingConfigDict["ckptPath"] = self.ckptPath
    #     self.isTilingSetup = True
    #     return tilingConfigDict

    #####
    # Export to Disk
    #####

    def exportModel(self) -> Path|None:
        if self.model is not None:
            self.modelWeightsPath:Path|None = self.engine.export(model=self.model,
                        export_type=ExportType.TORCH,
                        export_root=self.outputPath,
                        model_file_name="model")
        else:
            logger.info("No model available")
            exit(1)

        return self.modelWeightsPath
    
    def exportResults(self, exportPath:Path) -> None:
        # Create the destination directory if it doesn't exist
        os.makedirs(exportPath, exist_ok=True)

        # Copy each item in the source directory
        for item in os.listdir(self.outputPath):
            src_path = os.path.join(self.outputPath, item)
            dst_path = os.path.join(exportPath, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    def exportDataset(self, exportPath:Path) -> None:
        exportDataset(self.dataset, exportPath)
    
    #####
    # Fiftyone Features
    #####

    def launchSession(self):
        if self.dataset is not None:
            self.session = fo.launch_app(self.dataset)
            logger.info(f"Session addess and port: {self.session.server_address}:{self.session.server_port}")
        else:
            logger.info("Load a dataset first!")

if __name__ == "__main__":
    # Point FiftyOne at your MongoDB instance
    os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://localhost"
    os.environ["WANDB_API_KEY"] = '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'

    # Try to setup wandb
    wandb.login()

    logger = logging.getLogger("logger")
    logger.debug("This should go to mongodb.log")

    tiling      = True
    modelName   = "Padim"
    datasetName = "MVTecADShort"
    category    = "cable"
    train       = False
    evaluate    = True
    iad = IAD()

    iad.generateModel(f"{modelName}.yaml")
    datasetPath = Path(os.path.join("datasets", datasetName))
    iad.loadDatasetFromDisk(datasetPath, datasetName, overwrite=True, merge=False)
    iad.selectCategory(category)
    iad.adjustOutputPath()
    iad.copyFilesToOutputPath()
    if iad.dataset is None:
        exit(1)
    if tiling:
        iad.setupTiling(Path("configs/TiledEnsemble.yaml"))
    if train:
        iad.train(Path(f"configs/{modelName}_Training.yaml"), tiling=tiling)
    # iad.launchSession()
    if evaluate:
        if iad.ckptPath is not None:
            if not tiling:
                iad.loadCheckpoint(iad.ckptPath, f"{modelName}")
            iad.eval(Path("configs/Eval.yaml"), tiling=tiling)
    iad.launchSession()
    shutdown = False
    while not shutdown:
        userInput = input("shutdown?\n")
        if userInput:
            shutdown = True
        
    