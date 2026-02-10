import wandb
import os
import sys
import yaml
import shutil
import fiftyone as fo
import fiftyone.core.dataset as fod
import fiftyone.zoo as foz # zoo datasets and models
import cv2
import datetime
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse

from numpy.typing import NDArray
from jsonargparse import ArgumentParser, Namespace
from cv2.typing import MatLike
from enum import Enum, IntFlag, auto
from logging import Logger
from contextlib import contextmanager
from fiftyone import ViewField as F # helper for defining views
from pathlib import Path
from copy import deepcopy
from functools import partial
from typing import Any, List, Tuple, Dict, Optional

# ANOMALIB
from anomalib.models import Padim
from anomalib.metrics import Evaluator
from anomalib.deploy import ExportType
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import AUROC, AUPR, F1AdaptiveThreshold, F1Score
from anomalib.callbacks import LoadModelCallback
from anomalib.models.components import AnomalibModule
from anomalib.loggers import AnomalibTensorBoardLogger, AnomalibWandbLogger
from anomalib.callbacks import ModelCheckpoint, GraphLogger, TimerCallback
from anomalib.engine import Engine
from anomalib.data.utils import DirType, LabelName, Split

# PYTROCH LIGHTNING
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import TQDMProgressBar

# OWN FILES
from AnomalyDataset import TestDataImporter, TrainTestDataImporter, importDataset, importPredictDataset, exportDataset, FODataModule, FODataset
from setup import create_datamodule, create_model, setupTensorboardLoggingAndCallbacks, setupLogging, LoggerWriter, LoggerStdin, setupWandBLoggingAndCallbacks
from visualisation import clipEmbedding, resnetEmbedding
from settings import DATASETS, CATEGORIES, MODELS, DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG, ENGINE_PARAMS, DATAMODULE_PARAMS
from Training import run_inference, train_and_export_model, setupModel
from cameraProcessor import CameraProcessor
from tiling.tiled_ensemble import TrainTiledEnsemble, EvalTiledEnsemble
from utils import find_first_file, exclude_from_logger, loadConfig

os.environ["TRUST_REMOTE_CODE"] = "1"

class modelFlags(IntFlag):
    default = auto()
    hasModel = auto()
    hasTrainingData = auto()
    hasValidationData = auto()
    hasEmbedding = auto()

class IAD():
    """Class managing the training and validation of the Industrial Anomaly Detection
    """

    def __init__(self, logFileName:str="general.log", outputPath:Path=Path("results"), configDir:Path=Path("configs"), datasetDir:Path=Path("datasets")) -> None:
        self.state:modelFlags = modelFlags.default
        
        self.dataset:fod.Dataset|None = None
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
        self.tilingConfigPath:Path = self.configDir / "TiledEnsemble.yaml"
        self.isTilingSetup = False

        self.logDir:Path = self.outputPath / "logs"
        self.logFileNameGeneral = logFileName + f"_{self.now}"
        self.logPathGeneral = self.logDir / self.logFileNameGeneral

        self.version:int = 0
        self.versionName:str = "version"
        self.runDir:Path|None = None
        self.ckptDir:Path|None = None
        self.ckptFileName:str = "best.ckpt"
        self.ckptPath:Path|None = None

        self.weightsPath:Path|None = None
        
        self.callbacks:dict[str, Callback]
        self.loadModelCallback:LoadModelCallback|None = None
        self.modelLogger:AnomalibWandbLogger|None = None
        self.runLogger:AnomalibWandbLogger|None = None

        self.shutdown:bool = False


        # Create the general logger
        self.generalLogger = logging.getLogger('general')
        self.generalLogger.setLevel(logging.INFO)
        logFormater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Create a file handler for the general logger
        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)
        fileHandler = logging.FileHandler(self.logPathGeneral)
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(logFormater)
        self.generalLogger.addHandler(fileHandler)
        # Create a StreamHandler to duplicate console output to the logger
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.generalLogger.addHandler(console_handler)
        sys.stdout = LoggerWriter(self.generalLogger, logging.INFO)
        sys.stdin = LoggerStdin(self.generalLogger, logging.INFO)

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
            print("Set dataset before adjusting path!")
            return self.outputPath
        if self.category is not None:
            outputPath = outputPath / self.category
        if self.modelName is not None:
            outputPath = outputPath / self.modelName
            if self.model is not None:
                self.model.visualizer.output_dir = outputPath / "imagesByLabels"
            else:
                print("No model set; Cant adjust path for visualiser (saving result images to disk)")
        if self.isTilingSetup:
            outputPath /= "tiled"

        self.ckptDir = outputPath / "checkpoints"
        self.ckptPath = self.ckptDir / self.ckptFileName

        self.outputPath = outputPath
        return outputPath

    def generateModel(self, configModelName:str="padim.yaml") -> None:
        """Generate a model based on a config file always looks under the self.configsDir directory

        Keyword Arguments:
            configModelName -- Name of the config file (default: {"padim.yaml"})
        """
        if not configModelName.lower().endswith('.yaml'):
            configModelName = configModelName + '.yaml'
        
        modelConfigPath = self.configDir / configModelName
        if not os.path.exists(modelConfigPath):
            FileNotFoundError(f"Error: Config file {modelConfigPath} not found.")
        self.modelConfigPath = modelConfigPath

        self.modelConfig = loadConfig(self.modelConfigPath)

        self.modelName = self.modelConfig["model"].get("class_path").lower()

        self.postProcessorPath = self.modelConfig["model"]["init_args"].pop("post_processor_path",None)
        self.preProcessorPath = self.modelConfig["model"]["init_args"].pop("pre_processor_path",None)
        self.evaluatorPath = self.modelConfig["model"]["init_args"].pop("evaluator_path",None)

        print(f"Model {self.modelName} loaded: {self.modelConfig}")
        self.model = create_model(self.modelName, self.modelConfig["model"]["init_args"])
        print(f"Model {self.model} created.")

    def loadTrainedModel(self, folder:Path) -> None:
        """Load a trained model from a model.pt from a folder; Especially look for a model.pt file there. Then search for a config .yaml-file
        in that folder

        Arguments:
            folderPath -- Path to the folder of the trained model
        """
        # Find the model weights (the actual model) at the given folderPath
        modelPath:Path|None = find_first_file(folder, "model.pt")
        if modelPath is None:
            raise FileNotFoundError(f"model.pt not found in {folder}!")
        else:
            print(f"Found model weights at {modelPath}")
            self.modelPath = modelPath
        
        # Find the config to the model.pt at the given folderPath
        modelConfigPath = None
        for model in MODELS:
            modelConfigPath = find_first_file(folder, f"{model}.yaml")
        if modelConfigPath is None:
            FileNotFoundError(f"No .yaml config file found in {folder}!")
        else:
            print(f"Found model config at {modelConfigPath}")
            self.modelConfigPath = modelConfigPath
        
        self.generateModel(str(self.modelConfigPath))
        loadModelCallback = LoadModelCallback(weights_path=str(self.modelPath))
        self.callbacks["loadModel"] = loadModelCallback
        self.state |= modelFlags.hasModel
        print("Model loaded.")
        self.adjustOutputPath()

    def loadCheckpoint(self, path:Path):
        """Load model from a lightning Checkpoint file

        Arguments:
            path -- Path to the checkpoint file.
        """
        # TODO: Adapt for other model types; check path or ask for model type
        self.model = Padim.load_from_checkpoint(path)

    def loadDatasetFromDatabase(self, datasetName:str):
        if fo.dataset_exists(datasetName):
            # print(f"Dataset '{datasetName}' exists in database")
            self.dataset = fo.load_dataset(datasetName)
            if Split.TRAIN in list(self.dataset.tags):
                self.state |= modelFlags.hasTrainingData
            if Split.TEST in list(self.dataset.tags):
                self.state |= modelFlags.hasValidationData
            self.datasetName = datasetName
            self.categories = self.dataset.distinct("category.label")
            print(f"Loaded dataset '{datasetName}' from database!")
        else:
            print(f"Dataset '{datasetName}' does not exist in database")
        
    def loadDatasetFromDisk(self, datasetPath: Path, datasetName:str = "", overwrite:bool=False, merge:bool=True, split:Tuple[str,...] = ("train", "test")):
        if overwrite and merge:
            print("Overwrite and merge should not both be true. Overwrite is ignored...")
            overwrite = False

        elif not datasetPath.exists():
            print(f"Dataset {datasetPath} does not exit")
            exit(1)
        elif fo.dataset_exists(datasetName) and not overwrite and not merge:
            print(f"Dataset '{datasetName}' already exists in database")
            self.loadDatasetFromDatabase(datasetName)            
        elif fo.dataset_exists(datasetName) and overwrite:
            print(f"Dataset '{datasetName}' already exists in database")
            print("Overwriting")
            dataset, _ = importDataset(
                path=datasetPath,
                name=datasetName,
                overwrite=overwrite,
                split=split
            )
            self.dataset = dataset
            self.datasetName = datasetName
        elif fo.dataset_exists(datasetName) and merge:
            print(f"Dataset '{datasetName}' already exists in database")
            print("Merging")
            dataset, _ = importDataset(
                path=datasetPath,
                name=datasetName+"_"+str(self.now),
                overwrite=False,
                split=split
            )
            self.dataset = fo.load_dataset(datasetName)
            print(f"Loading {datasetName} from database for merging")
            self.dataset.merge_samples(dataset)
            self.datasetName = datasetName
        else:
            dataset, _ = importDataset(
                path=datasetPath,
                name=datasetName,
                overwrite=overwrite,
                split=split
            )
            self.dataset = dataset
            self.datasetName = datasetName

        if Split.TRAIN in list(self.dataset.tags):
            self.state |= modelFlags.hasTrainingData
        if Split.TEST in list(self.dataset.tags):
            self.state |= modelFlags.hasValidationData
        self.categories = self.dataset.distinct("category.label")

    def _setupDatamodule(self, datamoduleParams: dict[str, Any]) -> FODataModule:
        if self.dataset is not None:
            if self.datasetName is None:
                self.datasetName = "unnamedDataset"
            datamodule = FODataModule(name=self.datasetName, samples=self.dataset, root=self.outputPath, **datamoduleParams)
            datamodule.setup()
            self.datamodule = datamodule
        else:
            print("No dataset available")
            exit(1)
        if self.model is None:
            print("No model available")
            exit(1)
        return self.datamodule

    def generateEmbedding(self) -> None:
        """Generate an embedding of the dataset into a 2d space to visually inspect the data. Opens a voxel51 session"""
        if self.dataset is None:
            raise AttributeError("No dataset loaded. Please load or create a dataset first.")
        else:
            if not modelFlags.hasEmbedding in self.state:
                with exclude_from_logger():
                    clipEmbedding(self.dataset)
                print("Finished embedding computation.")
                print("Please reload the FiftyOne app to see the new visualizations.")
            else:
                print("Data already has embedding")
                print("You find the visualizations by clicking the '+' next to Samples and choosing Embeddings.")
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
        else:
            if (self.categories == None) or len(self.categories) == 0:
                raise AttributeError("There are no categories to select from")
            elif category not in self.categories:
                AttributeError(f"Category {category} not found! Available categories are: {self.categories}")
            else:
                print(f"Category found in categories for this dataset:")
                self.category = category

    #####
    # Logging
    #####

    def setupRunLogging(self, runName:str, runDir:Path, logFileName:str="run.log") -> Logger:
        logger = logging.getLogger(f"general.{runName}")
        logger.setLevel(logging.INFO)
        fileHandler = logging.FileHandler(os.path.join(runDir, logFileName))
        fileHandler.setLevel(logging.INFO)
        logFormater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fileHandler.setFormatter(logFormater)
        logger.addHandler(fileHandler)
        return logger

    def setupCallbacks(self, path:Path):
        # self.runDir:Path = Path(os.path.join(self.outputPath, runName, f"version_{version}"))
        # self.ckptDir = path / "checkpoints"
        if not os.path.exists(self.ckptDir):
            os.mkdir(self.ckptDir)

        checkpointCallback = ModelCheckpoint(
            dirpath=self.ckptDir,
            filename=self.ckptFileName,
            monitor="image_F1AdaptiveThreshold",  # val_loss not found?
            verbose=True,
            save_top_k=1,  # Save only the best model
            mode="min",  # Save the model with the minimum training loss,
            enable_version_counter=False
        )
        # self.ckptPath = self.ckptDir / self.ckptFileName
        
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
        if (tilingConfigPath.suffix == ".yaml") & tilingConfigPath.exists():
            with open(tilingConfigPath, "r") as file:
                self.tilingConfigDict:dict[str,Any] = yaml.safe_load(file)
        else:
            print()
            raise FileNotFoundError(f"{tilingConfigPath} file not found.")
        self.isTilingSetup = True

    def _checkBeforeTraining(self) -> bool:
        if self.model is None:
            raise AttributeError("Expected model attribute to be set")
        if self.dataset is None:
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
        if not self._checkBeforeTraining():
            exit(1)
        if not trainingConfigPath.suffix == ".yaml":
            FileNotFoundError("Config file needs to have .yaml suffix")
        if not trainingConfigPath.exists():
            FileNotFoundError("Error: .yaml config file not found in configs folder.")
        if tiling and not self.isTilingSetup:
            AttributeError("tiling is not setup: Call 'setupTiling()")

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
        self.setupCallbacks(self.outputPath)
        self.setupWandBLogger(self.runName, self.outputPath, self.version)
        if tiling:
            self._trainTiledModel(self.trainingConfig)
        else:
            self._trainSingleModel(self.trainingConfig)

    def _trainSingleModel(self, trainingConfig:dict[str, Any]):
        self._checkBeforeTraining()
        self.engineParams = {key: trainingConfig[key] for key in ENGINE_PARAMS if key in trainingConfig}
        self.datamoduleParams = {key: trainingConfig[key] for key in DATAMODULE_PARAMS if key in trainingConfig}
        self.datamodule = self._setupDatamodule(self.datamoduleParams)
        self.engine = Engine(callbacks=list(self.callbacks.values()), logger=self.runLogger, **self.engineParams)
        self.engine.fit(model=self.model, datamodule=self.datamodule)

        print("Running inference on dataset...")
        with exclude_from_logger():
            run_inference(self.dataset, self.engine, self.modelName)

        self.currentSession = fo.launch_app(self.dataset)

    def _trainTiledModel(self, config:dict[str, Any]):
        self._checkBeforeTraining()

        # Setup datamodule for the tiling 
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule = self._setupDatamodule(self.datamoduleParams)

        self.adjustOutputPath()
        self.tilingConfigDict = self.parseTilingConfig(self.tilingConfigPath)
        trainPipeline = TrainTiledEnsemble()
        trainPipeline.setDatamodule(datamodule=self.datamodule)
        if self.model:
            self.tilingConfigDict["TrainModels"]["model"]["class_path"] = "Padim"
            self.tilingConfigDict["TrainModels"]["model"]["init_args"] = self.modelConfig
        trainPipeline.run(self.tilingConfigDict, self.logFileNameGeneral)
        # trainPipeline.run(args)
        self.trainPipeline = trainPipeline
        return self.trainPipeline
    
    def eval(self, trainingConfigPath:Path, tiling:bool=False):
        if not self._checkBeforeTraining():
            exit(1)
        if not trainingConfigPath.suffix == ".yaml":
            FileNotFoundError("Config file needs to have .yaml suffix")
        if not trainingConfigPath.exists():
            FileNotFoundError("Error: .yaml config file not found in configs folder.")
        if tiling and not self.isTilingSetup:
            AttributeError("tiling is not setup: Call 'setupTiling()")

        with open(trainingConfigPath, 'r') as f:
            self.trainingConfig = yaml.safe_load(f) 
        if self.category is not None:
            self.runName = f"{self.modelName}-{self.datasetName}-{self.category}"
        else:
            self.runName =f"{self.modelName}-{self.datasetName}"

        self.setupCallbacks(self.outputPath)
        self.setupWandBLogger(self.runName, self.outputPath, self.version)
        if tiling:
            self._evalTiledModel(self.trainingConfig)
        else:
            self._evalSingleModel(self.trainingConfig)
        
    def _evalTiledModel(self, config:dict[str, Any]):
        # Setup datamodule for the tiling 
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule = self._setupDatamodule(self.datamoduleParams)

        self.adjustOutputPath()

        print("Running tiled ensemble test pipeline.")
        # pass the root dir from train run to load checkpoints
        self.tilingConfigDict = self.parseTilingConfig(self.tilingConfigPath)
        test_pipeline = EvalTiledEnsemble(self.tilingConfigDict["rootDir"])
        test_pipeline.setDatamodule(datamodule=self.datamodule)
        test_pipeline.run(self.tilingConfigDict, self.logFileNameGeneral)

    def _evalSingleModel(self, config:dict[str, Any]):
        self._checkBeforeTraining()
        self.engineParams = {key: config[key] for key in ENGINE_PARAMS if key in config}
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule = self._setupDatamodule(self.datamoduleParams)
        self.engine = Engine(callbacks=list(self.callbacks.values()), logger=self.runLogger, **self.engineParams)
        self.engine.predict(model=self.model, datamodule=self.datamodule)

        # print("Running inference on dataset...")
        # with exclude_from_logger():
        #     run_inference(self.dataset, self.engine, self.modelName)

        self.currentSession = fo.launch_app(self.dataset)

    def parseTilingConfig(self, path:Path):
        with Path(path).open(encoding="utf-8") as file:
            tilingConfigDict = yaml.safe_load(file)
        
        rootDir:str|None = tilingConfigDict.get("rootDir", None)
        if rootDir is None:
            tilingConfigDict["rootDir"] = self.outputPath
        else:
            if rootDir == "":
                tilingConfigDict["rootDir"] = self.outputPath

        ckptPath:str|None = tilingConfigDict.get("ckptPath", None)
        if ckptPath is None:
            tilingConfigDict["ckptPath"] = self.ckptPath
        else:
            if ckptPath == "":
                tilingConfigDict["ckptPath"] = self.ckptPath
        self.isTilingSetup = True
        return tilingConfigDict
        

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
            print("No model available")
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
            print(f"Session addess and port: {self.session.server_address}:{self.session.server_port}")
        else:
            print("Load a dataset first!")




if __name__ == "__main__":
    # Point FiftyOne at your MongoDB instance
    os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://localhost"
    os.environ["WANDB_API_KEY"] = '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'

    # Try to setup wandb
    wandb.login()
    iad = IAD()
    
    iad.generateModel("padim.yaml")
    # datasetPath1 = Path(os.path.join("datasets", "traintest"))
    # iad.loadDatasetFromDisk(datasetPath1, "traintest", overwrite=False, merge=False)
    # iad.launchSession()
    datasetPath2 = Path(os.path.join("datasets", "traintest"))
    iad.loadDatasetFromDisk(datasetPath2, "traintest", overwrite=True, merge=False)
    # iad.launchSession()
    # iad.loadDatasetFromDatabase("MVTecADShort")
    if iad.dataset is None:
        exit(1)
    iad.selectCategory("all")
    iad.adjustOutputPath()
    iad.copyFilesToOutputPath()
    #iad.launchSession()
    # iad.loadCheckpoint(Path("results/MVTecADShort/bottle/padim/checkpoints/best.ckpt"))
    # iad.setupTiling(Path("configs/TiledEnsemble.yaml"))
    # iad.train(Path("configs/padim_Training.yaml"), tiling=False)
    if iad.ckptPath is not None:
        iad.loadCheckpoint(iad.ckptPath)
        iad.eval(Path("configs/padim_Training.yaml"), tiling=False)
    # iad.exportResults(Path("EXPORT_TEST"))
    # iad.exportDataset(Path("EXPORT_TEST"))
    shutdown = False
    while not shutdown:
        userInput = input("shutdown?")
        if userInput:
            shutdown = True
        
    