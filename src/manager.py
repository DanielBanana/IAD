"""
Main class for managing the Industrial Anomaly Detection (IAD)
"""
# GENERAL
import wandb
import os
import yaml
import shutil
import copy
import datetime
import logging
import warnings
from logging.config import dictConfig
from enum import IntFlag, auto
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict

# FIFTYONE
import fiftyone.core.dataset as fod
import fiftyone as fo
import fiftyone.zoo as foz # zoo datasets and models
import fiftyone.brain as fob # ML methods
from fiftyone import ViewField as F # helper for defining views
from fiftyone import DatasetView


# ANOMALIB
from anomalib.deploy import ExportType
from anomalib.callbacks import LoadModelCallback
from anomalib.models.components import AnomalibModule
from anomalib.loggers import AnomalibWandbLogger
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.engine import Engine
from anomalib.data.utils import Split
from anomalib.data import PredictDataset
from anomalib.visualization import ImageVisualizer

# PYTROCH LIGHTNING
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import TQDMProgressBar

# OWN FILES
from .data.anomaly_datasets import importDataset, exportDataset, FODataModule, FODataset, importPredictDataset
from .setup import mapNameToModule, create_model
from .settings import MODELS, ENGINE_PARAMS, DATAMODULE_PARAMS
from .tiling.tiled_ensemble import TrainTiledEnsemble, EvalTiledEnsemble, PredTiledEnsemble
from .utils import find_first_file, exclude_from_logger, loadModelConfig

os.environ["TRUST_REMOTE_CODE"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="openvino.runtime")

class modelFlags(IntFlag):
    default = auto()
    hasModel = auto()
    hasTrainingData = auto()
    hasValidationData = auto()
    hasEmbedding = auto()

# Create the general logger
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetSession:
    """Manages FiftyOne dataset viewing and session lifecycle.
    Separate concern: how data is explored, not how models interact with it.
    """
    FO_Dataset: fo.Dataset
    FO_DatasetView: Optional[fo.DatasetView] = None
    AL_PredictDataset: Optional[PredictDataset] = None
    currentSession: Optional[fo.Session] = None  # fo.Session
    category: Optional[str] = None
    categories: Optional[List[str]] = None
    datasetName: Optional[str] = None

    @classmethod
    def loadDatasetFromDatabase(cls, dataset_name: str) -> "DatasetSession":
        """Load a dataset from MongoDB and track it here."""
        if fo.dataset_exists(dataset_name):
            FO_Dataset = fo.load_dataset(dataset_name)
            session:DatasetSession = cls(FO_Dataset)
            session.datasetName = dataset_name
            session.categories = session.FO_Dataset.distinct("category.label")
            logger.info(f"Loaded dataset '{dataset_name}' from database!")
            return session
        else:
            logger.error(f"Dataset '{dataset_name}' does not exist in database")
            raise FileNotFoundError()

    @classmethod
    def loadDatasetFromDisk(cls, datasetPath: Path, datasetName:str,  overwrite:bool=True, merge:bool=False, split:Tuple[str,...] = ("train", "test")) -> "DatasetSession":
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
        datasetSession: DatasetSession|None = None

        AL_PredictDataset:Optional[PredictDataset] = None


        if split == ("pred",):
            FO_Dataset, AL_PredictDataset = importPredictDataset(datasetPath, name=datasetName, overwrite=overwrite)
        else:
            if overwrite and merge:
                logger.info("Overwrite and merge should not both be true. Overwrite is ignored...")
                overwrite = False

            if not datasetPath.exists():
                raise FileNotFoundError(f"Dataset {datasetPath} does not exit")
            elif fo.dataset_exists(datasetName) and not overwrite and not merge:
                logger.info(f"Dataset '{datasetName}' already exists in database")
                logger.info("Loading from database")
                datasetSession = cls.loadDatasetFromDatabase(dataset_name=datasetName)  
                FO_Dataset = datasetSession.FO_Dataset  
            elif fo.dataset_exists(datasetName) and overwrite:
                logger.info(f"Dataset '{datasetName}' already exists in database")
                logger.info("Overwriting")
                FO_Dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName,
                    overwrite=overwrite,
                    split=split
                )
            elif fo.dataset_exists(datasetName) and merge:
                logger.info(f"Dataset '{datasetName}' already exists in FiftyOne database")
                logger.info(f"Importing '{datasetName}' dataset from disk")
                now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
                dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName+"_"+str(now),
                    overwrite=False,
                    split=split
                )
                logger.info(f"Importing '{datasetName}' dataset from FiftyOne database")
                FO_Dataset = fo.load_dataset(datasetName) # TODO Safety
                logger.info(f"Merging both datasets")
                FO_Dataset.merge_samples(dataset)
            else:
                logger.info(f"Loading {datasetName} dataset from disk")
                FO_Dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName,
                    overwrite=overwrite,
                    split=split
                )

        if datasetSession is not None:
            return datasetSession
        else:
            if FO_Dataset is not None:
                datasetSession = DatasetSession(FO_Dataset=FO_Dataset, AL_PredictDataset=AL_PredictDataset, categories=FO_Dataset.distinct("category.label"))
                logger.info(f"There are {datasetSession.FO_Dataset.count()} images in the {datasetName} dataset.")
                logger.info(f"There are {len(datasetSession.categories)} categorie(s) in the {datasetName} dataset.")
                logger.info(datasetSession.categories)
                datasetSession.datasetName = datasetName
                return datasetSession
            else:
                logger.warning(f"Import was not successfull")
                raise FileNotFoundError()

    def select_category(self, category: str) -> fo.DatasetView | None:
        """Select a category and create a view."""
        
        if category == "all":
            self.category = None
            self.FO_DatasetView = self.FO_Dataset.exists("file_path")
        else:
            if category not in (self.categories or []):
                raise AttributeError(f"Category {category} not found! Available: {self.categories}")
            self.category = category
            self.FO_DatasetView = self.FO_Dataset.filter_labels("category", F("label").is_in([category]))
        
        logger.info(f"Selected category: {self.category}, {len(self.FO_DatasetView)} images")
        return self.FO_DatasetView

    def launchSession(self) -> Any:
        """Launch the FiftyOne app for exploration."""

        if self.FO_DatasetView is not None: # Category is set which generates a datasetview
            # we make a separate dataset out of the View since some operations do not work on dataset views
            try:
                FO_Dataset_ = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
            except ValueError as e:
                logger.error(e)
                logger.info(f"Deleting {self.datasetName}-{self.category} from database and reloading.")
                fo.delete_dataset(f"{self.datasetName}-{self.category}")
                FO_Dataset_ = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
            finally:
                if isinstance(FO_Dataset_, fo.Dataset):
                    FO_Dataset = FO_Dataset_
                else:
                    raise ValueError("FO_Dataset_ is not bound. Check for problems with dataset/datasetView (category/product selection)")
        else:
            FO_Dataset = self.FO_Dataset


        self.currentSession = fo.launch_app(FO_Dataset)
        logger.info(f"Session: {self.currentSession.server_address}:{self.currentSession.server_port}")
        return self.currentSession

    def get_dataset_for_operation(self) -> fo.Dataset:
        """Returns the dataset to use for training/eval: the view if a category is selected, else full dataset."""
        if self.FO_DatasetView is not None:
            return self.FO_DatasetView
        if self.FO_Dataset is None:
            raise AttributeError("No dataset loaded")
        return self.FO_Dataset
    
    def setupDatamodule(self, datamoduleParams:dict[str,Any], outputPath:Path) -> FODataModule:
        """Setup a datamodule from the dataset for running a model on the data

        Arguments:
            outputPath: Path

        Returns:
            _description_
        """

        if self.datasetName is None:
            self.datasetName = "unnamedDataset"
        datamodule = FODataModule(name=self.datasetName, samples=self.FO_Dataset, root=outputPath, **datamoduleParams)
        datamodule.setup()
        self.datamodule = datamodule
        return self.datamodule
    
    def generateEmbedding(self) -> None:
        """Generate an embedding of the dataset into a 2d space to visually inspect the data. Opens a voxel51 session

        Raises:
            AttributeError: Needs a dataset to be set
        """

        with exclude_from_logger():
            clipEmbedding(self.FO_Dataset)
        logger.info("Finished embedding computation.")
        logger.info("Please reload the FiftyOne app to see the new visualizations.")
        logger.info("Data already has embedding")
        logger.info("You find the visualizations by clicking the '+' next to Samples and choosing Embeddings.")

    def save(self) -> None:
        """
        Saves the current status of the dataset to the database
        """

        logger.info(f"Saving current state of the dataset {self.datasetName} to the database")
        self.FO_Dataset.save()

class AnomalyDetectionManager:
    """Manages model lifecycle: creating, training, evaluating, exporting."""
    
    def __init__(self, 
                #  datasetSession:DatasetSession,
                 logFileName: str = "general.log", 
                 outputPath: Path = Path("results"),
                 configDir: Path = Path("configs"),
                 ) -> None:
        self.state: modelFlags = modelFlags.default
        # self.datasetSession = datasetSession
        # Model-specific state only
        self.modelConfigPath: Path | None = None
        self.preProcessorPath: Path | None = None
        self.postProcessorPath: Path | None = None
        self.evaluatorPath: Path | None = None
        self.visualizerPath: Path | None = None
        self.modelConfig: dict[str, Any] | None = None
        self.modelName: str | None = None
        self.model: AnomalibModule | None = None
        self.modelTrained: bool = False
        self.modelCopyPath:Path|None = None


        # Paths for checkpoints and outputs
        self.baseOutputPath: Path = outputPath
        self.outputPath: Path = outputPath
        self.configDir: Path = configDir
        self.ckptDir: Path | None = None
        self.ckptPath: Path | None = None

        
        # self.FO_Dataset:fo.Dataset|None = None
        # self.FO_DatasetView:fo.DatasetView|None = None
        # self.AL_Datamodule:FODataModule|None = None
        # self.AL_PredictDataset:PredictDataset|None = None
        # self.datasetName:str|None = None
        # self.datasetDir = datasetDir
        # self.datasetPath:Path|None = None
        # self.category:str|None = None
        # self.categories:List[str]|None = None

        # self.modelConfigPath:Path|None = None
        # self.preProcessorPath:Path|None = None
        # self.postProcessorPath:Path|None = None
        # self.evaluatorPath:Path|None = None
        # self.visualizerPath:Path|None = None
        # self.modelConfig:dict[str, Any]|None = None
        # self.modelName:str|None = None
        # self.model:AnomalibModule|None = None
        # self.modelTrained:bool = False
        # self.modelCopyPath:Path|None = None

        self.engine:Engine|None = None
        self.trainerPath: Path|None = None
        self.inferencerPath: Path|None = None

        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M")

        # general out folder like "results"; probably does not change often
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        self.baseOutpath:Path = outputPath
        self.outputPath:Path = outputPath
        self.logDir:Path = self.outputPath / "logs"
        self.logPath:Path = self.logDir / logFileName

        self.configDir:Path = configDir
        self.tilingConfigPath:Path|None = None
        self.isTilingSetup = False

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

        config_file = self.configDir / "Logging" / "logging.yaml"
        with open(config_file) as file:
            config:dict[str,Any] = yaml.safe_load(file)
        handlers:dict[str, dict[str,Any]]|None = config.get("handlers", None)
        if handlers is not None:
            for handlerName in handlers.keys():
                filename = handlers[handlerName].get("filename", None)
                if filename is not None:
                    config["handlers"][handlerName]["filename"] = self.logDir / filename
        dictConfig(config=config)

    def adjustPaths(self, datasetName: Optional[str]=None, category: Optional[str]=None, adjustCheckpoints:bool=True) -> Path:
        """This function should be called depending on the current values of dataset, model and the category for which the model is trained
        The idea is the following:

        Models are trained either for an entire dataset or a category of that dataset.
        -> If the dataset is added to the IAD object adjust path to include the dataset.                    outpath/dataset/
        -> If the model is trained for the entire dataset include the model in the path at this stage.      outpath/dataset/model/
        -> If a specific category is trained interject the category.                                        outpath/dataset/category/model
        """
        outputPath:Path = Path()

        if datasetName is not None:
            outputPath = self.baseOutpath / datasetName
        else:
            logger.info("Set dataset before adjusting path!")
            return self.outputPath
        if category is not None:
            outputPath = outputPath / category
        if self.modelName is not None:
            outputPath = outputPath / self.modelName
            if self.isTilingSetup:
                outputPath /= "tiled"
            if self.model is not None:
                if isinstance(self.model.visualizer, ImageVisualizer):
                    self.model.visualizer.output_dir = outputPath / "images"
            else:
                logger.info("No model set; Cant adjust path for visualiser (saving result images to disk)")

        if adjustCheckpoints:
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
        
        modelConfigPath = _configsDir / "Models" / configModelName
        if not os.path.exists(modelConfigPath):
            FileNotFoundError(f"Error: Config file {modelConfigPath} not found.")
        self.modelConfigPath = modelConfigPath

        self.modelConfig, self.preProcessorPath, self.postProcessorPath, self.evaluatorPath = loadModelConfig(configDir=self.configDir, modelConfigPath=modelConfigPath)

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

        logger.info(f"Model {self.modelName} loaded: {self.modelConfig}")
        self.model = create_model(self.modelName, self.modelConfig["model"]["init_args"])
        logger.info(f"Model {self.model} created.")
    
    def copyFilesToPath(self, path:Path):
        """To make it easier to understand how a model was trained copy all relevant files to the output folder.
        """
        # If the configs should be copied create the folder and copy the general config    
        if not path.exists():
            path.mkdir(parents=True)

        path = path / "configs"

        enginePath:Path = path / "Engine"
        modelsPath:Path = path / "Models"
        trainerPath:Path = path / "Trainer"
        tilingPath:Path = path / "Tiling"

        if not enginePath.exists():
            enginePath.mkdir(parents=True)
        if not modelsPath.exists():
            modelsPath.mkdir(parents=True)
        if not trainerPath.exists():
            trainerPath.mkdir(parents=True)
        if not tilingPath.exists():
            tilingPath.mkdir(parents=True)

        if self.trainerPath:
            _, fileName = os.path.split(self.trainerPath)
            shutil.copy2(self.trainerPath, trainerPath / fileName)

        # Copy the modelConfig file if possible
        if self.modelConfigPath is not None:
            _, configFileName = os.path.split(self.modelConfigPath)
            shutil.copy2(self.modelConfigPath, modelsPath / configFileName)
        else:
            raise AttributeError(f"Model config path is empty. Load model before calling this function!")
        
        if self.preProcessorPath is not None:
            _, fileName = os.path.split(self.preProcessorPath)
            shutil.copy2(self.preProcessorPath, enginePath / fileName)
        if self.postProcessorPath is not None:
            _, fileName = os.path.split(self.postProcessorPath)
            shutil.copy2(self.postProcessorPath, enginePath / fileName)
        if self.evaluatorPath is not None:
            _, fileName = os.path.split(self.evaluatorPath)
            shutil.copy2(self.evaluatorPath, enginePath / fileName)
        if self.visualizerPath is not None:
            _, fileName = os.path.split(self.visualizerPath)
            shutil.copy2(self.visualizerPath, enginePath / fileName)

    def copyFilesToOutputPath(self):
        self.copyFilesToPath(self.outputPath)
    
    @classmethod
    def loadProduct(cls, productConfigPath: Path, outputPath: Path, configDir: Path) -> Tuple["AnomalyDetectionManager", Dict[str,Any]]:
        with productConfigPath.open("r", encoding="utf-8") as f:
            productConfig:dict[Any, Any] = yaml.safe_load(f)
        manager = cls(
            productConfig["logging"]["logFileName"],
            outputPath=outputPath, 
            configDir=configDir,
            )
        
        manager.generateModel(f"{productConfig['model']['config']}", configDir)

        if productConfig["tiling"]["enable"]:
            manager.setupTiling(configDir / "Tiling" / productConfig["tiling"]["config"])
        manager.adjustPaths(datasetName=productConfig["dataset"]["name"], category=productConfig["dataset"]["category"])
        manager.trainerPath = configDir / "Trainer" / Path(productConfig["trainer"]["config"])
        manager.inferencerPath = configDir / "Trainer" / Path(productConfig["inferencer"]["config"])

        return manager, productConfig

    def setupTrainingCallbacks(self):
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
        logger.info(f"Setup tiling based on {self.tilingConfigPath}")
        return self.isTilingSetup

    def _check_before_training(self) -> AnomalibModule:
        """Just check the model — dataset validation is caller's responsibility."""
        if self.model is None:
            raise AttributeError("Expected model attribute to be set")
        return self.model
    
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

    def train(self,
              trainingConfigPath: Path,
              datasetSession:DatasetSession, 
              tiling: bool = False) -> None:
        """Train on a dataset passed in explicitly.
        
        Arguments:
            training_config_path -- Path to training config
            dataset -- The dataset to train on (caller manages which one)
            tiling -- Whether to use tiling
        """
        if not trainingConfigPath.suffix == ".yaml":
            raise FileNotFoundError("Config file needs .yaml suffix")
        if not trainingConfigPath.exists():
            raise FileNotFoundError("Config file not found")
        
        with open(trainingConfigPath, 'r') as f:
            trainingConfig:dict[str, Any] = yaml.safe_load(f)
        if datasetSession.category is not None:
            runName = f"{self.modelName}-{datasetSession.datasetName}-{datasetSession.category}"
        else:
            runName =f"{self.modelName}-{datasetSession.datasetName}"
        self.setupWandBLogger(runName, self.outputPath, self.version)
        self.adjustPaths(datasetName=datasetSession.datasetName, category=datasetSession.category)
        self.setupLogging()
        self.setupTrainingCallbacks()

        if tiling:
            self._trainTiledModel(datasetSession=datasetSession, trainingConfig=trainingConfig, modelConfig=self.modelConfig)
        else:
            self._trainSingleModel(datasetSession=datasetSession, trainingConfig=trainingConfig)

    def _trainSingleModel(self, datasetSession:DatasetSession, trainingConfig: dict[str, Any]) -> None:
        """Train without tiling."""
        model = self._check_before_training()
        
        self.engine_params = {key: trainingConfig[key] for key in ENGINE_PARAMS if key in trainingConfig}
        self.datamodule_params = {key: trainingConfig[key] for key in DATAMODULE_PARAMS if key in trainingConfig}
        
        # Dataset passed in explicitly — model doesn't own it
        datamodule:FODataModule = datasetSession.setupDatamodule(self.datamodule_params, self.outputPath)
        self.engine = Engine(callbacks=list(self.callbacks.values()), 
                            logger=self.modelLogger, **self.engine_params)
        self.copyFilesToOutputPath()
        self.engine.fit(model=model, datamodule=datamodule)
        
        logger.info("Training complete")

    def _trainTiledModel(self, datasetSession:DatasetSession, trainingConfig:dict[str,Any], modelConfig:dict[str,Any]|None):
        """Train a tiled model on the dataset based on a training config file.

        Arguments:
            config -- Config dictionary

        Returns:
            the pipeline for the training jobs
        """
        if modelConfig is not None:
            self.modelConfig = modelConfig
        
        datamoduleParams:dict[str,Any] = {key: trainingConfig[key] for key, value in trainingConfig.items() if key in DATAMODULE_PARAMS}
        trainerParams:dict[str,Any] = {key: trainingConfig[key] for key, value in trainingConfig.items() if key in ENGINE_PARAMS}
        datamodule = datasetSession.setupDatamodule(datamoduleParams, self.outputPath)
        gtAvail:bool = True if len(datasetSession.FO_Dataset.exists("ground_truth"))>0 else False

        # _:AnomalibModule
        # FO_Dataset:fo.Dataset 
        # _, FO_Dataset = self._checkBeforeTraining()
        # FO_Dataset:fo.Dataset|None = None
        if datasetSession.FO_DatasetView is not None: # Category is set which generates a datasetview
            # we make a separate dataset out of the View since some operations do not work on dataset views
            logger.info(f"Datasetview is used for training: {datasetSession.FO_DatasetView}")
            try:
                FO_Dataset_ = datasetSession.FO_DatasetView.clone(name=f"{datasetSession.datasetName}-{datasetSession.category}", persistent=False)
            except ValueError as e:
                logger.error(e)
                logger.info(f"Deleting {datasetSession.datasetName}-{datasetSession.category} from database and reloading.")
                fo.delete_dataset(f"{datasetSession.datasetName}-{datasetSession.category}")
                FO_Dataset_ = datasetSession.FO_DatasetView.clone(name=f"{datasetSession.datasetName}-{datasetSession.category}", persistent=False)
            finally:
                if isinstance(FO_Dataset_, fo.Dataset):
                    FO_Dataset = FO_Dataset_
                else:
                    raise ValueError("FO_Dataset_ is not bound. Check for problems with dataset/datasetView (category/product selection)")
        else:
            FO_Dataset = datasetSession.FO_Dataset

        logger.info(f"Dataset used for training: {FO_Dataset}")


        # self.adjustPaths()
        if self.tilingConfigPath is not None:
            self.setupTiling(self.tilingConfigPath)
        else:
            ValueError(f"self.tilingConfigPath is not set")


        trainPipeline = TrainTiledEnsemble(rootDir=self.outputPath,
                                           datamodule=datamodule,
                                           FO_Dataset=FO_Dataset,
                                           gtAvail=gtAvail)
        if self.modelConfig is not None:
            self.tilingConfigDict["model"] = self.modelConfig["model"]
        else:
            raise ValueError("modelConfig missing")
        self.tilingConfigDict["Trainer"] = trainerParams
        logger.info(f"Running tiled ensemble training pipeline with config: {self.tilingConfigDict}")
        trainPipeline.run(self.tilingConfigDict, "")

    def eval(self, datasetSession: DatasetSession, evalConfig:Path, tiling:bool=False, ckptDir:Path|None=None):
        """evaluate the current model on the current dataset

        Arguments: 
            config -- path to the config

        Keyword Arguments:
            tiling -- _description_ (default: {False})
        """
 
        if not evalConfig.suffix == ".yaml":
            FileNotFoundError("Config file needs to have .yaml suffix")
        if not evalConfig.exists():
            FileNotFoundError("Error: .yaml config file not found in configs folder.")
        if tiling and not self.isTilingSetup:
            AttributeError("tiling is not setup: Call 'setupTiling(tilingConfigPath)")

        with open(evalConfig, 'r') as f:
            self.evalConfig = yaml.safe_load(f) 
        if datasetSession.category is not None:
            self.runName = f"{self.modelName}-{datasetSession.datasetName}-{datasetSession.category}"
        else:
            self.runName =f"{self.modelName}-{datasetSession.datasetName}"
        self.adjustPaths(datasetName=datasetSession.datasetName, category=datasetSession.category)
        self.setupLogging()
        self.setupTrainingCallbacks()
        self.setupWandBLogger(self.runName, self.outputPath, self.version)

        if tiling:
            self._evalTiledModel(datasetSession, self.evalConfig, ckptDir=ckptDir)
        else:
            self._evalSingleModel(datasetSession, self.evalConfig)

    def _evalTiledModel(self, datasetSession: DatasetSession, evalConfig:dict[str, Any], ckptDir:Path|None=None):
        """Evaluate a tiled model on the dataset

        Arguments:
            config -- config dictionary
        """

        # Setup datamodule for the tiling 
        datamoduleParams = {key: evalConfig[key] for key in DATAMODULE_PARAMS if key in evalConfig}
        self.datamodule = datasetSession.setupDatamodule(datamoduleParams, self.outputPath)

        # self.adjustPaths()

        if datasetSession.FO_DatasetView is not None: # Category is set which generates a datasetview
            # we make a separate dataset out of the View since some operations do not work on dataset views
            try:
                FO_Dataset_ = datasetSession.FO_DatasetView.clone(name=f"{datasetSession.datasetName}-{datasetSession.category}", persistent=False)
            except ValueError as e:
                logger.error(e)
                logger.info(f"Deleting {datasetSession.datasetName}-{datasetSession.category} from database and reloading.")
                fo.delete_dataset(f"{datasetSession.datasetName}-{datasetSession.category}")
                FO_Dataset_ = datasetSession.FO_DatasetView.clone(name=f"{datasetSession.datasetName}-{datasetSession.category}", persistent=False)
            finally:
                if isinstance(FO_Dataset_, fo.Dataset):
                    FO_Dataset = FO_Dataset_
                else:
                    raise ValueError("FO_Dataset_ is not bound. Check for problems with dataset/datasetView (category/product selection)")
        else:
            FO_Dataset = datasetSession.FO_Dataset

        logger.info("Running tiled ensemble evaluation pipeline.")
        if self.isTilingSetup:
            if self.tilingConfigPath is not None:
                self.setupTiling(self.tilingConfigPath)
            else:
                logger.error(f"self.tilingConfigPath is not set even though isTilingSetup is true. This should not happen.")
                ValueError(f"self.tilingConfigPath is not set even though isTilingSetup is true. This should not happen.")
        else:
            logger.error(f"tiling is not setup")
            ValueError(f"tiling is not setup")

        evaluationPipeline = EvalTiledEnsemble(self.tilingConfigDict["rootDir"],
                                               FO_Dataset=FO_Dataset,
                                               datamodule=self.datamodule,
                                               ckptPath=ckptDir)
        # test_pipeline.setDatamodule(datamodule=self.datamodule)
        if self.modelConfig is not None:
            self.tilingConfigDict["model"] = self.modelConfig["model"]
        evaluationPipeline.run(self.tilingConfigDict, "")

    def _evalSingleModel(self, datasetSession:DatasetSession, evalConfig:dict[str, Any]):
        """Evaluate a singular model on the dataset

        Arguments:
            config -- config dictionary
        """
        model = self._check_before_training()
        self.engineParams = {key: evalConfig[key] for key in ENGINE_PARAMS if key in evalConfig}
        self.datamoduleParams = {key: evalConfig[key] for key in DATAMODULE_PARAMS if key in evalConfig}
        self.datamodule = datasetSession.setupDatamodule(self.datamoduleParams, self.outputPath)
        self.engine = Engine(callbacks=list(self.callbacks.values()), logger=self.runLogger, **self.engineParams)

        if datasetSession.FO_DatasetView is not None: # Category is set which generates a datasetview
            # we make a separate dataset out of the View since some operations do not work on dataset views
            try:
                FO_Dataset_ = datasetSession.FO_DatasetView.clone(name=f"{datasetSession.datasetName}-{datasetSession.category}", persistent=False)
            except ValueError as e:
                logger.error(e)
                logger.info(f"Deleting {datasetSession.datasetName}-{datasetSession.category} from database and reloading.")
                fo.delete_dataset(f"{datasetSession.datasetName}-{datasetSession.category}")
                FO_Dataset_ = datasetSession.FO_DatasetView.clone(name=f"{datasetSession.datasetName}-{datasetSession.category}", persistent=False)
            finally:
                if isinstance(FO_Dataset_, fo.Dataset):
                    FO_Dataset = FO_Dataset_
                else:
                    raise ValueError("FO_Dataset_ is not bound. Check for problems with dataset/datasetView (category/product selection)")
        else:
            FO_Dataset = datasetSession.FO_Dataset

        with exclude_from_logger():
            run_inference(FO_Dataset, self.engine, model, model.name)

    def inference(self,
                  datasetSession:DatasetSession, 
                  inferenceConfigPath:Path,
                  tiling:bool,
                  resultsDir:Path,
                  ckptPath:Path|None=None,
                  trainingDir:Path|None=None):
        """Run inference with the current model on the dataset

        Arguments: 
            config -- path to the config

        Keyword Arguments:
            tiling -- _description_ (default: {False})
        """
        if ckptPath is None:
            if self.ckptPath is not None:
                ckptPath = self.ckptPath.parent.resolve()
            else:
                logger.error("Neither ckptPath given nor self.ckptPath available.")
                return
        
        if not inferenceConfigPath.suffix == ".yaml":
            FileNotFoundError("Config file needs to have .yaml suffix")
        if not inferenceConfigPath.exists():
            FileNotFoundError("Error: .yaml config file not found in configs folder.")
        if tiling and not self.isTilingSetup:
            AttributeError("tiling is not setup: Call 'setupTiling(tilingConfigPath)")

        with open(inferenceConfigPath, 'r') as f:
            self.inferenceConfig = yaml.safe_load(f) 
        if datasetSession.category is not None:
            self.runName = f"{self.modelName}-{datasetSession.datasetName}-{datasetSession.category}"
        else:
            self.runName =f"{self.modelName}-{datasetSession.datasetName}"
        self.adjustPaths(datasetName=datasetSession.datasetName, category=datasetSession.category)
        self.setupLogging()
        self.setupTrainingCallbacks()
        self.setupWandBLogger(self.runName, self.outputPath, self.version)

        if tiling:
            self._inferenceTiledModel(datasetSession, self.inferenceConfig, ckptPath=ckptPath, resultsDir=resultsDir, trainingDir=trainingDir)
        else:
            self._evalSingleModel(datasetSession, self.inferenceConfig)

    def inferenceSingleImage(self,                             datasetName: str, 
                             datasetDir: Path,
                             category: str,
                             inferenceConfigPath:Path,
                             tiling:bool,
                             resultsDir:Path,
                             ckptPath:Path,
                             trainingDir:Path):
        """Run inference with the current model on the dataset

        Arguments: 
            config -- path to the config

        Keyword Arguments:
            tiling -- _description_ (default: {False})
        """
        split = ("pred",)
        # if ckptPath is None:
        #     if self.ckptPath is not None:
        #         ckptPath = self.ckptPath.parent.resolve()
        #     else:
        #         logger.error("Neither ckptPath given nor self.ckptPath available.")
        #         return
        
        if not inferenceConfigPath.suffix == ".yaml":
            FileNotFoundError("Config file needs to have .yaml suffix")
        if not inferenceConfigPath.exists():
            FileNotFoundError("Error: .yaml config file not found in configs folder.")
        if tiling and not self.isTilingSetup:
            AttributeError("tiling is not setup: Call 'setupTiling(tilingConfigPath)")

        with open(inferenceConfigPath, 'r') as f:
            self.inferenceConfig = yaml.safe_load(f) 
        self.runName = f"{self.modelName}-{datasetName}-{category}"
        self.adjustPaths(datasetName=datasetName, category=category, adjustCheckpoints=False)
        self.setupLogging()
        self.setupTrainingCallbacks()
        self.setupWandBLogger(self.runName, self.outputPath, self.version)

        datasetSession = DatasetSession.loadDatasetFromDisk(datasetDir/datasetName, datasetName, split=split)

        if tiling:
            self._inferenceTiledModel(datasetSession, self.inferenceConfig, ckptPath=ckptPath, resultsDir=resultsDir, trainingDir=trainingDir)
        else:
            self._evalSingleModel(datasetSession, self.inferenceConfig)

        
        

    def _inferenceTiledModel(self, datasetSession:DatasetSession, inferenceConfig:dict[str, Any], ckptPath:Path|None, resultsDir:Path, trainingDir:Path):
        """Evaluate a tiled model on the dataset

        Arguments:
            config -- config dictionary
        """
        _:AnomalibModule
        FO_Dataset:fo.Dataset 
        # _, FO_Dataset = self._checkBeforeTraining()

        if datasetSession.FO_DatasetView is not None: # Category is set which generates a datasetview
            # we make a separate dataset out of the View since some operations do not work on dataset views
            try:
                FO_Dataset_ = datasetSession.FO_DatasetView.clone(name=f"{datasetSession.datasetName}-{datasetSession.category}", persistent=False)
            except ValueError as e:
                logger.error(e)
                logger.info(f"Deleting {datasetSession.datasetName}-{datasetSession.category} from database and reloading.")
                fo.delete_dataset(f"{datasetSession.datasetName}-{datasetSession.category}")
                FO_Dataset_ = datasetSession.FO_DatasetView.clone(name=f"{datasetSession.datasetName}-{datasetSession.category}", persistent=False)
            finally:
                if isinstance(FO_Dataset_, fo.Dataset):
                    FO_Dataset = FO_Dataset_
                else:
                    raise ValueError("FO_Dataset_ is not bound. Check for problems with dataset/datasetView (category/product selection)")
        else:
            FO_Dataset = datasetSession.FO_Dataset

        # Setup datamodule for the tiling 
        datamoduleParams = {key: inferenceConfig[key] for key in DATAMODULE_PARAMS if key in inferenceConfig}
        self.datamodule:FODataModule = datasetSession.setupDatamodule(datamoduleParams, self.outputPath)
        
        self.adjustPaths()

        logger.info("Running tiled ensemble pred pipeline.")
        # pass the root dir from train run to load checkpoints
        if self.tilingConfigPath is not None:
            self.setupTiling(self.tilingConfigPath)
        else:
            ValueError(f"self.tilingConfigPath is not set")
        assert datasetSession.AL_PredictDataset is not None
        pred_pipeline = PredTiledEnsemble(self.tilingConfigDict["rootDir"],
                                          trainingDir=trainingDir,
                                          dataset=FO_Dataset,
                                          predictDataset=datasetSession.AL_PredictDataset,
                                          datamodule=self.datamodule,
                                          ckptDir=self.ckptDir)
        if self.model:
            modelConfig = copy.deepcopy(self.modelConfig)
            modelConfig["Evaluator"] = None
            self.tilingConfigDict["model"] = modelConfig["model"]
        
        self.tilingConfigDict["ckptPath"] = ckptPath
        logger.info(f"ckptPath: {ckptPath}")
        pred_pipeline.run(self.tilingConfigDict, "")
        self.FO_Dataset = FO_Dataset

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

    def launchSession(self):
        # TODO make View accepted; If we limit dataset to one category we get a view
        # launch app does not work with view
        # so we need a second dataset only with the class selected
        # call them dataset and subdataset
        if self.FO_Dataset is not None:
            self.session = fo.launch_app(self.FO_Dataset)
            logger.info("DatasetView (category selection) currently not supported. Shown dataset is the entire dataset.")
            logger.info(f"Session addess and port: {self.session.server_address}:{self.session.server_port}")
        else:
            logger.info("Load a dataset first!")










class AnomalyDetectionManager_():
    """Class managing the training and validation of the Industrial Anomaly Detection
    """

    def __init__(self, logFileName:str="general.log", outputPath:Path=Path("results"), configDir:Path=Path("configs"), datasetDir:Path=Path("datasets")) -> None:
        """
        Create the managing IAD class. Holds reference to Datasets, Models, Predictions, Settings, Paths

        Parameters
        ----------
        logFileName : str (optional)
            _description_. Default is `"general.log"`
        outputPath : Path (optional)
            Where should results like model weights or predictions be stored. Default is `Path("results")`
        configDir : Path (optional)
            Where do configurations for this session lay. Default is `Path("configs")`
        datasetDir : Path (optional)
            _description_. Default is `Path("datasets")`
        """
        self.state:modelFlags = modelFlags.default
        
        self.FO_Dataset:fo.Dataset|None = None
        self.FO_DatasetView:fo.DatasetView|None = None
        self.AL_Datamodule:FODataModule|None = None
        self.AL_PredictDataset:PredictDataset|None = None
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
        self.trainerPath: Path|None = None
        self.inferencerPath: Path|None = None

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

        config_file = self.configDir / "Logging" / "logging.yaml"
        with open(config_file) as file:
            config:dict[str,Any] = yaml.safe_load(file)
        handlers:dict[str, dict[str,Any]]|None = config.get("handlers", None)
        if handlers is not None:
            for handlerName in handlers.keys():
                filename = handlers[handlerName].get("filename", None)
                if filename is not None:
                    config["handlers"][handlerName]["filename"] = self.logDir / filename
        dictConfig(config=config)

    def adjustPaths(self) -> Path:
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
        
        modelConfigPath = _configsDir / "Models" / configModelName
        if not os.path.exists(modelConfigPath):
            FileNotFoundError(f"Error: Config file {modelConfigPath} not found.")
        self.modelConfigPath = modelConfigPath

        self.modelConfig, self.preProcessorPath, self.postProcessorPath, self.evaluatorPath = loadModelConfig(configDir=self.configDir, modelConfigPath=modelConfigPath)

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
        self.adjustPaths()

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

    def loadDatasetFromDatabase(self, datasetName:str) -> fo.Dataset|None:
        """Load a dataset from the voxel51 MongoDB

        Arguments:
            datasetName -- Name of the dataset as it was saved into the database
        """
        if fo.dataset_exists(datasetName):
            # logger.info(f"Dataset '{datasetName}' exists in database")
            self.FO_Dataset = fo.load_dataset(datasetName)
            if Split.TRAIN in list(self.FO_Dataset.tags):
                self.state |= modelFlags.hasTrainingData
            if Split.TEST in list(self.FO_Dataset.tags):
                self.state |= modelFlags.hasValidationData
            self.datasetName = datasetName
            self.categories = self.FO_Dataset.distinct("category.label")
            logger.info(f"Loaded dataset '{datasetName}' from database!")
        else:
            logger.info(f"Dataset '{datasetName}' does not exist in database")
        return self.FO_Dataset
        
    def loadDatasetFromDisk(self, datasetPath: Path, datasetName:str = "", overwrite:bool=True, merge:bool=False, split:Tuple[str,...] = ("train", "test")) -> fo.Dataset|None:
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
        if split == ("pred",):
            self.FO_Dataset, self.AL_PredictDataset = importPredictDataset(datasetPath, name=datasetName, overwrite=overwrite)
            self.datasetName = datasetName
        else:
            if overwrite and merge:
                logger.info("Overwrite and merge should not both be true. Overwrite is ignored...")
                overwrite = False
            elif not datasetPath.exists():
                FileNotFoundError(f"Dataset {datasetPath} does not exit")
            elif fo.dataset_exists(datasetName) and not overwrite and not merge:
                logger.info(f"Dataset '{datasetName}' already exists in database")
                logger.info("Loading from database")
                self.FO_Dataset = self.loadDatasetFromDatabase(datasetName)    
                self.datasetName = datasetName        
            elif fo.dataset_exists(datasetName) and overwrite:
                logger.info(f"Dataset '{datasetName}' already exists in database")
                logger.info("Overwriting")
                self.FO_Dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName,
                    overwrite=overwrite,
                    split=split
                )
                self.datasetName = datasetName
            elif fo.dataset_exists(datasetName) and merge:
                logger.info(f"Dataset '{datasetName}' already exists in FiftyOne database")
                logger.info(f"Importing '{datasetName}' dataset from disk")
                dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName+"_"+str(self.now),
                    overwrite=False,
                    split=split
                )
                logger.info(f"Importing '{datasetName}' dataset from FiftyOne database")
                self.FO_Dataset = fo.load_dataset(datasetName) # TODO Safety
                logger.info(f"Merging both datasets")
                self.FO_Dataset.merge_samples(dataset)
                self.datasetName = datasetName
            else:
                logger.info(f"Loading {datasetName} dataset from disk")
                self.FO_Dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName,
                    overwrite=overwrite,
                    split=split
                )
                self.datasetName = datasetName

        if self.FO_Dataset is not None:
            if Split.TRAIN in list(self.FO_Dataset.tags):
                self.state |= modelFlags.hasTrainingData
            if Split.TEST in list(self.FO_Dataset.tags):
                self.state |= modelFlags.hasValidationData
            self.categories = self.FO_Dataset.distinct("category.label")
            # self._setupDatamodule()
            logger.info(f"There are {self.FO_Dataset.count()} images in the {datasetName} dataset.")
            logger.info(f"There are {len(self.categories)} categorie(s) in the {datasetName} dataset.")
            logger.info(self.categories)
            return self.FO_Dataset
        else:
            logger.warning(f"Import was not successfull")
            return self.FO_Dataset

    def _setupDatamodule(self, dataset:fo.Dataset, datamoduleParams: dict[str, Any]) -> FODataModule:
        """Setup a datamodule from the dataset for running a model on the data

        Arguments:
            datamoduleParams -- _description_

        Returns:
            _description_
        """

        if self.datasetName is None:
            self.datasetName = "unnamedDataset"
        datamodule = FODataModule(name=self.datasetName, samples=dataset, root=self.outputPath, **datamoduleParams)
        datamodule.setup()
        self.datamodule = datamodule
        return self.datamodule

    def generateEmbedding(self) -> None:
        """Generate an embedding of the dataset into a 2d space to visually inspect the data. Opens a voxel51 session

        Raises:
            AttributeError: Needs a dataset to be set
        """

        if self.FO_Dataset is None:
            raise AttributeError("No dataset loaded. Please load or create a dataset first.")
        else:
            if not modelFlags.hasEmbedding in self.state:
                with exclude_from_logger():
                    clipEmbedding(self.FO_Dataset)
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

        path = path / "configs"

        enginePath:Path = path / "Engine"
        modelsPath:Path = path / "Models"
        trainerPath:Path = path / "Trainer"
        tilingPath:Path = path / "Tiling"

        if not enginePath.exists():
            enginePath.mkdir(parents=True)
        if not modelsPath.exists():
            modelsPath.mkdir(parents=True)
        if not trainerPath.exists():
            trainerPath.mkdir(parents=True)
        if not tilingPath.exists():
            tilingPath.mkdir(parents=True)

        if self.trainerPath:
            _, fileName = os.path.split(self.trainerPath)
            shutil.copy2(self.trainerPath, trainerPath / fileName)

        # Copy the modelConfig file if possible
        if self.modelConfigPath is not None:
            _, configFileName = os.path.split(self.modelConfigPath)
            shutil.copy2(self.modelConfigPath, modelsPath / configFileName)
        else:
            raise AttributeError(f"Model config path is empty. Load model before calling this function!")
        
        if self.preProcessorPath is not None:
            _, fileName = os.path.split(self.preProcessorPath)
            shutil.copy2(self.preProcessorPath, enginePath / fileName)
        if self.postProcessorPath is not None:
            _, fileName = os.path.split(self.postProcessorPath)
            shutil.copy2(self.postProcessorPath, enginePath / fileName)
        if self.evaluatorPath is not None:
            _, fileName = os.path.split(self.evaluatorPath)
            shutil.copy2(self.evaluatorPath, enginePath / fileName)
        if self.visualizerPath is not None:
            _, fileName = os.path.split(self.visualizerPath)
            shutil.copy2(self.visualizerPath, enginePath / fileName)

    def copyFilesToOutputPath(self):
        self.copyFilesToPath(self.outputPath)

    def selectCategory(self, category:str) -> fo.DatasetView|None:
        """Select a category from the dataset for which the model is trained.

        Keyword Arguments:
            category -- Choosen category; None -> all categories (default: {None})

        Returns:
            returns the category if it is available in the dataset
            returns None if category not found or dataset does not contain category
        """
        if category == "all":
            self.category = None
            logger.info("Selecting 'all' as category selects all categories")
            logger.info(f"There are {self.FO_Dataset.count()} images in the {self.datasetName} dataset")
            self.FO_DatasetView = self.FO_Dataset.exists("file_path")

        else:
            if (self.categories == None) or len(self.categories) == 0:
                raise AttributeError("There are no categories to select from")
            elif category not in self.categories:
                AttributeError(f"Category {category} not found! Available categories are: {self.categories}")
            else:
                logger.info(f"Category {self.category} found in categories for this dataset:")
                self.category = category
            self.FO_DatasetView = self.FO_Dataset.filter_labels("category", F("label").is_in([category]))
            logger.info(f"There are {len(self.FO_DatasetView)} images in the selected category '{category}' of the '{self.datasetName}' dataset")
        return self.FO_DatasetView
    
    @classmethod
    def loadProduct(cls, productConfig: Path, outputPath: Path, configDir: Path, datasetDir: Path) -> "AnomalyDetectionManager_":
        with productConfig.open("r", encoding="utf-8") as f:
            config:dict[Any, Any] = yaml.safe_load(f)
        
        manager = cls(config["logging"]["logFileName"],
                                      outputPath=outputPath, 
                                      configDir=configDir,
                                      datasetDir=datasetDir)
        
        manager.generateModel(f"{config['model']['config']}", configDir)
        datasetName:str = config["dataset"]["name"]
        datasetPath = datasetDir/datasetName
        manager.loadDatasetFromDisk(datasetPath, datasetName, overwrite=True, merge=False, split=tuple(config["dataset"]["split"]))
        manager.selectCategory(config["dataset"]["category"])
        if manager.FO_Dataset is None:
            raise RuntimeError("Dataset failed to load — check your dataset path and name.")
        if config["tiling"]["enable"]:
            manager.setupTiling(configDir / "Tiling" / config["tiling"]["config"])
        manager.adjustPaths()
        manager.trainerPath = configDir / "Trainer" / Path(config["trainer"]["config"])
        manager.inferencerPath = configDir / "Trainer" / Path(config["inferencer"]["config"])

        return manager

    def setupTrainingCallbacks(self):
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
        logger.info(f"Setup tiling based on {self.tilingConfigPath}")
        return self.isTilingSetup

    def _checkBeforeTraining(self) -> Tuple[AnomalibModule, fo.Dataset]:
        """Before the training can start check if a model and a dataset exist.

        Raises:
            AttributeError: model not found
            AttributeError: dataset not found

        Returns:
            model, FO_Dataset
        """
        if self.model is None:
            raise AttributeError("Expected model attribute to be set")
        if self.FO_Dataset is None:
                raise AttributeError("Expected data attribute to be set")
        
        return (self.model, self.FO_Dataset)

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
            self.trainerPath = trainingConfigPath
            self.trainingConfig = yaml.safe_load(f) 

        if self.category is not None:
            self.runName = f"{self.modelName}-{self.datasetName}-{self.category}"
        else:
            self.runName =f"{self.modelName}-{self.datasetName}"

        self.adjustPaths()
        self.setupLogging()
        self.setupTrainingCallbacks()
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
        model:AnomalibModule
        dataset:fo.Dataset 
        model, dataset = self._checkBeforeTraining()
        self.engineParams = {key: config[key] for key in ENGINE_PARAMS if key in config}
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule = self._setupDatamodule(dataset, self.datamoduleParams)
        self.engine = Engine(callbacks=list(self.callbacks.values()), logger=self.runLogger, **self.engineParams)
        self.copyFilesToOutputPath()
        self.engine.fit(model=model, datamodule=self.datamodule)

        logger.info("Running inference on dataset...")
        with exclude_from_logger():
            if model and self.modelName is not None:
                run_inference(self.FO_Dataset, self.engine, model, self.modelName)

        self.currentSession = fo.launch_app(dataset)

    def _trainTiledModel(self, config:dict[str, Any], modelConfig:dict[str,Any]|None=None):
        """Train a tiled model on the dataset based on a training config file.

        Arguments:
            config -- Config dictionary

        Returns:
            the pipeline for the training jobs
        """
        if modelConfig is not None:
            self.modelConfig = modelConfig

        _:AnomalibModule
        FO_Dataset:fo.Dataset 
        _, FO_Dataset = self._checkBeforeTraining()

        if self.FO_DatasetView is not None: # Category is set which generates a datasetview
            try:
                FO_Dataset = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
            except ValueError as e:
                logger.info(f"Deleting {self.datasetName}-{self.category} from database and reloading.")
                fo.delete_dataset(f"{self.datasetName}-{self.category}")
                FO_Dataset = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)


        # Setup datamodule for the tiling 
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule = self._setupDatamodule(FO_Dataset, self.datamoduleParams)

        self.adjustPaths()
        if self.tilingConfigPath is not None:
            self.setupTiling(self.tilingConfigPath)
        else:
            ValueError(f"self.tilingConfigPath is not set")

        gtAvail:bool = True if len(FO_Dataset.exists("ground_truth"))>0 else False

        trainPipeline = TrainTiledEnsemble(rootDir=self.outputPath,
                                           datamodule=self.datamodule,
                                           FO_Dataset=FO_Dataset,
                                           gtAvail=gtAvail)
        # trainPipeline.setDatamodule(datamodule=self.datamodule) # Split the dataset into different dataloaders for training, validation and test
        # trainPipeline.setFODataset(dataset=self.dataset) # The entire dataset with all samples. samples are tagged with split (train, val, test)
        if self.modelConfig is not None:
            self.tilingConfigDict["model"] = self.modelConfig["model"]
        else:
            raise ValueError("modelConfig missing")
        # logger.info(f"Running tiled ensemble training pipeline with config: {self.tilingConfigDict}")
        trainPipeline.run(self.tilingConfigDict, "")
        # trainPipeline.run(args)
        self.trainPipeline = trainPipeline
        self.FO_Dataset = FO_Dataset
        # return self.trainPipeline
    
    def predict(self, config:Path, tiling:bool, resultsDir:Path, ckptPath:Path|None=None):
        """predict with the current model on the dataset

        Arguments: 
            config -- path to the config

        Keyword Arguments:
            tiling -- _description_ (default: {False})
        """
        if ckptPath is None:
            if self.ckptPath is not None:
                ckptPath = self.ckptPath.parent.resolve()
            else:
                logger.error("Neither ckptPath given nor self.ckptPath available.")
                return
        
        if not config.suffix == ".yaml":
            FileNotFoundError("Config file needs to have .yaml suffix")
        if not config.exists():
            FileNotFoundError("Error: .yaml config file not found in configs folder.")
        if tiling and not self.isTilingSetup:
            AttributeError("tiling is not setup: Call 'setupTiling(tilingConfigPath)")

        with open(config, 'r') as f:
            self.trainingConfig = yaml.safe_load(f) 
        if self.category is not None:
            self.runName = f"{self.modelName}-{self.datasetName}-{self.category}"
        else:
            self.runName =f"{self.modelName}-{self.datasetName}"

        self.adjustPaths()
        self.setupLogging()
        self.setupTrainingCallbacks()
        self.setupWandBLogger(self.runName, self.outputPath, self.version)
        if tiling:
            self._predictTiledModel(self.trainingConfig, ckptPath=ckptPath, resultsDir=resultsDir)
        else:
            self._evalSingleModel(self.trainingConfig)
        
    def eval(self, evalConfig:Path, tiling:bool=False, ckptDir:Path|None=None):
        """evaluate the current model on the current dataset

        Arguments: 
            config -- path to the config

        Keyword Arguments:
            tiling -- _description_ (default: {False})
        """
 
        if not evalConfig.suffix == ".yaml":
            FileNotFoundError("Config file needs to have .yaml suffix")
        if not evalConfig.exists():
            FileNotFoundError("Error: .yaml config file not found in configs folder.")
        if tiling and not self.isTilingSetup:
            AttributeError("tiling is not setup: Call 'setupTiling(tilingConfigPath)")

        with open(evalConfig, 'r') as f:
            self.evalConfig = yaml.safe_load(f) 
        if self.category is not None:
            self.runName = f"{self.modelName}-{self.datasetName}-{self.category}"
        else:
            self.runName =f"{self.modelName}-{self.datasetName}"

        self.adjustPaths()
        self.setupLogging()
        self.setupTrainingCallbacks()
        self.setupWandBLogger(self.runName, self.outputPath, self.version)
        if tiling:
            self._evalTiledModel(self.evalConfig, ckptDir=ckptDir)
        else:
            self._evalSingleModel(self.evalConfig)

    

    def _evalTiledModel(self, evalConfig:dict[str, Any], ckptDir:Path|None=None):
        """Evaluate a tiled model on the dataset

        Arguments:
            config -- config dictionary
        """

        _:AnomalibModule
        FO_Dataset:fo.Dataset 
        _, FO_Dataset = self._checkBeforeTraining()

        self._checkBeforeTraining()
        # Setup datamodule for the tiling 
        self.datamoduleParams = {key: evalConfig[key] for key in DATAMODULE_PARAMS if key in evalConfig}
        self.datamodule = self._setupDatamodule(FO_Dataset, self.datamoduleParams)

        self.adjustPaths()

        logger.info("Running tiled ensemble evaluation pipeline.")
        if self.isTilingSetup:
            if self.tilingConfigPath is not None:
                self.setupTiling(self.tilingConfigPath)
            else:
                logger.error(f"self.tilingConfigPath is not set even though isTilingSetup is true. This should not happen.")
                ValueError(f"self.tilingConfigPath is not set even though isTilingSetup is true. This should not happen.")
        else:
            logger.error(f"tiling is not setup")
            ValueError(f"tiling is not setup")

        evaluationPipeline = EvalTiledEnsemble(self.tilingConfigDict["rootDir"],
                                               FO_Dataset=self.FO_Dataset,
                                               datamodule=self.datamodule,
                                               ckptPath=ckptDir)
        # test_pipeline.setDatamodule(datamodule=self.datamodule)
        if self.modelConfig is not None:
            self.tilingConfigDict["model"] = self.modelConfig["model"]
        evaluationPipeline.run(self.tilingConfigDict, "")

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
                run_inference(self.FO_Dataset, self.engine, self.model, self.modelName)
            else:
                AttributeError("Need self.model and self.modelName")
        if self.FO_Dataset is not None:
            self.currentSession = fo.launch_app(self.FO_Dataset)

    def _predictTiledModel(self, config:dict[str, Any], ckptPath:Path|None, resultsDir:Path):
        """Evaluate a tiled model on the dataset

        Arguments:
            config -- config dictionary
        """
        _:AnomalibModule
        FO_Dataset:fo.Dataset 
        _, FO_Dataset = self._checkBeforeTraining()

        if self.FO_DatasetView is not None: # Category is set which generates a datasetview
            try:
                FO_Dataset:fo.Dataset = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
            except ValueError as e:
                logger.info(f"Deleting {self.datasetName}-{self.category} from database and reloading.")
                fo.delete_dataset(f"{self.datasetName}-{self.category}")
                FO_Dataset:fo.Dataset = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
        logger.debug(f"FO_Dataset: \n {FO_Dataset}")

        # Setup datamodule for the tiling 
        self.datamoduleParams = {key: config[key] for key in DATAMODULE_PARAMS if key in config}
        self.datamodule:FODataModule = self._setupDatamodule(FO_Dataset, self.datamoduleParams)
        
        self.adjustPaths()

        logger.info("Running tiled ensemble pred pipeline.")
        # pass the root dir from train run to load checkpoints
        if self.tilingConfigPath is not None:
            self.setupTiling(self.tilingConfigPath)
        else:
            ValueError(f"self.tilingConfigPath is not set")
        assert self.AL_PredictDataset is not None
        pred_pipeline = PredTiledEnsemble(self.tilingConfigDict["rootDir"],
                                          trainingDir=resultsDir,
                                          dataset=FO_Dataset,
                                          predictDataset=self.AL_PredictDataset,
                                          datamodule=self.datamodule,
                                          ckptDir=self.ckptDir)
        if self.model:
            modelConfig = copy.deepcopy(self.modelConfig)
            modelConfig["Evaluator"] = None
            self.tilingConfigDict["model"] = modelConfig["model"]
        
        self.tilingConfigDict["ckptPath"] = ckptPath
        logger.info(f"ckptPath: {ckptPath}")
        pred_pipeline.run(self.tilingConfigDict, "")
        self.FO_Dataset = FO_Dataset

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

    # def exportDataset(self, exportPath:Path) -> None:
    #     exportDataset(self., exportPath)
    
    #####
    # Fiftyone Features
    #####

    def launchSession(self):
        # TODO make View accepted; If we limit dataset to one category we get a view
        # launch app does not work with view
        # so we need a second dataset only with the class selected
        # call them dataset and subdataset
        if self.FO_Dataset is not None:
            self.session = fo.launch_app(self.FO_Dataset)
            logger.info("DatasetView (category selection) currently not supported. Shown dataset is the entire dataset.")
            logger.info(f"Session addess and port: {self.session.server_address}:{self.session.server_port}")
        else:
            logger.info("Load a dataset first!")


def clipEmbedding(dataset:fod.Dataset):
    model = foz.load_zoo_model(
        "clip-vit-base32-torch"
    )  # load the CLIP model from the zoo

    # Compute embeddings for the dataset
    dataset.compute_embeddings(
        model=model, embeddings_field="clip_embeddings", batch_size=64
    )

    # Dimensionality reduction using UMAP on the embeddings
    fob.compute_visualization(
        dataset, embeddings="clip_embeddings", method="umap", brain_key="clip_vis"
    )

def resnetEmbedding(dataset:fod.Dataset):
    model = foz.load_zoo_model(
        "resnet50-imagenet-torch"
    )  # load the ResNet50 model from the zoo

    # Compute embeddings for the dataset — this might take a while on a CPU
    dataset.compute_embeddings(model=model, embeddings_field="resnet50_embeddings")

    # Dimensionality reduction using UMAP on the embeddings
    fob.compute_visualization(
        dataset,
        embeddings="resnet50_embeddings",
        method="umap",
        brain_key="resnet50_vis",
    )

def run_inference(sample_collection:fo.Dataset, engine: Engine, model:AnomalibModule, key:str):
    for sample in sample_collection.iter_samples(autosave=True, progress=True):
        output = engine.predict(data_path=sample.filepath, model=model)[0]
        
        conf = output.pred_score.item()
        anomaly = "anomaly" if output.pred_label else "normal"

        sample[f"pred_anomaly_score_{key}"] = conf
        sample[f"pred_anomaly_{key}"] = fo.Classification(label=anomaly)
        sample[f"pred_anomaly_map_{key}"] = fo.Heatmap(map=output.anomaly_map.data.numpy().squeeze()*255, range=[0,255])
        sample[f"pred_defect_mask_{key}"] = fo.Segmentation(mask=output.pred_mask.data.numpy().squeeze().astype(np.int16)*255)

if __name__ == "__main__":
    # Point FiftyOne at your MongoDB instance
    os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://localhost"
    os.environ["WANDB_API_KEY"] = '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'

    # Try to setup wandb
    wandb.login()

    logger = logging.getLogger("logger")
    logger.debug("This should go to mongodb.log")

    tiling      = True
    modelName   = "padim"
    modelConfig = "PadimNoVal"
    modelTrainConfig = "Training_padimNoVal"
    datasetName = "onlyGood"
    split       = ("train","test")
    category    = "bottle"
    train       = True
    evaluate    = True
    predict     = True
    datasetDir  = Path("datasets/")
    configDir   = Path("configs/")
    outputPath  = Path("results/")
    # ckptPath    = Path("../results/MVTecADShort/cable/padim/tiled/checkpoints")
    manager = AnomalyDetectionManager()

    manager.generateModel(f"{modelName}.yaml")
    datasetPath = Path(os.path.join("datasets", datasetName))
    manager.loadDatasetFromDisk(datasetPath, datasetName, overwrite=True, merge=False)
    manager.selectCategory(category)
    manager.adjustPaths()
    if manager.FO_Dataset is None:
        exit(1)
    if tiling:
        manager.setupTiling(configDir / Path("Tiling/TiledEnsemble.yaml"))
    if train:
        manager.train(configDir / Path(f"Trainer/Training_{modelName}.yaml"), tiling=tiling)
    # manager.launchSession()
    if evaluate:
        if manager.ckptPath is not None:
            if not tiling:
                manager.loadCheckpoint(manager.ckptPath, f"{modelName}")
            manager.eval(configDir / "Trainer" / "Eval.yaml", tiling=tiling)
        manager.launchSession()

    ckptPath = manager.ckptPath.parent.resolve()
    print(ckptPath)
    manager.loadDatasetFromDisk(datasetPath=datasetDir / "MVTecADShortPred", datasetName="cablePred35", split=("pred",), overwrite=True)
    if manager.ckptPath is not None:
        if not tiling:
            manager.loadCheckpoint(manager.ckptPath, f"{modelName}")
        if tiling:
            manager.setupTiling(configDir / "Tiling" / "TiledEnsemblePred.yaml")
        manager.predict(config=configDir / "Trainer" / "Predict.yaml", tiling=tiling, ckptPath=ckptPath)
        print(manager.FO_Dataset)
        manager.launchSession()


    shutdown = False
    while not shutdown:
        userInput = input("shutdown?\n")
        if userInput:
            shutdown = True
        
    