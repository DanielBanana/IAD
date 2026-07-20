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
import logging.config
import sys
import warnings
from logging.config import dictConfig
from enum import IntFlag, auto
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict, Type
from dataclasses import dataclass
from typing import Optional
from jsonargparse import ArgumentParser, Namespace
import hashlib
import json
import uuid
from datetime import datetime

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
from data.anomaly_datasets import importDataset, exportDataset, FODataModule, FODataset, importPredictDataset
from setup import mapNameToModule, create_model
from settings import MODELS, ENGINE_PARAMS, DATAMODULE_PARAMS
from tiling.tiled_ensemble import TrainTiledEnsemble, EvalTiledEnsemble, PredTiledEnsemble
from tiling.tilingCheckpoints import checkTiledCheckpointsExist
from utils import find_first_file, exclude_from_logger
from userConfigs import (
    DatasetConfig, 
    DataModuleConfig,
    TrainerConfig, 
    ModelConfig, 
    BaseModelConfig, 
    TilingPipelineConfig, 
    Product, loadProductFromYaml, 
    DatasetSession, 
    loadModelConfig,
    TilingPipelineConfig
)
from run_registry import generate_run_id, serialize_effective_config, write_run_manifest, RunConfigFiles

os.environ["TRUST_REMOTE_CODE"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="openvino.runtime")

# --- pure path resolution, no side effects, independently testable ---

def resolve_output_dir(
    baseOutputDir: Path,
    datasetName: str,
    modelName: str,
    runId: str,
    category: Optional[str] = None,
    tiling: bool = True,
) -> Path:
    """
    Where new artifacts for a run should be written: base/dataset/[category]/model/[tiled


    Parameters
    ----------
    baseOutputDir : Path
        Base of the output directories usually .../results
    datasetName : str
        Name of the dataset to save the results for
    modelName : str
        Name of the model to save the results for
    category : Optional[str] (optional)
        Name of the category in the dataset the results are fore. Default is `None`
    tiling : bool (optional)
        Is a TiledEnsemble model used? I.e. is there a model for each tile being trained because the images are too large. Default is `True`

    Returns
    -------
    _name_ : Path
        The path where results are stored
    """
    path = baseOutputDir / datasetName
    if category is not None:
        path = path / category
    path = path / modelName
    if tiling:
        path = path / "tiled"
    return path / "runs" / runId


def resolve_checkpoint_paths(
    trainingDir: Path,
    # ckptFileName: str = "best",
    # ckptSuffix: str = ".ckpt",
) -> Path:
    """
    Where checkpoints live *within* a given (already-resolved) training run directory.

    Deliberately takes an explicit `trainingDir` rather than recomputing it —
    # inference reads checkpoints from a *different* prior run than the one it writes to.

    Parameters
    ----------
    trainingDir : Path
        Ouput_dir usualy given by resolve_output_dir
    ckptFileName : str (optional)
        Name of the file. Default is `"best"`
    ckptSuffix : str (optional)
        Suffix of the file. Default is `".ckpt"`

    Returns
    -------
    _name_ : Path
        Direcetory of the checkpoints
    _name_ : Path
        File if not tiled checkpoints; if tiled just use the dir because there are multiple files needed
    """
    ckptDir = trainingDir / "checkpoints"
    return ckptDir
    # ckptPath = ckptDir / (ckptFileName + ckptSuffix)
    # return ckptDir, ckptPath

@dataclass
class RunContext:
    runName: str
    outputDir: Path
    # ckptDir: Path
    # ckptPath: Path

class ManagerError(Exception):
    """Base class for all AnomalyDetectionManager errors."""

class ManagerStateError(ManagerError):
    """Raised when an action is requested before its prerequisites are met.
    Carries `missing` so a frontend can render actionable messages without
    parsing the exception text.
    """
    def __init__(self, message: str, missing: Optional[List[str]] = None):
        super().__init__(message)
        self.missing = missing or []

class NoModelLoadedError(ManagerStateError): ...
class NoDatasetLoadedError(ManagerStateError): ...
class TilingNotConfiguredError(ManagerStateError): ...
class ModelNotTrainedError(ManagerStateError): ...
class CheckpointNotFoundError(ManagerStateError): ...

class ConfigError(ManagerError):
    """Raised for malformed or unresolvable configuration (YAML, paths, etc.)."""

class ManagerState(IntFlag):
    NONE = 0
    MODEL_LOADED = auto()
    DATASET_LOADED = auto()
    TILING_CONFIGURED = auto()
    RUN_PREPARED = auto()      # paths resolved, callbacks/logger set up
    TRAINED = auto()           # a train() has completed for the current model+dataset
    CHECKPOINT_AVAILABLE = auto()

# Create the general logger
logger = logging.getLogger(__name__)

class AnomalyDetectionManager:

    # Each action declares what state it needs. Single source of truth —
    # used both to raise clear errors and to answer "can I do X yet?"
    _REQUIREMENTS: Dict[str, ManagerState] = {
        "setupTiling": ManagerState.MODEL_LOADED,
        "train": ManagerState.MODEL_LOADED | ManagerState.DATASET_LOADED | ManagerState.TILING_CONFIGURED,
        "eval": ManagerState.MODEL_LOADED | ManagerState.DATASET_LOADED | ManagerState.TILING_CONFIGURED | ManagerState.CHECKPOINT_AVAILABLE,
        "inference": ManagerState.MODEL_LOADED | ManagerState.DATASET_LOADED | ManagerState.TILING_CONFIGURED | ManagerState.CHECKPOINT_AVAILABLE
    }

    _STATE_DESCRIPTIONS: Dict[ManagerState, str] = {
        ManagerState.MODEL_LOADED: "Load a model with generateModel()",
        ManagerState.DATASET_LOADED: "Load a dataset (pass a DatasetSession)",
        ManagerState.TILING_CONFIGURED: "Configure tiling with setupTiling()",
        ManagerState.TRAINED: "Train the model with train() before evaluating",
        ManagerState.CHECKPOINT_AVAILABLE: "No checkpoint found — train first or point at an existing training run",
    }

    _STATE_ERROR_CLASSES: Dict[ManagerState, type[ManagerStateError]] = {
        ManagerState.MODEL_LOADED: NoModelLoadedError,
        ManagerState.DATASET_LOADED: NoDatasetLoadedError,
        ManagerState.TILING_CONFIGURED: TilingNotConfiguredError,
        ManagerState.TRAINED: ModelNotTrainedError,
        ManagerState.CHECKPOINT_AVAILABLE: CheckpointNotFoundError,
    }

    # Priority order when several things are missing at once — report the
    # earliest step in the pipeline first, since fixing it is usually a
    # prerequisite for the others anyway.
    _STATE_PRIORITY: List[ManagerState] = [
        ManagerState.MODEL_LOADED,
        ManagerState.DATASET_LOADED,
        ManagerState.TILING_CONFIGURED,
        ManagerState.TRAINED,
        ManagerState.CHECKPOINT_AVAILABLE,
    ]
        
    def __init__(self, 
                 outputDir: Path = Path("results"),
                 configDir: Path = Path("configs"),
                 ) -> None:
        
        self.state: ManagerState = ManagerState.NONE

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        self.baseOutputDir:Path = outputDir
        self.outputDir:Path = outputDir
        self.configDir:Path = configDir

        self.ckptDir: Optional[Path] = None
        # self.ckptPath: Optional[Path] = None
        # self.ckptFileName:str = "best"
        # self.ckptSuffix:str = ".ckpt"

        self.model: Optional[AnomalibModule] = None
        self.modelConfig: Optional[ModelConfig] = None
        self.modelConfigPath: Optional[Path] = None
        self.modelTrainingDir: Optional[Path] = None
        # self.modelCallbacks: Dict[str, Callback]

        # self.engine: Optional[Engine] = None
        self.trainerConfig: Optional[TrainerConfig] = None
        self.trainerConfigPath: Optional[Path] = None
        self.inferencerConfig: Optional[TrainerConfig] = None
        self.inferencerConfigPath: Optional[Path] = None

        self.tilingConfigPath: Optional[Path] = None
        self.tilingPipelineConfig: Optional[TilingPipelineConfig] = None
        self.isTilingSetup = False

        # self.version:int = 0
        # self.versionName:str = "version"
        # self.runDir:Path|None = None

        self.datasetSession: Optional[DatasetSession] = None
  
        self.setupLogging()

    def attachDatasetSession(self, datasetSession: DatasetSession) -> None:
        """
        Attach a DatasetSession instance to the current manager and set the appropriate state flag.
        This allows the manager to acces things like the category we train on for adjusting paths and such

        Parameters
        ----------
        datasetSession : DatasetSession
            DatasetSession instance with the interesting stats
        """
        self.datasetSession = datasetSession
        self.state |= ManagerState.DATASET_LOADED
        logger.info(f"Attached dataset '{datasetSession.datasetName}' (category={datasetSession.category})")

    def _resolve_dataset_session(self, datasetSession: Optional[DatasetSession]) -> DatasetSession:
        if datasetSession is not None:
            self.attachDatasetSession(datasetSession)
        if self.datasetSession is None:
            raise NoDatasetLoadedError(
                "No dataset attached. Call attachDatasetSession() or pass datasetSession=...",
                missing=["A dataset attached via attachDatasetSession() or a datasetSession argument"],
            )
        return self.datasetSession

    def now(self):
        """
        Return a format Date and time string for saving things to disk

        Returns
        -------
        dateAndTime : str
            formated date and time string of the current time
        """
        dateAndTime: str = datetime.now().strftime("%Y%m%d-%H%M")
        return dateAndTime


    def setupLogging(self, logDir: Optional[Path] = None, logConfigFile: Optional[Path] = None):
        """
        Prepare the logging for the current Manager session by creating a log directory, removing old logs and loading the logging
        config from a given file

        Parameters
        ----------
        logDir : Optional[Path] (optional)
            _description_. Default is `None`
        logConfigFile : Optional[Path] (optional)
            _description_. Default is `None`
        """


        if logDir is None:
            self.logDir = self.outputDir / "logs"
        else:
            self.logDir = logDir

        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)
        # else:
        #     shutil.rmtree(self.logDir) # TODO Make this safer
        #     os.makedirs(self.logDir)

        if logConfigFile is None:
            logConfigFile = self.configDir / "Logging" / "logging.yaml"
        
        exists = Path.exists(logConfigFile)
        if not exists:
            self._setupDefaultLogger()
        else:
            with open(logConfigFile) as file:
                config:Dict[str,Any] = yaml.safe_load(file)
            handlers:Dict[str, Dict[str,Any]]|None = config.get("handlers", None)
            if handlers is not None:
                for handlerName in handlers.keys():
                    filename = handlers[handlerName].get("filename", None)
                    if filename is not None:
                        config["handlers"][handlerName]["filename"] = self.logDir / filename
            dictConfig(config=config)

    def _setupDefaultLogger(self):
        """
        If the given logging file does not exist for whatever reaseon we create this default logger.

        Returns
        -------
        logger : logger.Logger
            The default logger
        """
        logging_config:Dict[str,Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {
                    "format": "%(levelname)s: %(message)s"
                },
                "detailed": {
                    "format": "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S%z"
                }
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": sys.stdout
                },
                "stderr": {
                    "class": "logging.StreamHandler",
                    "level": "WARNING",
                    "formatter": "simple",
                    "stream": sys.stderr
                },
                "infoFile": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                    "filename": "info.log",
                    "maxBytes": 10000000,
                    "backupCount": 3
                },
                "debugFile": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "debug.log",
                    "maxBytes": 10000000,
                    "backupCount": 3
                }
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["stderr", "stdout", "debugFile", "infoFile"]
            }
        }
        logging.config.dictConfig(logging_config)
        logger = logging.getLogger()
        return logger

    def _apply_visualizer_output_dir(self, outputDir: Path) -> None:
        """
        Change the output directory for the visualizer according to the current outputDir

        Parameters
        ----------
        outputDir : Path
            The current outputDir where the images with the training results should be returned to
        """
        if self.model is not None and isinstance(self.model.visualizer, ImageVisualizer):
            self.model.visualizer.output_dir = outputDir / "images"
        else:
            logger.info("No model set; cannot adjust visualizer output_dir.")

    def has_state(self, flag: ManagerState) -> bool:
        """
        Check if the manager state has a specific state like CHECKPOINT_AVAILABLE

        Parameters
        ----------
        flag : ManagerState
            ManagerState IntFlag to check

        Returns
        -------
        _name_ : bool
            Flag contained in state? True or False
        """
        return flag in self.state
    
    def get_missing_requirements(self, action: str) -> List[str]:
        """
        Human-readable list of what's missing before `action` can run.
        Empty list means the action is currently valid.

        Parameters
        ----------
        action : str
            action we want to perform

        Returns
        -------
        _name_ : List[str]
            List of States that are missing for the given action as description for reading

        Raises
        ------
        _name_ : ValueError
            If the name of the action is unknown. See AnomalyDetectionManager._REQUIREMENTS
        """
        required = self._REQUIREMENTS.get(action)
        if required is None:
            raise ValueError(f"Unknown action '{action}'. Known actions: {list(self._REQUIREMENTS)}")
        missing_flags = required & ~self.state
        return [
            desc for flag, desc in self._STATE_DESCRIPTIONS.items()
            if flag in missing_flags
        ]

    def can_run(self, action: str) -> bool:
        """
        Check if the manager can run a certain action listed under _REQUIREMENTS (train, eval, inference, setupTiling)

        Parameters
        ----------
        action : str
            string of action listed under _REQUIREMENTS

        Returns
        -------
        _name_ : bool
            Can the selected action be run? (True or False)
        """
        return not self.get_missing_requirements(action)

    def _require(self, action: str) -> None:
        """
        Raises the exception type matching the first (in pipeline order) missing
        requirement for `action`. `.missing` on the exception still lists everything
        missing, not just the one that determined the exception type.

        Parameters
        ----------
        action : str
            action to perform. See AnomalyDetectionManager._REQUIREMENTS

        Raises
        ------
        _name_ : ValueError
            If action is not known
        _name_ : error_cls
            If the state misses flags for the chosen action 
        """
        required = self._REQUIREMENTS.get(action)
        if required is None:
            raise ValueError(f"Unknown action '{action}'. Known actions: {list(self._REQUIREMENTS)}")

        missing_flags = required & ~self.state
        if not missing_flags:
            return

        missing_descriptions = [
            self._STATE_DESCRIPTIONS[flag] for flag in self._STATE_PRIORITY if flag in missing_flags
        ]
        first_missing_flag = next(flag for flag in self._STATE_PRIORITY if flag in missing_flags)
        error_cls = self._STATE_ERROR_CLASSES[first_missing_flag]

        raise error_cls(
            f"Cannot run '{action}': missing {', '.join(missing_descriptions)}",
            missing=missing_descriptions,
        )
    
    def loadModelConfig(self, modelConfigPath: Path) -> ModelConfig:
        modelConfig = ModelConfig.from_yaml(modelConfigPath)
        self.modelConfigPath = modelConfigPath
        self.modelConfig = modelConfig
        return modelConfig
    
    def generateModel(self, modelConfig: ModelConfig) -> None:
        """
        Create a model for the manager to manage. Either from a config or a path to a config yaml-file.

        Parameters
        ----------
        modelConfigPath : Optional[Path] (optional)
            Path to the config file. Has to be a yaml file. Default is `None`
        modelConfig : Optional[ModelConfig] (optional)
            Configuration either loaded from a file or written manually. Default is `None`

        Raises
        ------
        _name_ : ConfigError
            Neither ConfigFile nor Config where given.
        _name_ : ConfigError
            The model creation process failed for some reason.
        """

        model = create_model(modelConfig.name, modelConfig.to_dict())
        if model is None:
            raise ConfigError(f"Model creation failed for config: {modelConfig}")

        self.model = model
        self.state |= ManagerState.MODEL_LOADED
        # loading a new model invalidates any previous training/tiling state
        self.state &= ~(ManagerState.TILING_CONFIGURED | ManagerState.TRAINED | ManagerState.CHECKPOINT_AVAILABLE)
        logger.info(f"Successfully loaded model {self.model.name}: {self.model}")

    def setupTiling(self, tilingPipelineConfig: TilingPipelineConfig) -> bool:
        """
        Setup the tiling mechanism for a TiledEnsemble anomaly detection model. 
        The ensemble contains multiple models, one for each tile.

        Parameters
        ----------
        tilingPipelineConfig : TilingPipelineConfig
            Configuration file for the tilgin mechanism. Usually loaded from a TiledEnsemble.yaml file

        Returns
        -------
        _name_ : bool
            True if the tiling setup was successful
        """
        self._require("setupTiling")
        if tilingPipelineConfig.root_dir is None:
            tilingPipelineConfig.root_dir = self.outputDir
        self.tilingPipelineConfig = tilingPipelineConfig
        self.isTilingSetup = True
        self.state |= ManagerState.TILING_CONFIGURED
        logger.info(f"Tiling configured: {tilingPipelineConfig}")
        return self.isTilingSetup
    
    def _runConfigFiles(self) -> RunConfigFiles:
        """
        The set of config files actually in use by this manager right now,
        anchored to `self.configDir`. See `RunConfigFiles.copy_to` for how
        these get laid out under a run directory.

        Returns
        -------
        _name_ : RunConfigFiles
            Config file paths currently tracked by the manager
        """
        preProcessorPath = postProcessorPath = evaluatorPath = None
        if self.modelConfig is not None:
            preProcessorPath = self.modelConfig.preProcessorPath
            postProcessorPath = self.modelConfig.postProcessorPath
            evaluatorPath = self.modelConfig.evaluatorPath

        return RunConfigFiles(
            configDir=self.configDir,
            modelConfigPath=self.modelConfigPath,
            trainerConfigPath=self.trainerConfigPath,
            tilingConfigPath=self.tilingConfigPath,
            inferencerConfigPath=self.inferencerConfigPath,
            preProcessorPath=preProcessorPath,
            postProcessorPath=postProcessorPath,
            evaluatorPath=evaluatorPath,
        )

    @classmethod
    def loadProduct(cls, productConfigPath: Path, outputPath: Path, configDir: Path) -> Tuple["AnomalyDetectionManager", Product]:
        """
        Loads a Product configuration from disk, which should contain everything for a training run

        Parameters
        ----------
        productConfigPath : Path
            Configuration to the specific product, i.e. MDF6 or cable (from MVTecAD)
        outputPath : Path
            Where the output path of the manager should be
        configDir : Path
            Where the configuration files can be found. Usually a dir named 'configs'

        Returns
        -------
        _name_ : Tuple[AnomalyDetectionManager, Product]
            A tuple of the created manager class and a product that contains all important information about the product.
        """
        product = loadProductFromYaml(productConfigPath, config_dir=configDir, baseOutputDir=outputPath)
        manager = cls(outputDir=outputPath, configDir=configDir)
        manager.generateModel(modelConfig=product.modelConfig)
        manager.setupTiling(product.tilingPipelineConfig)
        manager.tilingConfigPath = product.tilingConfigPath
        manager.trainerConfigPath = product.trainerConfigPath
        manager.modelConfigPath = product.modelConfigPath
        manager.modelTrainingDir = product.modelTrainingDir

        # loadProduct points at a model/weights path that presumably already has a checkpoint
        if manager.modelTrainingDir is not None:
            manager.ckptDir = resolve_checkpoint_paths(manager.modelTrainingDir)
            if manager.ckptDir.exists():
                manager.state |= ManagerState.CHECKPOINT_AVAILABLE | ManagerState.TRAINED
        manager._apply_visualizer_output_dir(outputPath)

        # runID = generate_run_id
        # outputDir = resolve_output_dir(
        #     baseOutputDir=manager.baseOutputDir, datasetName=product.datasetConfig.name,
        #     modelName=product.modelConfig.name, category=product.datasetConfig.category, tiling=True,
        # )

        return manager, product

    def setupTrainingCallbacks(self):
        """
        Setup all callbacks for the training process. Currently these are a checkpoint and a timer callback.


        Returns
        -------
        _name_ : Dict[str,Callback]
            Dictionary of name and callback
        """
        # if self.ckptDir is not None:
        #     if not self.ckptDir.exists():
        #         self.ckptDir.mkdir(parents=True)
        #     self.ckptPath = self.ckptDir / (self.ckptFileName + self.ckptSuffix)
        # else:
        #     AttributeError(f"ckptDir is both None")

        # checkpointCallback = ModelCheckpoint(
        #     dirpath=self.ckptDir,
        #     filename=self.ckptFileName,
        #     monitor="image_F1AdaptiveThreshold",  # val_loss not found?
        #     verbose=True,
        #     save_top_k=1,  # Save only the best model
        #     mode="min",  # Save the model with the minimum training loss,
        #     enable_version_counter=False
        # )
        
        # graphCallback = GraphLogger()
        timerCallback = TimerCallback()
        # progressBar = TQDMProgressBar(refresh_rate=0)

        self.callbacks:Dict[str, Callback] = {
            # "progress_bar": progressBar,
            # "checkpoint": checkpointCallback,generate_run_id
            # "graph": graphCallback,
            "timer": timerCallback
        }

        return self.callbacks
    
    def setupWandBLogger(self, runName:str, runDir:Path, version:int|str):
        """
        Setup the weights and biases logger for this training run

        Parameters
        ----------
        runName : str
            Name of this specific run. Ideally identfying
        runDir : Path
            Directory of where to save the logger files
        version : int | str
            Increasing version index
        """
        self.runLogger = AnomalibWandbLogger(
            name=runName,
            save_dir=runDir,
            version=str(version),
            project="Glas 4.0",
            offline=False,
            entity="daniel-pommer-technische-hochschule-n-rnberg-georg-simon-ohm",
        )

    # def _check_before_training(self) -> AnomalibModule:
    #     """Just check the model — dataset validation is caller's responsibility."""
    #     if self.model is None:
    #         raise AttributeError("Expected model attribute to be set")
    #     return self.model

    def loadCheckpoint(self, path:Optional[Path], tilingPipelineConfig: TilingPipelineConfig):
        """
        Check if the checkpoints for a tiledEnsemble run (eval, inference) are available at the expected directory

        Parameters
        ----------
        path : Path
            Path where to check
        tilingPipelineConfig : TilingPipelineConfig
            Tiling config needed to what kind of files to expect

        Raises
        ------
        _name_ : CheckpointNotFoundError
            If not all expected checkpoints are found.
        """
        if path is None:
            if self.ckptDir is not None:
                path = self.ckptDir
                logger.info(f"Loading checkpoint from manger.ckptDir: {self.ckptDir}")
            else:
                raise AttributeError("Either path needs to be given to loadCheckpoint or ckptDir needs to be set for manager.")
        complete, missingCkpt = checkTiledCheckpointsExist(ckptDir=path, tilingPipelineConfig=tilingPipelineConfig)
        if complete:
            self.state |= ManagerState.CHECKPOINT_AVAILABLE
        else:
            missing:List[str] = [str(path) for path in missingCkpt]
            raise CheckpointNotFoundError(message=f"Follwing Checkpoints are missing from {path}", missing=missing)

    def _trainTiledModel(self,
                         datasetSession:DatasetSession,
                         trainerConfig: TrainerConfig,
                         datamoduleConfig:DataModuleConfig,
                         modelConfig:ModelConfig,
                         tilingPipelineConfig:TilingPipelineConfig):
        """
        Train a tiled ensemble model with the given training parameters on the given data with, considering the tiling.

        Parameters
        ----------
        datasetSession : DatasetSession
            On which data to train the model
        trainerConfig : TrainerConfig
            How the training should go concerning the training itself. (E.g. max iterations)
        datamoduleConfig : DataModuleConfig
            How the training should go concerning the data. (E.g. batch size)
        modelConfig : ModelConfig
            What kind of model to train
        tilingPipelineConfig : TilingPipelineConfig
            How the tiling process works (E.g. tile size, stride,...)
        """
        datamodule = datasetSession.setupDatamodule(datamoduleConfig, self.outputDir)
        gtAvail:bool = True if len(datasetSession.FO_Dataset.exists("ground_truth"))>0 else False

        logger.info(f"Dataset used for training: {datasetSession.FO_Dataset}")

        self.setupTiling(tilingPipelineConfig)

        trainPipeline = TrainTiledEnsemble(rootDir=self.outputDir,
                                           datamodule=datamodule,
                                           dataModuleConfig=datamoduleConfig,
                                           FO_Dataset=datasetSession.FO_Dataset,
                                           gtAvail=gtAvail,
                                           tilingPipelineConfig=tilingPipelineConfig,
                                           modelConfig=modelConfig,
                                           trainerConfig=trainerConfig)
       
        trainPipeline.run()

    def _evalTiledModel(self,
                         datasetSession:DatasetSession,
                         evalConfig: TrainerConfig,
                         datamoduleConfig:DataModuleConfig,
                         modelConfig:ModelConfig,
                         tilingPipelineConfig:TilingPipelineConfig):
        """
        Evaluate a trained tiled ensemble model on a dataset. Also creates a stats.json that desribes how the threshold
        should be set. (Threshold setting is independent of the model, in the sense that the model just gives a score for each image)
        We expect the need checkpoint weights to be set via a previous training or loadCheckpoint()

        Parameters
        ----------
        datasetSession : DatasetSession
            On which data to evaluate the model
        evalConfig : TrainerConfig
            How the evaluation should go concerning the evaluation itself. (E.g. device)
        datamoduleConfig : DataModuleConfig
            How the evaluation should go concerning the data. (E.g. batch size)
        modelConfig : ModelConfig
            What kind of model to train
        tilingPipelineConfig : TilingPipelineConfig
            How the tiling process works (E.g. tile size, stride,...)
        """

        datamodule = datasetSession.setupDatamodule(datamoduleConfig, self.outputDir)
        gtAvail:bool = True if len(datasetSession.FO_Dataset.exists("ground_truth"))>0 else False

        logger.info(f"Dataset used for training: {datasetSession.FO_Dataset}")

        self.setupTiling(tilingPipelineConfig)

        evalPipeline = EvalTiledEnsemble(rootDir=self.outputDir,
                                           datamodule=datamodule,
                                           dataModuleConfig=datamoduleConfig,
                                           FO_Dataset=datasetSession.FO_Dataset,
                                           gtAvail=gtAvail,
                                           tilingPipelineConfig=tilingPipelineConfig,
                                           modelConfig=modelConfig,
                                           evalConfig=evalConfig,
                                           ckptPath=self.ckptDir)
       
        evalPipeline.run()

    def _inferenceTiledModel(self,
                             trainingDir: Path,
                             ckptDir: Path,
                             datasetSession: DatasetSession,
                             dataModuleConfig: DataModuleConfig,
                             tilingPipelineConfig:TilingPipelineConfig,
                             inferencerConfig: TrainerConfig,
                             modelConfig: ModelConfig):
        """
        Predict/Inference a trained tiled ensemble model on a dataset. Also needs a stats.json that desribes how the threshold
        is set. (Threshold setting is independent of the model, in the sense that the model just gives a score for each image)
        We expect the need checkpoint weights to be set via a previous training or loadCheckpoint()

        Parameters
        ----------
        trainingDir : Path
            Directory where the training took place that contains the stats.json file
        ckptDir : Path
            Path to the directory of checkpoint files to use
        datasetSession : DatasetSession
            On which data to evaluate the model
        evalConfig : TrainerConfig
            How the evaluation should go concerning the evaluation itself. (E.g. device)
        datamoduleConfig : DataModuleConfig
            How the evaluation should go concerning the data. (E.g. batch size)
        modelConfig : ModelConfig
            What kind of model to train
        tilingPipelineConfig : TilingPipelineConfig
            How the tiling process works (E.g. tile size, stride,...)
        """

        datamodule = datasetSession.setupDatamodule(dataModuleConfig, self.outputDir)
        # gtAvail:bool = True if len(datasetSession.FO_Dataset.exists("ground_truth"))>0 else False

        logger.info(f"Dataset used for training: {datasetSession.FO_Dataset}")

        self.setupTiling(tilingPipelineConfig)

        assert datasetSession.AL_PredictDataset is not None

        inferencerPipeline = PredTiledEnsemble(root_dir=self.outputDir,
                                          trainingDir=trainingDir,
                                          ckptDir=ckptDir,
                                          predictDataset=datasetSession.AL_PredictDataset,
                                          dataset=datasetSession.FO_Dataset,
                                          datamodule=datamodule,
                                          dataModuleConfig=dataModuleConfig,
                                          tilingPipelineConfig=tilingPipelineConfig,
                                          inferencerConfig=inferencerConfig,
                                          modelConfig=modelConfig
                                          )
                                          
        inferencerPipeline.run()

    def _prepareRun(
        self,
        trainerConfig:TrainerConfig,
        modelConfig: ModelConfig,
        datasetSession: DatasetSession,
        datamoduleConfig: DataModuleConfig,
        tilingPipelineConfig: TilingPipelineConfig,
        runLabel: Optional[str] = None
    ) -> RunContext:
        """
        Shared setup for train()/eval(): resolve paths, wire up tiling,
        callbacks and the W&B logger. Output dir doubles as the checkpoint dir
        since these runs write and read within the same location.

        Parameters
        ----------
        modelConfig : ModelConfig
            Descibes an anomaly detection model 
        datasetSession : DatasetSession
            Contains the dataset 
        tilingPipelineConfig : TilingPipelineConfig
            How should the image be tiled?

        Returns
        -------
        _name_ : RunContext
            Run name, output directory, checkpoint directory, (checkpoint path, not applicable for tiled)
        """
        # if ManagerState.RUN_PREPARED in self.state:
        #     return (RunContext(runName=self.runId, outputDir=self.outputDir))
        runId = generate_run_id(runLabel)
        outputDir = resolve_output_dir(
            baseOutputDir=self.baseOutputDir, datasetName=datasetSession.datasetName,
            modelName=modelConfig.name, runId=runId, category=datasetSession.category, tiling=True,
        )
        # ckptDir = resolve_checkpoint_paths(outputDir)

        effective_config = serialize_effective_config(trainerConfig, modelConfig, datamoduleConfig, tilingPipelineConfig, datasetSession)
        print(effective_config)
        write_run_manifest(outputDir, effective_config)

        self.outputDir, self.runId = outputDir, runId
        self._apply_visualizer_output_dir(outputDir)
        self.setupTiling(tilingPipelineConfig)
        self.setupTrainingCallbacks()
        self.state |= ManagerState.RUN_PREPARED

        return RunContext(runName=runId, outputDir=outputDir)

    def train(
        self,
        trainerConfig: TrainerConfig,
        modelConfig: ModelConfig,
        datamoduleConfig: DataModuleConfig,
        tilingPipelineConfig: TilingPipelineConfig,
        datasetSession: Optional[DatasetSession] = None,
    ) -> None:
        datasetSession = self._resolve_dataset_session(datasetSession)
        self._require("train")
        ctx = self._prepareRun(trainerConfig, modelConfig, datasetSession, datamoduleConfig=datamoduleConfig, tilingPipelineConfig=tilingPipelineConfig)
        self._runConfigFiles().copy_to(ctx.outputDir) 
         # raw YAMLs alongside the manifest
        self.ckptDir = resolve_checkpoint_paths(ctx.outputDir)

        self._trainTiledModel(
            datasetSession=datasetSession, trainerConfig=trainerConfig,
            datamoduleConfig=datamoduleConfig, modelConfig=modelConfig,
            tilingPipelineConfig=tilingPipelineConfig,
        )
        complete, missing = checkTiledCheckpointsExist(self.ckptDir, tilingPipelineConfig)
        if complete:
            self.state |= ManagerState.TRAINED | ManagerState.CHECKPOINT_AVAILABLE
        else:
            logger.warning(f"Training finished but {len(missing)} tile checkpoint(s) missing: {missing}")

    def eval(
        self,
        evalConfig: TrainerConfig,
        modelConfig: ModelConfig,
        datamoduleConfig: DataModuleConfig,
        datasetSession: DatasetSession,
        tilingPipelineConfig: TilingPipelineConfig,
        modelTrainingDir: Optional[Path] = None,
    ) -> None:
        """Evaluate the current model on the current dataset. Tiled ensemble only for now."""
        datasetSession = self._resolve_dataset_session(datasetSession)
        if modelTrainingDir is not None:
            self.modelTrainingDir = modelTrainingDir
        else:
            if self.modelTrainingDir is None:
                raise AttributeError("Need a modelTrainingDir and neither attribute nor class atrribute are available")
            modelTrainingDir = self.modelTrainingDir
        self.ckptDir = resolve_checkpoint_paths(modelTrainingDir)
        complete, missing = checkTiledCheckpointsExist(self.ckptDir, tilingPipelineConfig)
        if not complete:
            raise CheckpointNotFoundError(
                f"Incomplete checkpoint set in {self.ckptDir}: missing {[p.name for p in missing]}",
                missing=[f"Checkpoint file {p.name}" for p in missing],
            )
        self._require("eval")
        self._prepareRun(evalConfig, modelConfig, datasetSession, datamoduleConfig, tilingPipelineConfig)
        self._evalTiledModel(
            datasetSession=datasetSession, evalConfig=evalConfig,
            datamoduleConfig=datamoduleConfig, modelConfig=modelConfig,
            tilingPipelineConfig=tilingPipelineConfig,
        )

    def inference(
        self,
        inferencerConfig: TrainerConfig,
        modelConfig: ModelConfig,
        datamoduleConfig: DataModuleConfig,
        datasetSession: DatasetSession,
        tilingPipelineConfig: TilingPipelineConfig,
        modelTrainingDir: Optional[Path] = None,
    ) -> None:
        """Run inference: write results under the *current* dataset's output dir,
        but read the checkpoint from `trainingDir` (a prior, separate run).
        This replaces the old adjustPaths(..., adjustCheckpoints=False) workaround —
        the two directories are now resolved independently and explicitly.
        """
        datasetSession = self._resolve_dataset_session(datasetSession)
        if modelTrainingDir is not None:
            self.modelTrainingDir = modelTrainingDir
        else:
            if self.modelTrainingDir is None:
                raise AttributeError("Need a modelTrainingDir and neither attribute nor class atrribute are available")
            modelTrainingDir = self.modelTrainingDir

        self.ckptDir = resolve_checkpoint_paths(modelTrainingDir)
        complete, missing = checkTiledCheckpointsExist(self.ckptDir, tilingPipelineConfig)
        if not complete:
            raise CheckpointNotFoundError(
                f"Incomplete checkpoint set in {self.ckptDir}: missing {[p.name for p in missing]}",
                missing=[f"Checkpoint file {p.name}" for p in missing],
            )
        
        self._require("inference")
        self._prepareRun(inferencerConfig, modelConfig, datasetSession, datamoduleConfig, tilingPipelineConfig)

        # self.outputDir = ctx.outputDir
        # self._apply_visualizer_output_dir(ctx.outputDir)
        # self.setupTiling(tilingPipelineConfig)

        self._inferenceTiledModel(
            trainingDir=modelTrainingDir, 
            ckptDir=self.ckptDir, 
            datasetSession=datasetSession,
            dataModuleConfig=datamoduleConfig, 
            tilingPipelineConfig=tilingPipelineConfig,
            inferencerConfig=inferencerConfig, 
            modelConfig=modelConfig,
        )

    def exportResults(self, exportPath:Path) -> None:
        # Create the destination directory if it doesn't exist
        os.makedirs(exportPath, exist_ok=True)

        # Copy each item in the source directory
        for item in os.listdir(self.outputDir):
            src_path = os.path.join(self.outputDir, item)
            dst_path = os.path.join(exportPath, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)