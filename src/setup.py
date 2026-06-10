"""
Helpful functions for setting up an anomaly detection model
"""

# GENERAL
import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Literal, Tuple, Dict, List, Any

from anomalib.metrics import F1Score, AUPR, AUROC, F1AdaptiveThreshold
from anomalib.data import MVTecAD, BTech, Visa, Kolektor, Folder
from anomalib.models import Padim, EfficientAd, Dsr, ReverseDistillation, Fastflow, Patchcore, Stfpm
from anomalib.callbacks import ModelCheckpoint, GraphLogger, TimerCallback
from anomalib.loggers import AnomalibTensorBoardLogger, AnomalibWandbLogger

from lightning.pytorch.callbacks import TQDMProgressBar

from src.settings import DATASETS, CATEGORIES, MODELS, DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG

def define_metrics() -> Tuple[List[Any], List[Any]]:
    # val metrics (needed for early stopping)
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    image_aupr = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_aupr = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    image_f1score = F1AdaptiveThreshold(fields=["pred_score", "gt_label"], prefix="image_")
    val_metrics:List[Any] = [image_auroc, pixel_auroc, image_aupr, pixel_aupr, image_f1score]

    # test_metrics
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    image_aupr = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_aupr = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    test_metrics:List[Any]  = [image_auroc, image_f1score, pixel_auroc, pixel_f1score, image_aupr, pixel_aupr]
    
    return val_metrics, test_metrics

def create_datamodule(dataset, category, train_batch_size, eval_batch_size, num_workers, test_split_mode, test_split_ratio, val_split_mode, val_split_ratio):
    # 2. Create a dataset
    if dataset.lower() == "mvtecad":
        if category == "metal_nut":
            category = "metal nut"
        datasetPath = f"datasets/MVTecAD"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = MVTecAD(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # MVTecAD category to use
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio
        )
    elif dataset.lower() == "visa":
        datasetPath = f"datasets/visa"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = Visa(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # Visa category to use
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio
        )
    elif dataset.lower() == "kolektor":
        datasetPath = f"datasets/kolektor"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = Kolektor(
            root=datasetPath,  # Path to download/store the dataset
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio
        )
    elif dataset.lower() == "btech":
        datasetPath = f"datasets/btech"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = BTech(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # BTech category to use
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio
        )
    elif dataset.lower() == "custom":
        datasetPath = f"datasets/custom"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = Folder(
            name="custom",
            root=datasetPath,  # Path to download/store the dataset
            normal_dir="simple/train/good",
            abnormal_dir="simple/test/writing",
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio
        )
        datamodule.setup()
        
    return datamodule

def mapNameToModule(modelName:str):
    if modelName.lower() == "efficientad-s":
        # model = EfficientAd(visualizer=visualizer, model_size="small", post_processor=postProcessor)
        model = EfficientAd
    elif modelName.lower() == "efficientad-m":
        model = EfficientAd
    elif modelName.lower() == "dsr":
        model = Dsr
    elif modelName.lower() == "reversedistillation":
        model = ReverseDistillation
    elif modelName.lower() == "reverse_distillation":
        model = ReverseDistillation
    elif modelName.lower() == "rd":
        model = ReverseDistillation
    elif modelName.lower() == "stfpm":
        model = Stfpm
    elif modelName.lower() == "fastflow":
        model = Fastflow
    elif modelName.lower() == "fast_flow":
        model = Fastflow
    elif modelName.lower() == "patchcore":
        model = Patchcore
    elif modelName.lower() == "padim":
        model = Padim
    else:
        KeyError(f"Model {modelName} not found! \n Available models are: {', '.join(MODELS)}")
        model=None
    return model

def create_model(modelName, modelConfig):
    if modelName == "EfficientAd":
        # model = EfficientAd(visualizer=visualizer, model_size="small", post_processor=postProcessor)
        model = EfficientAd(**modelConfig)
    elif modelName == "dsr":
        model = Dsr(**modelConfig)
    elif modelName == "reversedistillation":
        model = ReverseDistillation(**modelConfig)
    elif modelName == "reverse_distillation":
        model = ReverseDistillation(**modelConfig)
    elif modelName == "rd":
        model = ReverseDistillation(**modelConfig)
    elif modelName == "stfpm":
        model = Stfpm(**modelConfig)
    elif modelName == "Stfpm":
        model = Stfpm(**modelConfig)
    elif modelName == "fastflow":
        model = Fastflow(**modelConfig)
    elif modelName == "fast_flow":
        model = Fastflow(**modelConfig)
    elif modelName == "patchcore":
        model = Patchcore(**modelConfig)
    elif modelName == "Patchcore":
        model = Patchcore(**modelConfig)
    elif modelName == "padim":
        model = Padim(**modelConfig)
    elif modelName == "Padim":
        model = Padim(**modelConfig)
    else:
        print(f"Model {modelName} not found! \n Available models are: {', '.join(MODELS)}")
        model = None
    return model

from typing import Tuple, Dict, Any

def setupTensorboardLoggingAndCallbacks(logDir, runName, versionName, version) -> Tuple[AnomalibTensorBoardLogger, Dict[str, Any]]:
    checkpointDir = os.path.join(logDir, runName, versionName, "checkpoints")
    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)
    checkpointCallback = ModelCheckpoint(
        dirpath=checkpointDir,
        filename="best",
        monitor="image_F1AdaptiveThreshold",  # val_loss not found?
        verbose=True,
        save_top_k=1,  # Save only the best model
        mode="min",  # Save the model with the minimum training loss
    )
    
    checkpointCallback.FILE_EXTENSION = ".pt"  # Set the file extension for checkpoints
    
    graphCallback = GraphLogger()
    timerCallback = TimerCallback()
    progressBar = TQDMProgressBar(refresh_rate=50)
    
    callbacks = {
        "progress_bar": progressBar,
        "checkpoint": checkpointCallback,
        "graph": graphCallback,
        "timer": timerCallback
    }
    
    tblogger = AnomalibTensorBoardLogger(
        save_dir=logDir,
        name=runName,
        version=version)
    
    return tblogger, callbacks, os.path.join(checkpointDir, "best.pt")

def setupWandBLoggingAndCallbacks(logDir:str, runName:str, version:int, checkpointFileName:str="checkpoint_best", id:str|None=None) -> Tuple[AnomalibWandbLogger, dict[str,Any], Path, Path]:
    runDir:Path = Path(os.path.join(logDir, runName, f"version_{version}"))
    checkpointDir:Path = runDir / "checkpoints"
    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)

    checkpointCallback = ModelCheckpoint(
        dirpath=checkpointDir,
        filename=checkpointFileName,
        monitor="image_F1AdaptiveThreshold",  # val_loss not found?
        verbose=True,
        save_top_k=1,  # Save only the best model
        mode="min",  # Save the model with the minimum training loss
    )
    
    graphCallback = GraphLogger()
    timerCallback = TimerCallback()
    progressBar = TQDMProgressBar(refresh_rate=50)

    callbacks:dict[str, Any] = {
        "progress_bar": progressBar,
        "checkpoint": checkpointCallback,
        "graph": graphCallback,
        "timer": timerCallback
    }
    
    wandbLogger = AnomalibWandbLogger(name=runName, save_dir=runDir, version=str(version), project="Glas 4.0")

    return wandbLogger, callbacks, runDir, checkpointDir
    


class LoggerWriter:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

class LoggerStdin:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.builtin_stdin = sys.stdin

    def readline(self):
        line = self.builtin_stdin.readline()
        self.logger.log(self.log_level, f"Input: {line.rstrip()}")
        return line

    def __getattr__(self, attr):
        return getattr(self.builtin_stdin, attr)

def find_first_file(directory, target_filename):
    target_lower = target_filename.lower()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower() == target_lower:
                return os.path.join(root, file)
    return None  # Return None if the file is not found