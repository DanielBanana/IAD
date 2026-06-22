"""
Short collection of utility function. Should be dissolved into more appropriate files in the long run.
"""

# GENERAL
import os
import sys
import yaml
import torchvision.transforms.v2 as T
import shutil
from typing import Any, Dict, Tuple, List
from enum import Enum
from torchvision.transforms.v2 import Transform
from pathlib import Path
from contextlib import contextmanager

# ANOMALIB
from anomalib.metrics import Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import ImageVisualizer
from anomalib.metrics import AUROC, AUPR, F1AdaptiveThreshold, F1Score

# OWN FILES
from .setup import define_metrics
from .settings import DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG
from .tiling.post_processor import AOIPostProcessor


class VisualizerType(Enum):
    train = 0
    valNoGT = 1
    valGT = 2

def getVisualizer(vtype:VisualizerType, fieldSize:Tuple[int,int]):
    match vtype:
        case VisualizerType.train:
            visualizer = ImageVisualizer(# output_dir=prediction_path,
                        fields=["image"],
                        overlay_fields=[("image", ["anomaly_map"])],
                        field_size=fieldSize,
                        fields_config=DEFAULT_FIELDS_CONFIG,
                        overlay_fields_config=DEFAULT_OVERLAY_FIELDS_CONFIG,
                        text_config=DEFAULT_TEXT_CONFIG)
            return visualizer
        case VisualizerType.valNoGT:
            visualizer = ImageVisualizer(# output_dir=prediction_path,
                        fields=["image", "pred_mask"],
                        overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],
                        field_size=fieldSize,
                        fields_config=DEFAULT_FIELDS_CONFIG,
                        overlay_fields_config=DEFAULT_OVERLAY_FIELDS_CONFIG,
                        text_config=DEFAULT_TEXT_CONFIG)
            return visualizer

        case VisualizerType.valGT:
            visualizer = ImageVisualizer(# output_dir=prediction_path,
                        fields=["image", "gt_mask", "pred_mask"],
                        overlay_fields=[("image", ["anomaly_map"]), ("image", ["gt_mask"]), ("image", ["pred_mask"])],
                        field_size=fieldSize,
                        fields_config=DEFAULT_FIELDS_CONFIG,
                        overlay_fields_config=DEFAULT_OVERLAY_FIELDS_CONFIG,
                        text_config=DEFAULT_TEXT_CONFIG)
            return visualizer

def find_first_file(directory:Path, target_filename:str) -> Path|None:
    target_lower = target_filename.lower()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower() == target_lower:
                return Path(os.path.join(root, file))
    return None  # Return None if the file is not found

@contextmanager
def exclude_from_logger():
    original_stdout = sys.stdout
    sys.stdout = sys.__stdout__  # Restore original stdout
    try:
        yield
    finally:
        sys.stdout = original_stdout  # Restore logger stdout

def loadModelConfig(configDir:Path, modelConfigPath:Path, copyDir:Path|None=None, vtype:VisualizerType|None=None, fieldSize:Tuple[int,int]|None=None) -> Tuple[Dict[str,Any], Path|None, Path|None, Path|None]:
    """Load YAML config file."""

    # Open the general config file
    with open(modelConfigPath, 'r') as f:
        modelConfig = yaml.safe_load(f)

    modelConfig = dict(modelConfig)

    # If the configs should be copied create the folder and copy the general config
    if copyDir is not None:
        relativePath:Path = modelConfigPath.relative_to(configDir)
        copyPath = copyDir / relativePath
        copyPath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(modelConfigPath, copyPath)

    # Read the preprocessor and copy if desired
    
    pre_processor:str|None = modelConfig["model"]["init_args"].pop("pre_processor_path",None)
    post_processor:str|None = modelConfig["model"]["init_args"].pop("post_processor_path",None)
    evaluator:str|None = modelConfig["model"]["init_args"].pop("evaluator_path",None)

    preProcessorPath:Path|None
    postProcessorPath:Path|None
    evaluatorPath:Path|None

    if pre_processor is not None:
        preProcessorPath = configDir / "Engine" /  Path(pre_processor)
        if copyDir is not None:
            relativePath:Path = preProcessorPath.relative_to(configDir)
            copyPath = copyDir / relativePath
            copyPath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(modelConfigPath, copyPath)
        transform = load_transform_from_yaml(preProcessorPath)
        modelConfig["model"]["init_args"]["pre_processor"] = PreProcessor(transform)
    else:
        modelConfig["model"]["init_args"]["pre_processor"] = PreProcessor()
        preProcessorPath = None

    # Read the Postprocessor and copy if desired
    if post_processor is not None:
        postProcessorPath = configDir / "Engine" / Path(post_processor)
        if copyDir is not None:
            relativePath:Path = postProcessorPath.relative_to(configDir)
            copyPath = copyDir / relativePath
            copyPath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(modelConfigPath, copyPath)
        with open(postProcessorPath, 'r') as f:
            postProcessorConfig = yaml.safe_load(f)
        modelConfig["model"]["init_args"]["post_processor"] = AOIPostProcessor(**postProcessorConfig)
    else:
        modelConfig["model"]["init_args"]["post_processor"] = AOIPostProcessor(
            enable_normalization=True,
            enable_threshold_matching=True,
            enable_thresholding=True,
            image_sensitivity=0.01,
            pixel_sensitivity=0.01
        )
        postProcessorPath = None
    
    if vtype is None:
        modelConfig["model"]["init_args"]["visualizer"] = False
    else:
        if fieldSize is None:
            raise ValueError(f"If Visualiser is to be created a fieldSize needs to be given; {fieldSize} not allowed")
        else:
            modelConfig["model"]["init_args"]["visualizer"] = getVisualizer(vtype=vtype, fieldSize=fieldSize)

    if evaluator is not None:
        evaluatorPath = configDir / "Engine" / Path(evaluator)
        if copyDir is not None:
            relativePath:Path = evaluatorPath.relative_to(configDir)
            copyPath = copyDir / relativePath
            copyPath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(modelConfigPath, copyPath)
        modelConfig["model"]["init_args"]["evaluator"] = Evaluator(*load_metrics_from_yaml(evaluatorPath))
    else:
        modelConfig["model"]["init_args"]["evaluator"] = getDefaultEvaluator()
        evaluatorPath=None

    return modelConfig, preProcessorPath, postProcessorPath, evaluatorPath

def load_transform_from_yaml(configPath:Path):
    with open(configPath, "r") as f:
        config:Dict[str,Any] = yaml.safe_load(f)

    transform_list:List[Transform] = []
    for step in config["transform"]:
        transform_name = step["name"]
        transform_args = step["args"]

        # Get the transform class from torchvision.transforms
        transform_class = getattr(T, transform_name)
        # Instantiate the transform with the provided arguments
        transform:Transform = transform_class(**transform_args)
        transform_list.append(transform)

    # Compose all transforms into a pipeline
    return T.Compose(transform_list)


def load_metrics_from_yaml(configPath: Path) -> tuple[List[Any], List[Any]]:
    with open(configPath, "r") as f:
        config:Dict[str,List[Dict[str,Any]]] = yaml.safe_load(f)

    # Map metric types to their classes
    metric_classes: Dict[str,Any] = {
        "AUROC": AUROC,
        "AUPR": AUPR,
        "F1AdaptiveThreshold": F1AdaptiveThreshold,
        "F1Score": F1Score,
    }

    def instantiate_metrics(metrics_config:List[Dict[str,Any]]) -> List[Any]:
        metrics:List[Any] = []
        for metric_config in metrics_config:
            metric_type = metric_config["type"]
            fields = metric_config["fields"]
            prefix = metric_config["prefix"]
            # Instantiate the metric
            metric = metric_classes[metric_type](fields=fields, prefix=prefix)
            metrics.append(metric)
        return metrics

    val_metrics:List[Any] = instantiate_metrics(config["val_metrics"])
    test_metrics:List[Any]  = instantiate_metrics(config["test_metrics"])

    return val_metrics, test_metrics


def getDefaultEvaluator():
    val_metrics, test_metrics = define_metrics()
    evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
    return evaluator