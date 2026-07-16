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
from setup import define_metrics
from settings import DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG
from tiling.post_processor import AOIPostProcessor
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