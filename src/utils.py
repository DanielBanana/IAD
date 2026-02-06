import os
import sys

from pathlib import Path
from contextlib import contextmanager

import yaml
from anomalib.metrics import Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import ImageVisualizer
from anomalib.metrics import AUROC, AUPR, F1AdaptiveThreshold, F1Score
from setup import define_metrics
from settings import DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG
import shutil
import os
from pathlib import Path
import yaml
import torchvision.transforms.v2 as T
from typing import Any

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

def loadConfig(config_path:Path, copyPath:Path|None=None) -> Any:
    """Load YAML config file."""

    # Open the general config file
    with open(config_path, 'r') as f:
        modelConfig = yaml.safe_load(f)

    # If the configs should be copied create the folder and copy the general config
    if copyPath is not None:
        copyPath = copyPath / modelConfig["model"]
        if copyPath is not None:
            if not copyPath.exists():
                copyPath.mkdir()
            _, configFileName = os.path.split(config_path)
            shutil.copy2(config_path, copyPath / configFileName)

    # Read the preprocessor and copy if desired
    if modelConfig["pre_processor_path"]:
        transform = load_transform_from_yaml(modelConfig["pre_processor_path"], copyPath=copyPath)
        modelConfig["pre_processor"] = PreProcessor(transform)
    else:
        modelConfig["pre_processor"] = PreProcessor()

    # Read the Postprocessor and copy if desired
    if modelConfig["post_processor_path"]:
        with open(modelConfig["post_processor_path"], 'r') as f:
            postProcessorConfig = yaml.safe_load(f)
        configFileName = modelConfig["post_processor_path"].split(os.sep)[-1]
        if copyPath is not None:
            shutil.copy2(config_path, os.path.join(copyPath, configFileName))
        modelConfig["post_processor"] = PostProcessor(**postProcessorConfig)
    else:
        modelConfig["post_processor"] = PostProcessor(
            enable_normalization=True,
            enable_threshold_matching=True,
            enable_thresholding=True,
            image_sensitivity=0.01,
            pixel_sensitivity=0.01
        )

    modelConfig["visualizer"] = ImageVisualizer(# output_dir=prediction_path,
                                fields=["image", "gt_mask"],
                                overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],
                                field_size=(256,256),
                                fields_config=DEFAULT_FIELDS_CONFIG,
                                overlay_fields_config=DEFAULT_OVERLAY_FIELDS_CONFIG,
                                text_config=DEFAULT_TEXT_CONFIG)

    if modelConfig["evaluator_path"]:
        if copyPath is not None:
            configFileName = modelConfig["evaluator_path"].split(os.sep)[-1]
            shutil.copy2(config_path, os.path.join(copyPath, configFileName))
        modelConfig["evaluator"] = Evaluator(*load_metrics_from_yaml(modelConfig["evaluator_path"]))
    else:
        modelConfig["evaluator"] = getDefaultEvaluator()
    return modelConfig

def load_transform_from_yaml(configPath, copyPath=None):
    with open(configPath, "r") as f:
        config = yaml.safe_load(f)

    # Copy file to project folder
    if copyPath is not None:
        configFileName = configPath.split(os.sep)[-1]
        shutil.copy2(configPath, os.path.join(copyPath, configFileName))

    transform_list = []
    for step in config["transform"]:
        transform_name = step["name"]
        transform_args = step["args"]

        # Get the transform class from torchvision.transforms
        transform_class = getattr(T, transform_name)
        # Instantiate the transform with the provided arguments
        transform = transform_class(**transform_args)
        transform_list.append(transform)

    # Compose all transforms into a pipeline
    return T.Compose(transform_list)

def load_metrics_from_yaml(configPath: Path):
    with open(configPath, "r") as f:
        config = yaml.safe_load(f)

    # Map metric types to their classes
    metric_classes = {
        "AUROC": AUROC,
        "AUPR": AUPR,
        "F1AdaptiveThreshold": F1AdaptiveThreshold,
        "F1Score": F1Score,
    }

    def instantiate_metrics(metrics_config):
        if metrics_config is None:
            return None
        metrics = []
        for metric_config in metrics_config:
            metric_type = metric_config["type"]
            fields = metric_config["fields"]
            prefix = metric_config["prefix"]
            # Instantiate the metric
            metric = metric_classes[metric_type](fields=fields, prefix=prefix)
            metrics.append(metric)
        return metrics

    val_metrics = instantiate_metrics(config["val_metrics"])
    test_metrics = instantiate_metrics(config["test_metrics"])

    return val_metrics, test_metrics


def getDefaultEvaluator():
    val_metrics, test_metrics = define_metrics()
    evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
    return evaluator