import yaml
from anomalib.metrics import Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import AUROC, AUPR, F1AdaptiveThreshold, F1Score

import yaml
from torchvision import transforms

def load_transform_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    transform_list = []
    for step in config["transform"]:
        transform_name = step["name"]
        transform_args = step["args"]

        # Get the transform class from torchvision.transforms
        transform_class = getattr(transforms, transform_name)
        # Instantiate the transform with the provided arguments
        transform = transform_class(**transform_args)
        transform_list.append(transform)

    # Compose all transforms into a pipeline
    return transforms.Compose(transform_list)

def load_metrics_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Map metric types to their classes
    metric_classes = {
        "AUROC": AUROC,
        "AUPR": AUPR,
        "F1AdaptiveThreshold": F1AdaptiveThreshold,
        "F1Score": F1Score,
    }

    def instantiate_metrics(metrics_config):
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

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        modelConfig = yaml.safe_load(f)
    if modelConfig["pre_processor"]:
        transform = load_transform_from_yaml(modelConfig["pre_processor"])
        preProcessor = PreProcessor(transform)
    else:
        preProcessor = PreProcessor()
    if modelConfig["post_processor"]:
        with open(modelConfig["post_processor"], 'r') as f:
            postProcessorConfig = yaml.safe_load(f)
        postProcessor = PostProcessor(**postProcessorConfig)
    else:
        postProcessor = PostProcessor(
            enable_normalization=True,
            enable_threshold_matching=True,
            enable_thresholding=True,
            image_sensitivity=0.01,
            pixel_sensitivity=0.01
        )

    if modelConfig["evaluator"]:
        evaluator = Evaluator(*load_metrics_from_yaml(modelConfig["evaluator"]))
    else:
        evaluator = getDefaultEvaluator()
    return modelConfig, preProcessor, postProcessor, evaluator