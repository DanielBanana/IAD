from dataclasses import asdict, dataclass, field
from pathlib import Path
import importlib
import os
import shutil
import numpy as np
import torchvision.transforms.v2 as T
from anomalib.metrics import Evaluator
from anomalib.pre_processing import PreProcessor
from tiling.post_processor import AOIPostProcessor
from setup import define_metrics, create_model
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, Tuple
import yaml
from enum import Enum
from cv2.typing import MatLike

from typing import Generic, TypeVar

import uuid

def generate_unique_id() -> str:
    return str(uuid.uuid4())


def load_transform_from_yaml(config_path: Path):
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    transform_list: List[T.Transform] = []
    for step in config.get("transform", []):
        transform_name = step["name"]
        transform_args = step.get("args", {})
        transform_class = getattr(T, transform_name)
        transform_list.append(transform_class(**transform_args))

    return T.Compose(transform_list)


def load_metrics_from_yaml(config_path: Path) -> tuple[List[Any], List[Any]]:
    with open(config_path, "r") as f:
        config: Dict[str, List[Dict[str, Any]]] = yaml.safe_load(f)

    metric_classes: Dict[str, Any] = {
        "AUROC": getattr(__import__("anomalib.metrics", fromlist=["AUROC"]), "AUROC"),
        "AUPR": getattr(__import__("anomalib.metrics", fromlist=["AUPR"]), "AUPR"),
        "F1AdaptiveThreshold": getattr(__import__("anomalib.metrics", fromlist=["F1AdaptiveThreshold"]), "F1AdaptiveThreshold"),
        "F1Score": getattr(__import__("anomalib.metrics", fromlist=["F1Score"]), "F1Score"),
    }

    def instantiate_metrics(metrics_config: List[Dict[str, Any]]) -> List[Any]:
        metrics: List[Any] = []
        for metric_config in metrics_config:
            metric_type = metric_config["type"]
            fields = metric_config["fields"]
            prefix = metric_config["prefix"]
            metrics.append(metric_classes[metric_type](fields=fields, prefix=prefix))
        return metrics

    val_metrics: List[Any] = instantiate_metrics(config.get("val_metrics", []))
    test_metrics: List[Any] = instantiate_metrics(config.get("test_metrics", []))
    return val_metrics, test_metrics


def get_default_evaluator() -> Evaluator:
    val_metrics, test_metrics = define_metrics()
    return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)


def resolve_config_path(path_str: str, model_yaml_path: Path, config_dir: Optional[Path] = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates: List[Path] = []
    if config_dir is not None:
        candidates.append(config_dir / path)
        candidates.append(config_dir / "Engine" / path)

    candidates.append(model_yaml_path.parent / path)
    if len(model_yaml_path.parents) >= 2:
        candidates.append(model_yaml_path.parents[1] / path)
    if len(model_yaml_path.parents) >= 3:
        candidates.append(model_yaml_path.parents[2] / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to resolve path '{path_str}' from model YAML '{model_yaml_path}'"
    )


def copy_file_to_dir(source_path: Path, copy_dir: Path, base_dir: Optional[Path] = None) -> Path:
    try:
        relative_path = source_path.relative_to(base_dir) if base_dir is not None else source_path.name
    except Exception:
        relative_path = source_path.name

    destination = copy_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination


def parse_pre_processor(init_args: Dict[str, Any], model_yaml_path: Path, config_dir: Optional[Path]) -> tuple[Any, Optional[Path]]:
    pre_processor_value = init_args.pop("pre_processor_path", None)
    if pre_processor_value is None and "pre_processor" in init_args:
        pre_processor_value = init_args["pre_processor"]

    if pre_processor_value is False:
        return False, None
    if isinstance(pre_processor_value, str):
        resolved = resolve_config_path(pre_processor_value, model_yaml_path, config_dir)
        return PreProcessor(load_transform_from_yaml(resolved)), resolved
    return PreProcessor(), None


def parse_post_processor(init_args: Dict[str, Any], model_yaml_path: Path, config_dir: Optional[Path]) -> tuple[Any, Optional[Path]]:
    post_processor_value = init_args.pop("post_processor_path", None)
    if post_processor_value is None and "post_processor" in init_args:
        post_processor_value = init_args["post_processor"]

    if post_processor_value is False:
        return False, None
    if isinstance(post_processor_value, str):
        resolved = resolve_config_path(post_processor_value, model_yaml_path, config_dir)
        with open(resolved, "r") as f:
            post_processor_config = yaml.safe_load(f)
        return AOIPostProcessor(**post_processor_config), resolved
    return AOIPostProcessor(), None


def parse_evaluator(init_args: Dict[str, Any], model_yaml_path: Path, config_dir: Optional[Path]) -> tuple[Any, Optional[Path]]:
    evaluator_value = init_args.pop("evaluator_path", None)
    if evaluator_value is None and "evaluator" in init_args:
        evaluator_value = init_args["evaluator"]

    if evaluator_value is False:
        return False, None
    if isinstance(evaluator_value, str):
        resolved = resolve_config_path(evaluator_value, model_yaml_path, config_dir)
        return Evaluator(*load_metrics_from_yaml(resolved)), resolved
    return get_default_evaluator(), None


def parse_model_init_args(
    model_yaml_path: Path,
    config_dir: Optional[Path] = None,
    copy_dir: Optional[Path] = None,
) -> tuple[Dict[str, Any], Optional[Path], Optional[Path], Optional[Path]]:
    with open(model_yaml_path, "r") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("YAML model config must contain a mapping at the top level.")

    model_section = config.get("model")
    if not isinstance(model_section, dict):
        raise ValueError("YAML model config must contain a 'model' section as a mapping.")

    init_args = dict(model_section.get("init_args", {}))
    if not isinstance(init_args, dict):
        raise TypeError("'model.init_args' must be a mapping.")

    if copy_dir is not None:
        if config_dir is None:
            raise ValueError("copy_dir requires config_dir to be provided")
        copy_file_to_dir(model_yaml_path, copy_dir, base_dir=config_dir)

    pre_processor, pre_processor_path = parse_pre_processor(init_args, model_yaml_path, config_dir)
    post_processor, post_processor_path = parse_post_processor(init_args, model_yaml_path, config_dir)
    evaluator, evaluator_path = parse_evaluator(init_args, model_yaml_path, config_dir)

    init_args["pre_processor"] = pre_processor
    init_args["post_processor"] = post_processor
    init_args["evaluator"] = evaluator

    if copy_dir is not None:
        if pre_processor_path is not None:
            copy_file_to_dir(pre_processor_path, copy_dir, base_dir=config_dir)
        if post_processor_path is not None:
            copy_file_to_dir(post_processor_path, copy_dir, base_dir=config_dir)
        if evaluator_path is not None:
            copy_file_to_dir(evaluator_path, copy_dir, base_dir=config_dir)

    return init_args, pre_processor_path, post_processor_path, evaluator_path


def resolve_product_config_path(path_str: str, product_yaml_path: Path, config_dir: Optional[Path] = None, subdir: Optional[str] = None) -> Path:
    if Path(path_str).is_absolute():
        return Path(path_str)

    candidates: List[Path] = []
    if config_dir is not None:
        if subdir:
            candidates.append(config_dir / subdir / path_str)
        candidates.append(config_dir / path_str)
    candidates.append(product_yaml_path.parent / path_str)
    if subdir is not None:
        candidates.append(product_yaml_path.parent / subdir / path_str)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to resolve product-config path '{path_str}' from '{product_yaml_path}'"
    )


def load_dataset_config(config: Dict[str, Any], product_yaml_path: Path) -> "DatasetConfig":
    if not isinstance(config, dict):
        raise TypeError("dataset config must be a dict")

    name = config.get("name")
    category = config.get("category")
    splits = config.get("split") or config.get("splits")
    path_value = config.get("path")

    if not isinstance(name, str) or not name:
        raise ValueError("dataset.name must be a non-empty string")
    if not isinstance(category, str) or not category:
        raise ValueError("dataset.category must be a non-empty string")
    if not isinstance(splits, list) or not all(isinstance(item, str) for item in splits):
        raise ValueError("dataset.split must be a list of strings")

    if isinstance(path_value, str) and path_value:
        dataset_path = Path(path_value)
        if not dataset_path.is_absolute():
            dataset_path = (product_yaml_path.parent / dataset_path).resolve()
    else:
        dataset_path = (Path("datasets") / name).resolve()

    return DatasetConfig(name=name, category=category, splits=tuple(splits), path=dataset_path)



@dataclass
class Image():
    """Class that holds an image to process
    """
    image: NDArray[np.float32]|None = None
    imageSegments: List[NDArray[np.float32]]|None = None

    def addImage(self, image:NDArray[np.float32]):
        self.image = image

    def addImageSegments(self, segments:NDArray[np.float32]):
        segments = np.asarray(segments)
        self.Image = np.concat(segments, axis=0)



                                    # rotational angle of the bounding box of the error; in degree; 0 is parallel to x axis to the right, 90 is parallel to y axis upwards

@dataclass
class DatasetConfig:
    """Dataset parameters"""
    name:str
    category:str
    splits: Tuple[str,str]|Tuple[str]
    path: Path

@dataclass
class DataModuleParams:
    train_batch_size: Optional[int] = None
    train_augmentations: Optional[List[Any]] = None
    val_augmentations: Optional[List[Any]] = None
    test_augmentations: Optional[List[Any]] = None
    augmentations: Optional[List[Any]] = None
    val_split_mode: Optional[str] = None
    val_split_ratio: Optional[float] = None
    test_split_mode: Optional[str] = None
    test_split_ratio: Optional[float] = None
    seed: Optional[int] = None

    @classmethod
    def extract_datamodule_params(cls, config: Dict[str, Any]) -> "DataModuleParams":
        """Create a DataModuleParams instance from a dictionary."""
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        return cls(
            train_batch_size=config.get("train_batch_size"),
            train_augmentations=config.get("train_augmentations"),
            val_augmentations=config.get("val_augmentations"),
            test_augmentations=config.get("test_augmentations"),
            augmentations=config.get("augmentations"),
            val_split_mode=config.get("val_split_mode"),
            val_split_ratio=config.get("val_split_ratio"),
            test_split_mode=config.get("test_split_mode"),
            test_split_ratio=config.get("test_split_ratio"),
            seed=config.get("seed"),
        )

    @classmethod
    def load_datamodule_params_from_yaml(cls, yaml_path: str) -> "DataModuleParams":
        """Load DataModuleParams from a YAML file."""
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls.extract_datamodule_params(config)

class NormalizationStage(Enum):
    TILE = "tile"
    IMAGE = "image"
    NONE = "none"

class ThresholdingStage(Enum):
    TILE = "tile"
    IMAGE = "image"

@dataclass 
class SeamSmoothingParams:
    apply: bool = True
    sigma: int = 2
    width: float = 0.1

@dataclass
class TilingPipelineConfig:
    image_size: Tuple[int, int]
    tile_size: Tuple[int, int]
    stride: Tuple[int, int]
    normalization_stage: NormalizationStage = NormalizationStage.IMAGE
    thresholding_stage: ThresholdingStage = ThresholdingStage.IMAGE
    seam_smoothing: SeamSmoothingParams = SeamSmoothingParams()

    @classmethod
    def _parse_enum(cls, enum_cls: type[Enum], value: Any, default: Enum) -> Enum:
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            return enum_cls(value)
        return default

    @classmethod
    def _parse_tuple(cls, value: Any, name: str) -> Tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError(f"{name} must be a tuple/list of length 2")

    @classmethod
    def _parse_seam_smoothing(cls, value: Any) -> SeamSmoothingParams:
        if isinstance(value, SeamSmoothingParams):
            return value
        if isinstance(value, dict):
            return SeamSmoothingParams(
                apply=bool(value.get("apply", True)),
                sigma=int(value.get("sigma", 2)),
                width=float(value.get("width", 0.1)),
            )
        raise TypeError("seam_smoothing must be a SeamSmoothingParams or a dict")

    @classmethod
    def extract_tiling_pipeline_params(cls, config: Dict[str, Any]) -> "TilingPipelineConfig":
        """Create a TilingPipelineConfig instance from a dictionary."""
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        tiling = config.get("tiling", {}) or {}
        if not isinstance(tiling, dict):
            raise TypeError("tiling config must be a dict")

        seam_smoothing = config.get("SeamSmoothing", config.get("seam_smoothing", {}))

        return cls(
            image_size=cls._parse_tuple(tiling.get("image_size"), "image_size"),
            tile_size=cls._parse_tuple(tiling.get("tile_size"), "tile_size"),
            stride=cls._parse_tuple(tiling.get("stride"), "stride"),
            normalization_stage=NormalizationStage(cls._parse_enum(NormalizationStage, tiling.get("normalization_stage"), NormalizationStage.IMAGE)),
            thresholding_stage=ThresholdingStage(cls._parse_enum(ThresholdingStage, tiling.get("thresholding_stage"), ThresholdingStage.IMAGE)),
            seam_smoothing=cls._parse_seam_smoothing(seam_smoothing),
        )

    @classmethod
    def load_tiling_pipeline_config_from_yaml(cls, yaml_path: str) -> "TilingPipelineConfig":
        """Load TilingPipelineConfig from a YAML file."""
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls.extract_tiling_pipeline_params(config)


@dataclass
class BaseModelConfig:
    name: Optional[str] = None
    backbone: Optional[str] = None
    pre_trained: Optional[bool] = None
    pre_processor: Optional[Any] = True
    post_processor: Optional[Any] = True
    evaluator: Optional[Any] = True
    visualizer: Optional[Any] = True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BaseModelConfig":
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        kwargs: Dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            if key in config:
                kwargs[key] = config[key]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def load_model_config_from_yaml(
        cls,
        yaml_path: str | Path,
        config_dir: Optional[Path] = None,
        copy_dir: Optional[Path] = None,
    ) -> "BaseModelConfig":
        """Load BaseModelConfig from a model YAML file and resolve engine config references."""
        model_yaml_path = Path(yaml_path)
        if config_dir is None:
            config_dir = model_yaml_path.parents[1] if len(model_yaml_path.parents) > 1 else model_yaml_path.parent

        init_args, _, _, _ = parse_model_init_args(
            model_yaml_path,
            config_dir=config_dir,
            copy_dir=copy_dir,
        )

        return cls.from_dict(init_args)

@dataclass
class CFAConfig(BaseModelConfig):
    gamma_c: int = 1
    gamma_d: int = 1
    num_nearest_neighbors: int = 3
    num_hard_negative_features: int = 3
    radius: float = 1e-5


@dataclass
class FastFlowConfig(BaseModelConfig):
    flow_steps: int = 8
    conv3x3_only: bool = False
    hidden_ratio: float = 1.0


@dataclass
class PadimConfig(BaseModelConfig):
    layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])
    n_features: Optional[int] = None


@dataclass
class PatchcoreConfig(BaseModelConfig):
    layers: List[str] = field(default_factory=lambda: ["layer2", "layer3"])
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9


@dataclass
class ReverseDistillationConfig(BaseModelConfig):
    layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])
    anomaly_map_mode: Optional[Any] = None


@dataclass
class STFPMConfig(BaseModelConfig):
    layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])


@dataclass
class InpFormerConfig(BaseModelConfig):
    encoder_name: str = "dinov2reg_vit_base_14"
    target_layers: Optional[List[int]] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9])
    fuse_layer_encoder: Optional[List[List[int]]] = field(default_factory=lambda: [[0, 1, 2, 3], [4, 5, 6, 7]])
    fuse_layer_decoder: Optional[List[List[int]]] = field(default_factory=lambda: [[0, 1, 2, 3], [4, 5, 6, 7]])
    remove_class_token: bool = True
    inp_num: int = 6


@dataclass
class GlassConfig(BaseModelConfig):
    input_shape: Tuple[int, int] = (288, 288)
    anomaly_source_path: Optional[str] = None
    pretrain_embed_dim: int = 1536
    target_embed_dim: int = 1536
    patchsize: int = 3
    patchstride: int = 1
    layers: Optional[List[str]] = field(default_factory=lambda: ["layer2", "layer3"])
    pre_projection: int = 1
    discriminator_layers: int = 2
    discriminator_hidden: int = 1024
    learning_rate: float = 0.0001
    step: int = 20
    svd: int = 0
    gaussian_noise_std: float = 0.015
    radius_quantile: float = 0.75
    focal_loss_quantile_threshold: float = 0.5
    mining: bool = True


MODEL_CONFIG_CLASSES: Dict[str, type[BaseModelConfig]] = {
    "cfa": CFAConfig,
    "fastflow": FastFlowConfig,
    "padim": PadimConfig,
    "patchcore": PatchcoreConfig,
    "reversedistillation": ReverseDistillationConfig,
    "reverse_distillation": ReverseDistillationConfig,
    "stfpm": STFPMConfig,
    "inpformer": InpFormerConfig,
    "glass": GlassConfig,
}


@dataclass
class ModelConfig:
    name: str
    params: BaseModelConfig

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        model_name = config.get("name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model config requires a non-empty name")

        params = config.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("model params must be a dict")

        model_cls = MODEL_CONFIG_CLASSES.get(model_name.lower())
        if model_cls is None:
            raise ValueError(f"Unsupported model '{model_name}'. Supported models: {', '.join(sorted(MODEL_CONFIG_CLASSES))}")

        return cls(name=model_name, params=model_cls.from_dict(params))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params.to_dict(),
        }


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: Any = "auto"
    num_nodes: int = 1
    precision: Optional[int] = None
    logger: Optional[Any] = None
    callbacks: Optional[List[Any]] = None
    fast_dev_run: bool = False
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Optional[Any] = None
    limit_train_batches: Optional[Any] = None
    limit_val_batches: Optional[Any] = None
    limit_test_batches: Optional[Any] = None
    limit_predict_batches: Optional[Any] = None
    overfit_batches: float = 0.0
    val_check_interval: Optional[float] = None
    check_val_every_n_epoch: int = 1
    num_sanity_val_steps: Optional[int] = None
    log_every_n_steps: Optional[int] = None
    enable_checkpointing: Optional[bool] = None
    enable_progress_bar: Optional[bool] = None
    enable_model_summary: Optional[bool] = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    deterministic: Optional[bool] = None
    benchmark: Optional[bool] = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    profiler: Optional[Any] = None
    detect_anomaly: bool = False
    barebones: bool = False
    plugins: Optional[Any] = None
    sync_batchnorm: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: Optional[str] = None
    # enable_autolog_hparams: bool = True
    # model_registry: Optional[Any] = None

    @staticmethod
    def instantiate_callback(callback_spec: Dict[str, Any]) -> Any:
        if not isinstance(callback_spec, dict):
            raise TypeError("callback spec must be a dict")

        class_path = callback_spec.get("class_path")
        if not isinstance(class_path, str):
            raise ValueError("callback spec requires a non-empty class_path string")

        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        callback_cls = getattr(module, class_name)

        init_args = callback_spec.get("init_args", {})
        if not isinstance(init_args, dict):
            raise TypeError("callback init_args must be a dict")

        return callback_cls(**init_args)

    @classmethod
    def instantiate_callbacks(cls, callbacks: Optional[List[Any]]) -> Optional[List[Any]]:
        if callbacks is None:
            return None

        instantiated: List[Any] = []
        for callback in callbacks:
            if isinstance(callback, dict) and "class_path" in callback:
                instantiated.append(cls.instantiate_callback(callback))
            else:
                instantiated.append(callback)
        return instantiated

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainerConfig":
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        kwargs: Dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            if key in config:
                if key == "callbacks":
                    kwargs[key] = cls.instantiate_callbacks(config[key])
                else:
                    kwargs[key] = config[key]
        return cls(**kwargs)

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a dict suitable for passing into Trainer(**kwargs)."""
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def extract_trainer_config(cls, config: Dict[str, Any]) -> "TrainerConfig":
        """Create a TrainerConfig instance from a dictionary."""
        return cls.from_dict(config)

    @classmethod
    def load_trainer_config_from_yaml(cls, yaml_path: str) -> "TrainerConfig":
        """Load TrainerConfig from a YAML file."""
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls.extract_trainer_config(config)



TModelConfig = TypeVar("TModelConfig", bound=BaseModelConfig)

@dataclass
class Product(Generic[TModelConfig]):
    name: str
    logFileName: str
    modelParameters: TModelConfig
    modelConfigPath: Path
    modelWeightsPath: Path
    modelTrainingDir: Path
    tilingPipelineConfig: TilingPipelineConfig
    trainerConfig: TrainerConfig
    inferencerConfig: TrainerConfig
    datasetConfig: DatasetConfig
    imageCrop: Optional[Tuple[int, int, int, int]] = None  # (x_min, y_min, x_max, y_max)
    id: str = generate_unique_id()
    enableTiling: bool = False

def load_product_from_yaml(product_yaml_path: str | Path, config_dir: Optional[Path] = None) -> Product["BaseModelConfig"]:
    product_yaml_path = Path(product_yaml_path)
    with product_yaml_path.open("r", encoding="utf-8") as f:
        product_config = yaml.safe_load(f)

    if not isinstance(product_config, dict):
        raise TypeError("product YAML must contain a mapping at the top level")

    if config_dir is None:
        config_dir = product_yaml_path.parent

    product_name = product_config.get("product")
    if not isinstance(product_name, str) or not product_name:
        raise ValueError("product field must be a non-empty string")

    logging_config = product_config.get("logging", {})
    if not isinstance(logging_config, dict):
        raise TypeError("logging config must be a dict")
    log_file_name = logging_config.get("logFileName")
    if not isinstance(log_file_name, str) or not log_file_name:
        raise ValueError("logging.logFileName must be a non-empty string")

    model_section = product_config.get("model")
    if not isinstance(model_section, dict):
        raise TypeError("model config must be a dict")

    model_config_name = model_section.get("config")
    if not isinstance(model_config_name, str) or not model_config_name:
        raise ValueError("model.config must be a non-empty string")
    model_config_path = resolve_product_config_path(model_config_name, product_yaml_path, config_dir, subdir="Models")

    with model_config_path.open("r", encoding="utf-8") as f:
        model_yaml = yaml.safe_load(f)
    if not isinstance(model_yaml, dict):
        raise TypeError("model YAML must contain a mapping at the top level")

    model_class_path = model_yaml.get("model", {}).get("class_path")
    if not isinstance(model_class_path, str) or not model_class_path:
        raise ValueError("model YAML must contain model.class_path")

    normalized_model_key = model_class_path.lower().replace("_", "")
    model_cls = MODEL_CONFIG_CLASSES.get(normalized_model_key)
    if model_cls is None:
        raise ValueError(f"Unsupported model '{model_class_path}' in model YAML")

    model_parameters = model_cls.load_model_config_from_yaml(model_config_path, config_dir=config_dir)

    weights_path = model_section.get("weights_path")
    if weights_path is None:
        raise ValueError("model.weights_path must be provided")
    model_weights_path = Path(weights_path)
    if not model_weights_path.is_absolute():
        model_weights_path = (product_yaml_path.parent / model_weights_path).resolve()

    training_dir = model_section.get("trainingDir")
    if not isinstance(training_dir, str) or not training_dir:
        raise ValueError("model.trainingDir must be a non-empty string")
    model_training_dir = Path(training_dir)
    if not model_training_dir.is_absolute():
        model_training_dir = (product_yaml_path.parent / model_training_dir).resolve()

    tiling_section = product_config.get("tiling", {})
    if not isinstance(tiling_section, dict):
        raise TypeError("tiling config must be a dict")
    enable_tiling = bool(tiling_section.get("enable", False))
    if enable_tiling:
        tiling_config_name = tiling_section.get("config")
        if not isinstance(tiling_config_name, str) or not tiling_config_name:
            raise ValueError("tiling.config must be a non-empty string when tiling.enable is true")
        tiling_config_path = resolve_product_config_path(tiling_config_name, product_yaml_path, config_dir, subdir="Tiling")
        tiling_pipeline_config = TilingPipelineConfig.load_tiling_pipeline_config_from_yaml(str(tiling_config_path))
    else:
        tiling_pipeline_config = TilingPipelineConfig(
            image_size=(0, 0),
            tile_size=(0, 0),
            stride=(0, 0),
        )

    trainer_section = product_config.get("trainer", {})
    if not isinstance(trainer_section, dict):
        raise TypeError("trainer config must be a dict")
    trainer_config_name = trainer_section.get("config")
    if not isinstance(trainer_config_name, str) or not trainer_config_name:
        raise ValueError("trainer.config must be a non-empty string")
    trainer_config_path = resolve_product_config_path(trainer_config_name, product_yaml_path, config_dir, subdir="Trainer")
    trainer_config = TrainerConfig.load_trainer_config_from_yaml(trainer_config_path)

    inferencer_section = product_config.get("inferencer", {})
    if not isinstance(inferencer_section, dict):
        raise TypeError("inferencer config must be a dict")
    inferencer_config_name = inferencer_section.get("config")
    if not isinstance(inferencer_config_name, str) or not inferencer_config_name:
        raise ValueError("inferencer.config must be a non-empty string")
    inferencer_config_path = resolve_product_config_path(inferencer_config_name, product_yaml_path, config_dir, subdir="Trainer")
    inferencer_config = TrainerConfig.load_trainer_config_from_yaml(inferencer_config_path)

    dataset_config = load_dataset_config(product_config.get("dataset", {}), product_yaml_path)

    return Product[
        BaseModelConfig
    ](
        name=product_name,
        logFileName=log_file_name,
        modelParameters=model_parameters,
        modelConfigPath=model_config_path,
        modelWeightsPath=model_weights_path,
        modelTrainingDir=model_training_dir,
        tilingPipelineConfig=tiling_pipeline_config,
        trainerConfig=trainer_config,
        inferencerConfig=inferencer_config,
        datasetConfig=dataset_config,
        enableTiling=enable_tiling,
    )


@dataclass
class Fault():
    """Fault Description"""
    type:str
    products:List[Product]
    size:Tuple[int, int]                                # in pixels
    location:Tuple[int,int]                             # in pixels
    angle:int       

if __name__ == "__main__":
    # Example usage
    dataset_config = DatasetConfig(
        name="MVTecADShort",
        category="cable",
        splits=("train", "test"),
        path=Path("datasets/MVTecADShort")
    )
    print(dataset_config)

    trainerConfig = TrainerConfig.load_trainer_config_from_yaml("configs/Trainer/Training_InpFormer.yaml")
    print(trainerConfig)
    from lightning.pytorch import Trainer
    trainer = Trainer(**trainerConfig.to_kwargs())
    print(trainer)

    tilingPipelineConfig = TilingPipelineConfig.load_tiling_pipeline_config_from_yaml("configs/Tiling/TiledEnsemble.yaml")
    print(tilingPipelineConfig)

    inpFormerConfig = InpFormerConfig.load_model_config_from_yaml("configs/Models/InpFormer.yaml")
    print(inpFormerConfig)
    model = create_model("InpFormer", inpFormerConfig.to_dict())
    print(model)

    product = load_product_from_yaml("configs/Products/cable.yaml")



