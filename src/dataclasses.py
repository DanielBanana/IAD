from dataclasses import asdict, dataclass, field
import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, Tuple
import yaml
from enum import Enum
from cv2.typing import MatLike

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

@dataclass
class Product():
    """Product description
    """
    type:str
    dimensions: NDArray[np.float32]
    id:int
    image: Image

@dataclass
class Error():
    """Error Description"""
    type:str
    products:List[Product]
    size:Tuple[int, int]                                # in pixels
    location:Tuple[int,int]                             # in pixels
    angle:int                                           # rotational angle of the bounding box of the error; in degree; 0 is parallel to x axis to the right, 90 is parallel to y axis upwards


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
class TilingPipelineParams:
    image_size: Tuple[int, int]
    tile_size: Tuple[int, int]
    stride: Tuple[int, int]
    normalization_stage: NormalizationStage = NormalizationStage.IMAGE
    thresholding_stage: ThresholdingStage = ThresholdingStage.IMAGE
    seam_smoothing: SeamSmoothingParams = SeamSmoothingParams()

@dataclass
class BaseModelConfig:
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
    enable_autolog_hparams: bool = True
    model_registry: Optional[Any] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainerConfig":
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        kwargs: Dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            if key in config:
                kwargs[key] = config[key]
        return cls(**kwargs)

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a dict suitable for passing into Trainer(**kwargs)."""
        return {key: value for key, value in asdict(self).items() if value is not None}


def extract_trainer_config(config: Dict[str, Any]) -> TrainerConfig:
    """Create a TrainerConfig instance from a dictionary."""
    return TrainerConfig.from_dict(config)


def load_trainer_config_from_yaml(yaml_path: str) -> TrainerConfig:
    """Load TrainerConfig from a YAML file."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return extract_trainer_config(config)




