from dataclasses import asdict, dataclass, field
from pathlib import Path
import importlib
import os
import shutil
import numpy as np
import torchvision.transforms.v2 as T
from anomalib.metrics import Evaluator
from anomalib.pre_processing import PreProcessor
from anomalib.data.utils.split import ValSplitMode, TestSplitMode
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage, ThresholdingStage
from tiling.post_processor import AOIPostProcessor
from setup import define_metrics, create_model
from run_registry import resolve_run_dir
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, Tuple
import yaml
from enum import Enum
from cv2.typing import MatLike
import datetime
from anomalib.data import PredictDataset
from utils import find_first_file, exclude_from_logger, getVisualizer, VisualizerType

from data.anomaly_datasets import importDataset, exportDataset, FODataModule, FODataset, importPredictDataset



# FIFTYONE
import fiftyone.core.dataset as fod
import fiftyone as fo
import fiftyone.zoo as foz # zoo datasets and models
import fiftyone.brain as fob # ML methods
from fiftyone import ViewField as F # helper for defining views
from fiftyone import DatasetView
import logging

from typing import Generic, TypeVar

import uuid

logger = logging.getLogger(__name__)

import platform
import torch

def get_device():
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

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
    logger.info(f"Loading metrics from {config_path}")
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
    logger.info(f"Loaded val_metrics: {val_metrics}")
    logger.info(f"Loaded test_metrics: {test_metrics}")

    return val_metrics, test_metrics


def get_default_evaluator() -> Evaluator:
    val_metrics, test_metrics = define_metrics()
    return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)


def resolve_config_path(path_str: str, model_yaml_path: Path, config_dir: Optional[Path] = None) -> Path:
    logger.info(f"Resolving config path: {path_str}; config_dir: {config_dir}")
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
            logger.info(f"Success; Returning {candidate.resolve()}")
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to resolve path '{path_str}' from model YAML '{model_yaml_path}'"
    )


def copy_file_to_dir(source_path: Path, copy_dir: Path, base_dir: Optional[Path] = None) -> Path:
    logger.info(f"Trying to copy files from {source_path} to {copy_dir}")
    try:
        relative_path = source_path.relative_to(base_dir) if base_dir is not None else source_path.name
    except Exception:
        relative_path = source_path.name

    destination = copy_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    logger.info(f"Success!; Copied to {destination}")
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
) -> tuple[str, Dict[str, Any], Optional[Path], Optional[Path], Optional[Path]]:
    logger.info(f"Parsing model init_args from {model_yaml_path}" + f"; Copying to {copy_dir}" if copy_dir is not None else "")
    with open(model_yaml_path, "r") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("YAML model config must contain a mapping at the top level.")

    model_section = config.get("model")
    if not isinstance(model_section, dict):
        raise ValueError("YAML model config must contain a 'model' section as a mapping.")
    
    class_path:str = model_section.get("class_path", "")

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

    logger.info(f"Successfully parsed config: class_path:{class_path}, init_args: {init_args}")

    return class_path, init_args, pre_processor_path, post_processor_path, evaluator_path

def loadModelConfig(configDir:Path, modelConfigPath:Path, copyDir:Path|None=None, vtype:VisualizerType|None=None, fieldSize:Tuple[int,int]|None=None) -> Tuple[Dict[str,Any], Path|None, Path|None, Path|None]:
    """Load YAML model config and resolve engine config references."""
    class_path, init_args, pre_processor_path, post_processor_path, evaluator_path = parse_model_init_args(
        modelConfigPath,
        config_dir=configDir,
        copy_dir=copyDir,
    )

    if vtype is None:
        init_args["visualizer"] = False
    else:
        if fieldSize is None:
            raise ValueError(f"If Visualiser is to be created a fieldSize needs to be given; {fieldSize} not allowed")
        init_args["visualizer"] = getVisualizer(vtype=vtype, fieldSize=fieldSize)

    with open(modelConfigPath, 'r') as f:
        model_config = yaml.safe_load(f)

    model_config = dict(model_config)
    model_config["model"]["init_args"] = init_args

    return model_config, pre_processor_path, post_processor_path, evaluator_path

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

    @classmethod
    def load_dataset_config(cls, config: Dict[str, Any], product_yaml_path: Path) -> "DatasetConfig":
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
class DataModuleConfig:
    val_split_mode: ValSplitMode
    test_split_mode: TestSplitMode
    train_batch_size: Optional[int | str] = None  # int, or "auto" to probe the largest batch size that fits in GPU memory
    eval_batch_size: Optional[int | str] = None  # int, or "auto" (see train_batch_size)
    train_augmentations: Optional[List[Any]] = None
    val_augmentations: Optional[List[Any]] = None
    test_augmentations: Optional[List[Any]] = None
    augmentations: Optional[List[Any]] = None
    val_split_ratio: Optional[float] = None
    test_split_ratio: Optional[float] = None
    seed: Optional[int] = None
    num_workers: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d:Dict[str,Any] = {}
        for k,v in asdict(self).items():
            if v is None:
                continue
            if isinstance(v,Enum):
                d[k] = v.value
            else:
                d[k] = v
        return d

    @classmethod
    def extract_datamodule_config(cls, config: Dict[str, Any]) -> "DataModuleConfig":
        """Create a DataModuleConfig instance from a dictionary."""
        return cls(
            train_batch_size=config.get("train_batch_size"),
            eval_batch_size=config.get("eval_batch_size"),
            train_augmentations=config.get("train_augmentations"),
            val_augmentations=config.get("val_augmentations"),
            test_augmentations=config.get("test_augmentations"),
            augmentations=config.get("augmentations"),
            val_split_mode=ValSplitMode(config.get("val_split_mode", ValSplitMode.NONE)),
            val_split_ratio=config.get("val_split_ratio"),
            test_split_mode=TestSplitMode(config.get("test_split_mode", TestSplitMode.NONE)),
            test_split_ratio=config.get("test_split_ratio"),
            seed=config.get("seed"),
            # accept both "num_workers" and the legacy "max_workers" key
            num_workers=config.get("num_workers", config.get("max_workers")),
        )

    @classmethod
    def load_datamodule_config_from_yaml(cls, yaml_path: Path) -> "DataModuleConfig":
        """Load DataModuleConfig from a YAML file."""
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls.extract_datamodule_config(config)

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

@dataclass
class DatasetSession:
    """Manages FiftyOne dataset viewing and session lifecycle.
    Separate concern: how data is explored, not how models interact with it.
    """
    FO_DatasetOriginal: fo.Dataset
    FO_Dataset: fo.Dataset
    datasetName: str
    categories: List[str]
    category: Optional[List[str]] = None
    config: Optional[DatasetConfig] = None
    # FO_DatasetView: Optional[fo.DatasetView] = None
    AL_PredictDataset: Optional[PredictDataset] = None
    currentSession: Optional[fo.Session] = None  # fo.Session

    @classmethod
    def loadDatasetFromConfig(cls, config:DatasetConfig, overwrite:bool=True, merge:bool=False, split:Optional[Tuple[str]]=None) -> "DatasetSession":
        session = cls.loadDatasetFromDisk(datasetPath=config.path,
                                          datasetName=config.name, 
                                          overwrite=overwrite,
                                          merge=merge,
                                          split=config.splits if split is None else split)
        session.config=config
        return session  

    @classmethod
    def loadDatasetFromDatabase(cls, datasetName: str) -> "DatasetSession":
        """Load a dataset from MongoDB and track it here."""
        if fo.dataset_exists(datasetName):
            FO_Dataset = fo.load_dataset(datasetName)

            try:
                session:DatasetSession = cls(datasetName=datasetName, categories=list(FO_Dataset.distinct("category.label")), FO_DatasetOriginal=FO_Dataset, FO_Dataset=FO_Dataset)
            except:
                logger.error(f"Could not create DatasetSession.")
                raise ValueError
            
            logger.info(f"Loaded dataset '{datasetName}' from database!")
            return session
        else:
            logger.error(f"Dataset '{datasetName}' does not exist in database")
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
                datasetSession = cls.loadDatasetFromDatabase(datasetName=datasetName)  
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
                session = DatasetSession(datasetName=datasetName, FO_DatasetOriginal=FO_Dataset, FO_Dataset=FO_Dataset, AL_PredictDataset=AL_PredictDataset, categories=FO_Dataset.distinct("category.label"))
                logger.info(f"There are {session.FO_Dataset.count()} images in the {datasetName} dataset.")
                logger.info(f"There are {len(session.categories)} categorie(s) in the {datasetName} dataset.")
                logger.info(session.categories)
                # session.name = datasetName
                return session
            else:
                logger.warning(f"Import was not successfull")
                raise FileNotFoundError()

    def select_category(self, category: str) -> fo.Dataset | None:
        """Select a category and create a view."""
        

        if category == "all":
            self.category = None
            self.FO_DatasetView:fo.DatasetView = self.FO_Dataset.exists("file_path")
        else:
            if category not in (self.categories or []):
                raise AttributeError(f"Category {category} not found! Available: {self.categories}")
            self.category = category
            self.FO_DatasetView:fo.DatasetView = self.FO_Dataset.filter_labels("category", F("label").is_in([category]))
        
        logger.info(f"Selected category: {self.category}, {len(self.FO_DatasetView)} images")

        if self.FO_DatasetView is not None:
            try:
                FO_Dataset_:fo.Dataset = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
            except ValueError as e:
                logger.error(e)
                logger.info(f"Deleting {self.datasetName}-{self.category} from database and reloading.")
                fo.delete_dataset(f"{self.datasetName}-{self.category}")
                FO_Dataset_:fo.Dataset = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
            finally:
                if isinstance(FO_Dataset_, fo.Dataset):
                    FO_Dataset = FO_Dataset_
                else:
                    raise ValueError("FO_Dataset_ is not bound. Check for problems with dataset/datasetView (category/product selection)")
                self.FO_Dataset = FO_Dataset
                return self.FO_Dataset
        else:
            return None

        # return FO_Dataset

    def launchSession(self) -> Any:
        """Launch the FiftyOne app for exploration."""

        # if self.FO_DatasetView is not None: # Category is set which generates a datasetview
        #     # we make a separate dataset out of the View since some operations do not work on dataset views
        #     try:
        #         FO_Dataset_ = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
        #     except ValueError as e:
        #         logger.error(e)
        #         logger.info(f"Deleting {self.datasetName}-{self.category} from database and reloading.")
        #         fo.delete_dataset(f"{self.datasetName}-{self.category}")
        #         FO_Dataset_ = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
        #     finally:
        #         if isinstance(FO_Dataset_, fo.Dataset):
        #             FO_Dataset = FO_Dataset_
        #         else:
        #             raise ValueError("FO_Dataset_ is not bound. Check for problems with dataset/datasetView (category/product selection)")
        # else:
        #     FO_Dataset = self.FO_Dataset

        self.currentSession = fo.launch_app(self.FO_Dataset)
        logger.info(f"Session: {self.currentSession.server_address}:{self.currentSession.server_port}")
        return self.currentSession

    def get_dataset_for_operation(self) -> fo.Dataset:
        """Returns the dataset to use for training/eval: the view if a category is selected, else full dataset."""
        if self.FO_DatasetView is not None:
            return self.FO_DatasetView
        if self.FO_Dataset is None:
            raise AttributeError("No dataset loaded")
        return self.FO_Dataset
    
    def setupDatamodule(self, datamoduleConfig:DataModuleConfig, outputPath:Path) -> FODataModule:
        """Setup a datamodule from the dataset for running a model on the data

        Arguments:
            outputPath: Path

        Returns:
            _description_
        """

        if self.datasetName == "":
            self.datasetName = "unnamedDataset"
        dm_kwargs = datamoduleConfig.to_dict()
        # "auto" batch sizes are resolved later, per tile, by the tiled ensemble pipeline
        # (it needs the model and tile size to probe GPU memory). Use a safe placeholder
        # here so datamodule construction doesn't receive a non-int batch size.
        for key in ("train_batch_size", "eval_batch_size"):
            if dm_kwargs.get(key) == "auto":
                dm_kwargs[key] = 2
        datamodule = FODataModule(name=self.datasetName, samples=self.FO_Dataset, root=outputPath, **dm_kwargs)
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

@dataclass 
class SeamSmoothingConfig:
    apply: bool = True
    sigma: int = 2
    width: float = 0.1

@dataclass
class TilingPipelineConfig:
    image_size: Tuple[int, int]
    tile_size: Tuple[int, int]
    stride: Tuple[int, int]
    root_dir:Optional[Path] = None
    normalization_stage: NormalizationStage = NormalizationStage.IMAGE
    thresholding_stage: ThresholdingStage = ThresholdingStage.IMAGE
    seam_smoothing: SeamSmoothingConfig = SeamSmoothingConfig()

    def to_dict(self) -> Dict[str, Any]:
        d:Dict[str,Any] = {}
        for k,v in asdict(self).items():
            if v is None:
                continue
            if isinstance(v,Enum):
                d[k] = v.value
            else:
                d[k] = v
        return d

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
    def _parse_seam_smoothing(cls, value: Any) -> SeamSmoothingConfig:
        if isinstance(value, SeamSmoothingConfig):
            return value
        if isinstance(value, dict):
            return SeamSmoothingConfig(
                apply=bool(value.get("apply", False)),
                sigma=int(value.get("sigma", 1)),
                width=float(value.get("width", 0.0)),
            )
        raise TypeError(f"seam_smoothing must be a SeamSmoothingConfig or a dict. Instead got {type(value)}")

    @classmethod
    def extract_tiling_pipeline_params(cls, config: Dict[str, Any]) -> "TilingPipelineConfig":
        """Create a TilingPipelineConfig instance from a dictionary."""
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        tiling:Dict[str,Any]|None = config.get("tiling")
        if not isinstance(tiling, dict):
            raise TypeError("tiling config must be a dict")

        seam_smoothing:Optional[Dict[str,Any]] = tiling.get("SeamSmoothing", tiling.get("seam_smoothing", None))
        if seam_smoothing is None:
            Warning("Seem smoothing has not been found in manifest.yaml")
            seam_smoothing = {
                "apply" : True,
                "sigma" : 2,
                "width" : 0.1,
            }

        return cls(
            root_dir=config.get("root_dir", None),
            image_size=cls._parse_tuple(tiling.get("image_size"), "image_size"),
            tile_size=cls._parse_tuple(tiling.get("tile_size"), "tile_size"),
            stride=cls._parse_tuple(tiling.get("stride"), "stride"),
            normalization_stage=NormalizationStage(cls._parse_enum(NormalizationStage, tiling.get("normalization_stage"), NormalizationStage.IMAGE)),
            thresholding_stage=ThresholdingStage(cls._parse_enum(ThresholdingStage, tiling.get("thresholding_stage"), ThresholdingStage.IMAGE)),
            seam_smoothing=cls._parse_seam_smoothing(seam_smoothing),
        )

    @classmethod
    def load_tiling_pipeline_config_from_yaml(cls, yaml_path: Path) -> "TilingPipelineConfig":
        """Load TilingPipelineConfig from a YAML file."""
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls.extract_tiling_pipeline_params(config)


@dataclass
class BaseModelConfig:
    backbone: Optional[str] = None
    pre_trained: Optional[bool] = None
    pre_processor: Optional[Any] = False
    post_processor: Optional[Any] = False
    evaluator: Optional[Any] = False
    visualizer: Optional[Any] = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BaseModelConfig":
        # if not isinstance(config, dict):
        #     raise TypeError("config must be a dict")

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

        class_path, init_args, _, _, _ = parse_model_init_args(
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
    config: BaseModelConfig
    preProcessorPath: Optional[Path] = None
    postProcessorPath: Optional[Path] = None
    evaluatorPath: Optional[Path] = None

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        config_dir: Optional[Path] = None,
        copy_dir: Optional[Path] = None,
    ) -> "ModelConfig":
        model_yaml_path = Path(yaml_path)
        if config_dir is None:
            config_dir = (
                model_yaml_path.parents[1]
                if len(model_yaml_path.parents) > 1
                else model_yaml_path.parent
            )

        class_path, init_args, preProcessorPath, postProcessorPath, evaluatorPath = parse_model_init_args(
            model_yaml_path,
            config_dir=config_dir,
            copy_dir=copy_dir,
        )



        # Requires "name" to be present in the YAML's init_args
        # model_name = init_args.get("class_path")
        if not class_path:
            raise ValueError("YAML must contain a 'name' field")

        model_cls = MODEL_CONFIG_CLASSES.get(class_path.lower())
        if model_cls is None:
            raise ValueError(
                f"Unsupported model '{class_path}'. "
                f"Supported: {', '.join(sorted(MODEL_CONFIG_CLASSES))}"
            )

        return cls(name=class_path,
                   config=model_cls.from_dict(init_args),
                   preProcessorPath=preProcessorPath,
                   postProcessorPath=postProcessorPath,
                   evaluatorPath=evaluatorPath)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        # if not isinstance(config, dict):
        #     raise TypeError("config must be a dict")

        model_name = config.get("name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model config requires a non-empty name")

        config = config.get("config", {})
        # if not isinstance(config, dict):
        #     raise TypeError("model config must be a dict")

        model_cls = MODEL_CONFIG_CLASSES.get(model_name.lower())
        if model_cls is None:
            raise ValueError(f"Unsupported model '{model_name}'. Supported models: {', '.join(sorted(MODEL_CONFIG_CLASSES))}")

        return cls(name=model_name,
                   config=model_cls.from_dict(config))

    def to_dict(self) -> Dict[str, Any]:
        return self.config.to_dict()
        # return {
        #     "name": self.name,
        #     "config": self.config.to_dict(),
        # }


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
    profiler: Optional[Any] = "simple"
    detect_anomaly: bool = False
    barebones: bool = False
    plugins: Optional[Any] = None
    sync_batchnorm: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: Optional[str] = None
    # enable_autolog_hparams: bool = True
    # model_registry: Optional[Any] = None
    n_parallel_jobs: Optional[int] = None  # tiles to train in parallel; None = one per GPU

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
    def load_trainer_config_from_yaml(cls, yaml_path: Path) -> "TrainerConfig":
        """Load TrainerConfig from a YAML file."""
        with open(yaml_path, 'r') as file:
            config:dict[str,Any] = yaml.safe_load(file)
        if "accelerator" in config.keys():
            if config["accelerator"] == "auto":
                config["accelerator"] = get_device()
        return cls.extract_trainer_config(config)



TModelConfig = TypeVar("TModelConfig", bound=BaseModelConfig)

@dataclass
class Product:
    name: str
    logFileName: str
    modelConfig: ModelConfig
    modelConfigPath: Path
    modelTrainingDir: Optional[Path]
    tilingPipelineConfig: TilingPipelineConfig
    tilingConfigPath:Optional[Path]
    trainerConfig: TrainerConfig
    trainerConfigPath: Path
    datamoduleConfig: DataModuleConfig
    inferencerConfig: TrainerConfig
    inferencerConfigPath: Path
    datasetConfig: DatasetConfig
    datasetPath:Path
    imageCrop: Optional[Tuple[int, int, int, int]] = None  # (x_min, y_min, x_max, y_max)
    # id: str = generate_unique_id()
    enableTiling: bool = True
    selection: str = "latest"  # how modelWeightsPath/modelTrainingDir were picked when not pinned in the YAML: "latest" or "best"

    def __str__(self) -> str:
        string: str = ""
        for k,v in self.__dict__.items():
            string += f"{k}: {v}\n"
        return string

    def refresh_training_dir(self, baseOutputDir: Path) -> Optional[Path]:
        """
        Re-resolve modelTrainingDir against the current state of `baseOutputDir`,
        e.g. after a train()/eval() call may have changed which run counts as
        "latest"/"best". Only re-runs the resolution lookup - no config YAML is
        re-parsed and no model is rebuilt. Updates self.modelTrainingDir in place.

        Parameters
        ----------
        baseOutputDir : Path
            Root results directory to search for runs (e.g. manager.baseOutputDir)

        Returns
        -------
        _name_ : Optional[Path]
            The freshly resolved training directory, or None if no complete run
            for this product/model/selection exists (yet).
        """
        try:
            resolved = resolve_run_dir(
                baseOutputDir=Path(baseOutputDir),
                category=self.datasetConfig.category,
                modelName=self.modelConfig.name,
                selection=self.selection,
            )
        except FileNotFoundError:
            logger.info(
                f"refresh_training_dir: no complete '{self.selection}' run found for "
                f"{self.modelConfig.name}/{self.datasetConfig.category} under {baseOutputDir}"
            )
            resolved = None

        self.modelTrainingDir = resolved
        return resolved


def loadProductFromYaml(product_yaml_path: Path, config_dir: Optional[Path] = None, baseOutputDir: Optional[Path] = None) -> Product:
    with product_yaml_path.open("r", encoding="utf-8") as f:
        product_config = yaml.safe_load(f)

    if not isinstance(product_config, dict):
        raise TypeError("product YAML must contain a mapping at the top level")

    if config_dir is None:
        config_dir = product_yaml_path.parent.parent
        logger.info(f"Config dir not given for loadProductFromYaml. Choosing the double parent directory of the product.yaml: {config_dir}.")

    product_name:str = product_config.get("product")
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

    # with model_config_path.open("r", encoding="utf-8") as f:
    #     model_yaml = yaml.safe_load(f)
    # if not isinstance(model_yaml, dict):
    #     raise TypeError("model YAML must contain a mapping at the top level")

    # model_class_path = model_yaml.get("model", {}).get("class_path")
    # if not isinstance(model_class_path, str) or not model_class_path:
    #     raise ValueError("model YAML must contain model.class_path")

    # normalized_model_key = model_class_path.lower().replace("_", "")
    # model_cls = MODEL_CONFIG_CLASSES.get(normalized_model_key)
    # if model_cls is None:
    #     raise ValueError(f"Unsupported model '{model_class_path}' in model YAML")

    modelConfig:ModelConfig = ModelConfig.from_yaml(model_config_path, config_dir=config_dir)

    # modelConfig = model_cls.load_model_config_from_yaml(model_config_path, config_dir=config_dir)
    dataset_section = product_config.get("dataset", None)

    if dataset_section is None:
        raise AttributeError("No dataset section found in product yaml.")
    category = str(dataset_section.get("category", None)) if isinstance(dataset_section, dict) else None

    selection = model_section.get("selection", "latest")
    if selection not in ("latest", "best"):
        raise ValueError(f"model.selection must be 'latest' or 'best', got {selection!r}")

    training_dir = model_section.get("trainingDir", None)
    if training_dir is None:
        if baseOutputDir is None:
            raise ValueError(
                "trainingDir is not set in the product and baseOutputDir is not given to loadProductFromYaml function."
                "trainingDir must be given explicitly, or baseOutputDir must be given so the "
                f"'{selection}' run can be resolved automatically"
            )
        training_dir:Path|None = resolve_run_dir(
            baseOutputDir=Path(baseOutputDir),
            category=category,
            modelName=modelConfig.name,
            selection=selection,
        )

    # if weights_path
    # model_weights_path = Path(weights_path)
    # if not model_weights_path.is_absolute():
    #     model_weights_path = model_weights_path.resolve()

    model_training_dir:Path|None
    if training_dir is not None:
        model_training_dir = Path(training_dir)
        if not model_training_dir.is_absolute():
            model_training_dir = model_training_dir.resolve()
    else:
        print("No training directory found. Either because it is not given and no dir could be found automatically or because given one does not exist.")
        logger.info("No training directory found. Either because it is not given and no dir could be found automatically or because given one does not exist.")
        model_training_dir = None

    tiling_section = product_config.get("tiling", {})
    if not isinstance(tiling_section, dict):
        raise TypeError("tiling config must be a dict")
    enable_tiling = bool(tiling_section.get("enable", False))
    if enable_tiling:
        tiling_config_name = tiling_section.get("config")
        if not isinstance(tiling_config_name, str) or not tiling_config_name:
            raise ValueError("tiling.config must be a non-empty string when tiling.enable is true")
        tiling_config_path = resolve_product_config_path(tiling_config_name, product_yaml_path, config_dir, subdir="Tiling")
        tiling_pipeline_config = TilingPipelineConfig.load_tiling_pipeline_config_from_yaml(tiling_config_path)
    else:
        tiling_pipeline_config = TilingPipelineConfig(
            image_size=(0, 0),
            tile_size=(0, 0),
            stride=(0, 0),
        )
        tiling_config_path = None

    trainer_section = product_config.get("trainer", {})
    if not isinstance(trainer_section, dict):
        raise TypeError("trainer config must be a dict")
    trainer_config_name = trainer_section.get("config")
    if not isinstance(trainer_config_name, str) or not trainer_config_name:
        raise ValueError("trainer.config must be a non-empty string")
    trainer_config_path = resolve_product_config_path(trainer_config_name, product_yaml_path, config_dir, subdir="Trainer")
    trainer_config = TrainerConfig.load_trainer_config_from_yaml(trainer_config_path)
    datamoduleConfig = DataModuleConfig.load_datamodule_config_from_yaml(trainer_config_path)
    datasetConfig = DatasetConfig.load_dataset_config(dataset_section, product_yaml_path)
    datasetPath = datasetConfig.path

    inferencer_section = product_config.get("inferencer", None)
    if inferencer_section is None:
        raise ValueError("Need Inferencer for product. add `inferencer` section with path to `inferencer.yaml`")
    else:
        if not isinstance(inferencer_section, dict):
            # logger.warning("Inferencer section given but section is not a dict.")
            raise TypeError("inferencer config must be a dict")
        inferencer_config_name = inferencer_section.get("config")
        if not isinstance(inferencer_config_name, str) or not inferencer_config_name:
            raise ValueError("inferencer.config must be a non-empty string")
        inferencer_config_path = resolve_product_config_path(inferencer_config_name, product_yaml_path, config_dir, subdir="Trainer")
        inferencer_config = TrainerConfig.load_trainer_config_from_yaml(inferencer_config_path)


    return Product(
        name=product_name,
        logFileName=log_file_name,
        modelConfig=modelConfig,
        modelConfigPath=model_config_path,
        modelTrainingDir=model_training_dir,
        tilingPipelineConfig=tiling_pipeline_config,
        tilingConfigPath=tiling_config_path,
        trainerConfig=trainer_config,
        trainerConfigPath=trainer_config_path,
        datamoduleConfig=datamoduleConfig,
        inferencerConfig=inferencer_config,
        inferencerConfigPath=inferencer_config_path,
        datasetConfig=datasetConfig,
        datasetPath=datasetPath,
        enableTiling=enable_tiling,
        selection=selection,
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

    trainerConfig = TrainerConfig.load_trainer_config_from_yaml(Path("configs/Trainer/Training_InpFormer.yaml"))
    print(f"Trainerconfig {trainerConfig}")
    print("################")
    from lightning.pytorch import Trainer
    trainer = Trainer(**trainerConfig.to_kwargs())
    print(f"Trainerconfig {trainer}")

    DMConfig = DataModuleConfig.load_datamodule_config_from_yaml(Path("configs/Trainer/Training_InpFormer.yaml"))
    print(DMConfig)

    # from tiling.tiled_ensemble import get_ensemble_engine, parse_trainer_kwargs, AOITiledEnsembleEngine
    # engine:AOITiledEnsembleEngine = get_ensemble_engine((0,0), trainer.accelerator, 1, Path(trainer.default_root_dir), trainerConfig.to_kwargs())
    # # tilingPipelineConfig = TilingPipelineConfig.load_tiling_pipeline_config_from_yaml("configs/Tiling/TiledEnsemble.yaml")
    # # print(tilingPipelineConfig)

    modelConfig = ModelConfig.from_yaml("configs/Models/InpFormer.yaml")
    print(modelConfig)
    model = create_model(modelConfig.name, modelConfig.to_dict())
    print(model)

    # engine._setup_trainer(model)

    # print(engine)


    product = loadProductFromYaml("configs/Products/cable.yaml")

    print(product)



