"""
For managing the data voxel51 is used which allows visualising the data in a web browser. For the anomaly detection anomalib datasets are needed.


"""

import csv
import os
import inspect
import itertools
import logging
import os
import random
import numpy as np
import pandas as pd


from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, Transform


from anomalib import TaskType
from anomalib.data import PredictDataset
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode, DirType
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import DirType, LabelName, Split
from anomalib.deploy import ExportType, OpenVINOInferencer, TorchInferencer, Inferencer
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, Stfpm

import eta.core.datasets as etad
import eta.core.image as etai
import eta.core.serial as etas
import eta.core.utils as etau
import eta.core.video as etav

import fiftyone as fo
import fiftyone.core.annotation as foa
import fiftyone.core.brain as fob
import fiftyone.core.dataset as fod
import fiftyone.core.evaluation as foe
import fiftyone.core.frame as fof
import fiftyone.core.groups as fog
import fiftyone.core.labels as fol
import fiftyone.core.media as fomm
import fiftyone.core.metadata as fom
import fiftyone.core.odm as foo
import fiftyone.core.runs as fors
import fiftyone.core.storage as fos
import fiftyone.core.utils as fou
import fiftyone.migrations as fomi
import fiftyone.types as fot
import fiftyone.utils.data as foud
import fiftyone.zoo as foz # zoo datasets and models
from fiftyone.core.sample import Sample
from fiftyone.utils.data.importers import LabeledImageDatasetImporter
from fiftyone.utils.data.exporters import LabeledImageDatasetExporter, GenericSampleDatasetExporter
from fiftyone import ViewField as F # helper for defining views


logger = logging.getLogger(__name__)

class MVTecStyleDataImporter(LabeledImageDatasetImporter):
    """Importer for an image classification directory tree stored on disk.

    See :ref:`this page <ImageClassificationDirectoryTree-import>` for format
    details.

    Args:
        dataset_dir: the dataset directory
        compute_metadata (False): whether to produce
            :class:`fiftyone.core.metadata.ImageMetadata` instances for each
            image when importing
        classes (None): an optional string or list of strings specifying a
            subset of classes to load
        unknown ("_unknown"): the name of the subdirectory containing
            unknown images
        shuffle (False): whether to randomly shuffle the order in which the
            samples are imported
        seed (None): a random seed to use when shuffling
        max_samples (None): a maximum number of samples to import. By default,
            all samples are imported
    """

    def __init__(
        self,
        dataset_dir,
        compute_metadata=False,
        classes=None,
        blacklist=["ground_truth"],
        unknown="_unknown",
        shuffle=False,
        seed=None,
        max_samples=None,
    ):
        classes = _to_list(classes)

        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )

        self.compute_metadata = compute_metadata
        self.classes = classes
        self.unknown = unknown
        self.blacklist = blacklist

        self._classes = None
        self._samples = None
        self._iter_samples = None
        self._num_samples = None

    def __iter__(self):
        self._iter_samples = iter(self._samples)
        return self

    def __len__(self):
        return self._num_samples

    def __next__(self):
        paths, category, _split, _anomalyType = next(self._iter_samples)

        if self.compute_metadata:
            image_metadata = fom.ImageMetadata.build_for(paths["image"])
        else:
            image_metadata = None
        if _anomalyType == "normal" or _anomalyType == "good":
            label_index = LabelName.NORMAL
        elif _anomalyType == self.unknown:
            label_index = LabelName.UNKNOWN
        else:
            label_index = LabelName.ABNORMAL
        
        if _anomalyType is not None:
            anomalyType = fol.Classification(label=_anomalyType)
        if category is not None:
            category = fol.Classification(label=category)
        if _split == "train":
            split = Split.TRAIN
        elif _split == "test":
            split = Split.TEST
        elif _split == "val" or _split == "validation":
            split = Split.VAL
        else:
            raise ValueError()

        return paths, image_metadata, category, split, anomalyType, label_index

    @property
    def has_image_metadata(self):
        return self.compute_metadata

    @property
    def has_dataset_info(self):
        return True

    @property
    def label_cls(self):
        return fol.Classification

    def setup(self):
        samples = []
        classes = set()
        categories = set()
        whitelist = set(self.classes) if self.classes is not None else None
        for relpath in etau.list_files(self.dataset_dir, recursive=True):
            chunks = relpath.split(os.path.sep, 3)

            paths = {}

            if len(chunks) <= 3:
                continue

            category, split, anomalyType, file = relpath.split(os.path.sep, 3)

            if anomalyType.startswith("."):
                continue
            if whitelist is not None and anomalyType not in whitelist:
                continue
            if split == "ground_truth":
                continue
            if split == "test":
                # Look for ground truth segmentation masks
                nr, ext = file.split(".")
                gtPath = os.path.join(category, "ground_truth", anomalyType, nr + "_mask." + ext)
                gtPath = os.path.join(self.dataset_dir, gtPath)
                if os.path.exists(gtPath):
                    paths["ground_truth"] = gtPath                
            path = os.path.join(self.dataset_dir, relpath)
            paths["image"] = path
            if anomalyType != self.unknown:
                classes.add(anomalyType)

            categories.add(category)

            samples.append((paths, category, split, anomalyType))

        samples = self._preprocess_list(samples)

        if whitelist is not None:
            classes = self.classes
        else:
            classes = sorted(classes)

        self._classes = classes
        self._categories = sorted(categories)
        self._samples = samples
        self._num_samples = len(samples)

    def get_dataset_info(self):
        return {"classes": self._classes}

    @staticmethod
    def _get_classes(dataset_dir):
        # Used only by dataset zoo
        return sorted(etau.list_subdirs(dataset_dir))

    @staticmethod
    def _get_num_samples(dataset_dir):
        # Used only by dataset zoo
        return len(etau.list_files(dataset_dir, recursive=True))

class TrainTestDataImporter(LabeledImageDatasetImporter):
    """Importer for an image classification directory tree stored on disk.

    See :ref:`this page <ImageClassificationDirectoryTree-import>` for format
    details.

    Args:
        dataset_dir: the dataset directory
        compute_metadata (False): whether to produce
            :class:`fiftyone.core.metadata.ImageMetadata` instances for each
            image when importing
        classes (None): an optional string or list of strings specifying a
            subset of classes to load
        unknown ("_unknown"): the name of the subdirectory containing
            unknown images
        shuffle (False): whether to randomly shuffle the order in which the
            samples are imported
        seed (None): a random seed to use when shuffling
        max_samples (None): a maximum number of samples to import. By default,
            all samples are imported
    """

    def __init__(
        self,
        dataset_dir,
        compute_metadata=False,
        classes=None,
        blacklist=["ground_truth"],
        # unknown="_unknown",
        shuffle=False,
        seed=None,
        max_samples=None,
    ):
        classes = _to_list(classes)

        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )

        self.compute_metadata = compute_metadata
        self.classes = classes
        self.unknown = "unknown"
        self.blacklist = blacklist

        self._classes = None
        self._samples = None
        self._iter_samples = None
        self._num_samples = None

    def __iter__(self):
        self._iter_samples = iter(self._samples)
        return self

    def __len__(self):
        return self._num_samples

    def __next__(self):
        paths, _category, _split, _anomalyType = next(self._iter_samples)

        if self.compute_metadata:
            image_metadata = fom.ImageMetadata.build_for(paths["image"])
        else:
            image_metadata = None
        if _anomalyType == "normal" or _anomalyType == "good":
            label_index = LabelName.NORMAL
        elif _anomalyType == self.unknown:
            label_index = LabelName.UNKNOWN
        else:
            label_index = LabelName.ABNORMAL
        if _anomalyType is not None:
            anomalyType = fol.Classification(label=_anomalyType)
        if _category is not None:
            category = fol.Classification(label=_category)
        if _split == "train":
            split = Split.TRAIN
        elif _split == "test":
            split = Split.TEST
        elif _split == "val" or _split == "validation":
            split = Split.VAL
        else:
            raise ValueError()

        return paths, image_metadata, category, split, anomalyType, label_index

    @property
    def has_image_metadata(self):
        return self.compute_metadata

    @property
    def has_dataset_info(self):
        return True

    @property
    def label_cls(self):
        return fol.Classification

    def setup(self):
        """Load the dataset from disk.
        """
        samples = []
        classes = set()
        whitelist = set(self.classes) if self.classes is not None else None
        for relpath in etau.list_files(self.dataset_dir, recursive=True):
            chunks = relpath.split(os.path.sep, 3)

            # category = self.dataset_dir.split(os.path.sep)[-1]

            paths = {}

            if len(chunks) <= 2:
                continue

            if len(chunks) == 3:
                # Prediction case, there is no label like good or anomaly because we do not know
                category, split, file = relpath.split(os.path.sep)
                anomalyType = "unknown"

            if len(chunks) == 4:
                # Normal case
                category, split, anomalyType, file = relpath.split(os.path.sep)
                if anomalyType.startswith("."):
                    continue
                if whitelist is not None and anomalyType not in whitelist:
                    continue

            if split == "test":
                # Look for ground truth segmentation masks
                nr, ext = file.split(".")
                gtPath = os.path.join(category, "ground_truth", anomalyType, nr + "_mask." + ext)
                gtPath = os.path.join(self.dataset_dir, gtPath)
                if os.path.exists(gtPath):
                    paths["ground_truth"] = gtPath
            elif split == "train":
                pass
            else:
                continue

            path = os.path.join(self.dataset_dir, relpath)
            paths["image"] = path
            if anomalyType == self.unknown:
                anomalyType = None
            else:
                classes.add(anomalyType)

            samples.append((paths, category, split, anomalyType))

        samples = self._preprocess_list(samples)

        if whitelist is not None:
            classes = self.classes
        else:
            classes = sorted(classes)

        self._classes = classes
        self._samples = samples
        self._num_samples = len(samples)

    def get_dataset_info(self):
        return {"classes": self._classes}

    @staticmethod
    def _get_classes(dataset_dir):
        # Used only by dataset zoo
        return sorted(etau.list_subdirs(dataset_dir))

    @staticmethod
    def _get_num_samples(dataset_dir):
        # Used only by dataset zoo
        return len(etau.list_files(dataset_dir, recursive=True))

class TestDataImporter(LabeledImageDatasetImporter):
    """Importer for an image classification directory tree stored on disk.

    See :ref:`this page <ImageClassificationDirectoryTree-import>` for format
    details.

    Args:
        dataset_dir: the dataset directory
        compute_metadata (False): whether to produce
            :class:`fiftyone.core.metadata.ImageMetadata` instances for each
            image when importing
        classes (None): an optional string or list of strings specifying a
            subset of classes to load
        unknown ("_unknown"): the name of the subdirectory containing
            unknown images
        shuffle (False): whether to randomly shuffle the order in which the
            samples are imported
        seed (None): a random seed to use when shuffling
        max_samples (None): a maximum number of samples to import. By default,
            all samples are imported
    """

    def __init__(
        self,
        dataset_dir,
        compute_metadata=False,
        classes=None,
        blacklist=["ground_truth"],
        unknown="_unknown",
        shuffle=False,
        seed=None,
        max_samples=None,
    ):
        classes = _to_list(classes)

        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )

        self.compute_metadata = compute_metadata
        self.classes = classes
        self.unknown = unknown
        self.blacklist = blacklist

        self._classes = None
        self._samples = None
        self._iter_samples = None
        self._num_samples = None

    def __iter__(self):
        self._iter_samples = iter(self._samples)
        return self

    def __len__(self):
        return self._num_samples

    def __next__(self):
        paths, _category, _split, _anomalyType = next(self._iter_samples)

        if self.compute_metadata:
            image_metadata = fom.ImageMetadata.build_for(paths["image"])
        else:
            image_metadata = None
        if _anomalyType == "normal" or _anomalyType == "good":
            label_index = LabelName.NORMAL
        elif _anomalyType == self.unknown:
            label_index = LabelName.UNKNOWN
        else:
            label_index = LabelName.ABNORMAL
        if _anomalyType is not None:
            anomalyType = fol.Classification(label=_anomalyType)
        if _category is not None:
            category = fol.Classification(label=_category)
        if _split == "train":
            split = Split.TRAIN
        elif _split == "test":
            split = Split.TEST
        elif _split == "val" or _split == "validation":
            split = Split.VAL
        else:
            raise ValueError()

        return paths, image_metadata, category, split, anomalyType, label_index

    @property
    def has_image_metadata(self):
        return self.compute_metadata

    @property
    def has_dataset_info(self):
        return True

    @property
    def label_cls(self):
        return fol.Classification

    def setup(self):
        samples = []
        classes = set()
        whitelist = set(self.classes) if self.classes is not None else None
        for relpath in etau.list_files(self.dataset_dir, recursive=True):
            chunks = relpath.split(os.path.sep, 3)

            category = self.dataset_dir.split(os.path.sep)[-1]

            paths = {}

            if len(chunks) <= 2:
                continue

            split, anomalyType, file = relpath.split(os.path.sep, 2)

            if anomalyType.startswith("."):
                continue
            if whitelist is not None and anomalyType not in whitelist:
                continue
            if split == "test":
                # Look for ground truth segmentation masks
                nr, ext = file.split(".")
                gtPath = os.path.join("ground_truth", anomalyType, nr + "_mask." + ext)
                gtPath = os.path.join(self.dataset_dir, gtPath)
                if os.path.exists(gtPath):
                    paths["ground_truth"] = gtPath 
            else:
                # skip non-test folders
                continue           
            path = os.path.join(self.dataset_dir, relpath)
            paths["image"] = path
            if anomalyType != self.unknown:
                classes.add(anomalyType)

            samples.append((paths, category, split, anomalyType))

        samples = self._preprocess_list(samples)

        if whitelist is not None:
            classes = self.classes
        else:
            classes = sorted(classes)

        self._classes = classes
        self._samples = samples
        self._num_samples = len(samples)

    def get_dataset_info(self):
        return {"classes": self._classes}

    @staticmethod
    def _get_classes(dataset_dir):
        # Used only by dataset zoo
        return sorted(etau.list_subdirs(dataset_dir))

    @staticmethod
    def _get_num_samples(dataset_dir):
        # Used only by dataset zoo
        return len(etau.list_files(dataset_dir, recursive=True))
    
class AnomalyImageTreeExporter(GenericSampleDatasetExporter):
    """Interface for exporting datasets of arbitrary
    :class:`fiftyone.core.sample.Sample` instances.

    See :ref:`this page <writing-a-custom-dataset-exporter>` for information
    about implementing/using dataset exporters.

    Args:
        export_dir (None): the directory to write the export. This may be
            optional for some exporters
    """
    def __init__(self, export_dir=None):
        super().__init__(export_dir)
        self.setup()

    def setup(self):
        """Performs any necessary setup before exporting the first sample in
        the dataset.

        This method is called when the exporter's context manager interface is
        entered, :func:`DatasetExporter.__enter__`.
        """
        self._data_dir = self.export_dir
        self._labels_path = os.path.join(self.export_dir, "labels.csv")
        self._labels = [] 

        # The `ImageExporter` utility class provides an `export()` method
        # that exports images to an output directory with automatic handling
        # of things like name conflicts
        self._image_exporter = foud.ImageExporter(
            True, export_path=self._data_dir, default_ext=".png",
        )
        self._image_exporter.setup()

    def export_sample(self, sample:fo.Sample):
        """Exports the given sample to the dataset.

        Args:
            sample: a :class:`fiftyone.core.sample.Sample`
        """
        file = sample.filepath.split(os.path.sep)[-1]
        
        category = sample["category"].label
        anomalyType = sample["anomalyType"].label
        split = sample["split"]

        outpath = os.path.join(self._data_dir, category, split, anomalyType, file)

        if sample.metadata is None:
            metadata = fo.ImageMetadata.build_for(sample.filepath)
        else:
            metadata = sample.metadata

        self._labels.append((
            out_image_path,
            metadata.size_bytes,
            metadata.mime_type,
            metadata.width,
            metadata.height,
            metadata.num_channels,
            category,
            split,
            anomalyType,
            #sample.tags[0]
        ))

        out_image_path, _ = self._image_exporter.export(sample.filepath, outpath=outpath)

        
    def close(self, *args):
        """Performs any necessary actions after the last sample has been
        exported.

        This method is called when the exporter's context manager interface is
        exited, :func:`DatasetExporter.__exit__`.

        Args:
            *args: the arguments to :func:`DatasetExporter.__exit__`
        """
        # Ensure the base output directory exists
        basedir = os.path.dirname(self._labels_path)
        if basedir and not os.path.isdir(basedir):
            os.makedirs(basedir)

        # Write the labels CSV file
        with open(self._labels_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filepath",
                "size_bytes",
                "mime_type",
                "width",
                "height",
                "num_channels",
                "category",
                "anomalyType",
                "split"                
            ])
            for row in self._labels:
                writer.writerow(row)

class FODataModule(AnomalibDataModule):
    """FiftyOne (51) DataModule.

    Args:
        name (str): Name of the dataset. Used for logging/saving.
        samples (Dataset): Fiftyone ``Dataset``
        root (str | Path | None): Root folder containing normal and abnormal
            directories. Defaults to ``None``.
        normal_split_ratio (float): Ratio to split normal training images for
            test set when no normal test images exist.
            Defaults to ``0.2``.
        train_batch_size (int): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int): Validation/test batch size.
            Defaults to ``32``.
        num_workers (int): Number of workers for data loading.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Method to obtain test subset.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of train images for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to obtain validation subset.
            Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float): Fraction of images for validation.
            Defaults to ``0.5``.
        seed (int | None): Random seed for splitting.
            Defaults to ``None``.

    Example:
        Create and setup a tabular datamodule::

            >>> from anomalib.data import Tabular
            >>> samples = {
            ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
            ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
            ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
            ... }
            >>> datamodule = Fiftyone(
            ...     name="custom",
            ...     samples=samples,
            ...     root="./datasets/custom",
            ... )
            >>> datamodule.setup()

        Get a batch from train dataloader::

            >>> batch = next(iter(datamodule.train_dataloader()))
            >>> batch.keys()
            dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])

        Get a batch from test dataloader::

            >>> batch = next(iter(datamodule.test_dataloader()))
            >>> batch.keys()
            dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])
    """

    def __init__(
        self,
        name: str,
        samples: fod.Dataset,
        root: str | Path | None = None,
        normal_split_ratio: float = 0.2,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._name = name
        self.root = root
        self._unprocessed_samples = samples
        test_split_mode = TestSplitMode(test_split_mode)
        val_split_mode = ValSplitMode(val_split_mode)
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.normal_split_ratio = normal_split_ratio

    def _dataloader_kwargs(self) -> dict[str, Any]:
        """Extra DataLoader kwargs the base AnomalibDataModule leaves off.

        ``pin_memory`` speeds up host->device transfer (page-locked staging buffers),
        and ``persistent_workers`` keeps worker processes alive across epochs instead
        of respawning them every epoch - both reduce CPU-side overhead that would
        otherwise leave the GPU waiting between batches.
        """
        return {
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader (see `_dataloader_kwargs` for why this differs from the base)."""
        return DataLoader(
            dataset=self.train_data,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.external_collate_fn or self.train_data.collate_fn,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader (see `_dataloader_kwargs` for why this differs from the base)."""
        return DataLoader(
            dataset=self.val_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.external_collate_fn or self.val_data.collate_fn,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader (see `_dataloader_kwargs` for why this differs from the base)."""
        return DataLoader(
            dataset=self.test_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.external_collate_fn or self.test_data.collate_fn,
            **self._dataloader_kwargs(),
        )

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = FODataset(
            name=self.name,
            samples=self._unprocessed_samples,
            split=Split.TRAIN,
            root=self.root,
        )

        self.test_data = FODataset(
            name=self.name,
            samples=self._unprocessed_samples,
            split=Split.TEST,
            root=self.root,
        )

    @property
    def name(self) -> str:
        """Get name of the datamodule.

        Returns:
            Name of the datamodule.
        """
        return self._name

    def __getstate__(self):
        state = self.__dict__.copy()
        # fod.Dataset uses mongoengine metaclasses that can't be pickled (needed for
        # multiprocessing spawn). By the time a job is submitted to ParallelRunner,
        # setup() has already run (train_data/val_data/test_data are populated and
        # _is_setup=True), so the subprocess never needs _unprocessed_samples.
        state['_unprocessed_samples'] = None
        return state

    @classmethod
    def from_file(
        cls: type["FODataset"],
        name: str,
        file_path: str | Path,
        file_format: str | None = None,
        pd_kwargs: dict | None = None,
        **kwargs,
    ) -> "FODataset":
        """Create Tabular Datamodule from file.

        Args:
            name (str): Name of the dataset. This is used to name the datamodule,
                especially when logging/saving.
            file_path (str | Path): Path to tabular file containing the datset
                information.
            file_format (str): File format supported by a pd.read_* method, such
                as ``csv``, ``parquet`` or ``json``.
                Defaults to ``None`` (inferred from file suffix).
            pd_kwargs (dict | None): Keyword argument dictionary for the pd.read_* method.
                Defaults to ``None``.
            kwargs (dict): Additional keyword arguments for the Tabular Datamodule class.

        Returns:
            Tabular: Tabular Datamodule

        Example:
            Prepare a tabular file (such as ``samples.csv`` or ``samples.parquet``) with the
            following columns: ``image_path`` (absolute or relative to ``root``), ``label_index``
            (``0`` for normal, ``1`` for anomalous samples), and ``split`` (``train`` or ``test``).
            For segmentation tasks, also include a ``mask_path`` column.

            From this file, create and setup a tabular datamodule::

                >>> from anomalib.data import Tabular
                >>> datamodule = Tabular.from_file(
                ...     name="custom",
                ...     file_path="./samples.csv",
                ...     root="./datasets/custom",
                ... )
                >>> datamodule.setup()

            Get a batch from train dataloader::

                >>> batch = next(iter(datamodule.train_dataloader()))
                >>> batch.keys()
                dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])

            Get a batch from test dataloader::

                >>> batch = next(iter(datamodule.test_dataloader()))
                >>> batch.keys()
                dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])
        """
        raise NotImplementedError

        # Check if file exists
        if not Path(file_path).is_file():
            msg = f"File not found: '{file_path}'"
            raise FileNotFoundError(msg)

        # Infer file_format and check if supported
        file_format = file_format or Path(file_path).suffix[1:]
        if not file_format:
            msg = f"File format not specified and could not be inferred from file name: '{Path(file_path).name}'"
            raise ValueError(msg)
        read_func = getattr(pd, f"read_{file_format}", None)
        if read_func is None:
            msg = f"Unsupported file format: '{file_format}'"
            raise ValueError(msg)

        # Read the file and return Tabular dataset
        pd_kwargs = pd_kwargs or {}
        samples = read_func(file_path, **pd_kwargs)
        return cls(name, samples, **kwargs)



from anomalib.data.utils import LabelName, read_image, read_mask
from torchvision.tv_tensors import Mask
import torch
from anomalib.data.dataclasses import DatasetItem, ImageBatch, ImageItem



class FODataset(AnomalibDataset):
    """Dataset class for loading images from paths and labels defined in a fiftyone Dataset.

    Args:
        name (str): Name of the dataset. Used for logging/saving.
        samples (dict | list | DataFrame): Pandas ``DataFrame`` or compatible ``list``
            or ``dict`` containing the dataset information.
        augmentations (Transform | None, optional): Augmentations to apply to the images.
            Defaults to ``None``.
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            Choose from ``Split.FULL``, ``Split.TRAIN``, ``Split.TEST``.
            Defaults to ``None``.

    Examples:
        Create a classification dataset:

        >>> from anomalib.data.utils import InputNormalizationMethod, get_transforms
        >>> from anomalib.data.datasets import FiftyOneDataset
        >>> transform = get_transforms(
        ...     image_size=256,
        ...     normalization=InputNormalizationMethod.NONE
        ... )
        >>> samples = {
        ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
        ... }
        >>> dataset = FiftyOneDataset(
        ...     name="custom",
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     transform=transform
        ... )

        Create a segmentation dataset:

        >>> samples = {
        ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
        ...     "mask_path": ["masks/mask1.png", "masks/mask2.png", "masks/mask3.png", ... ],
        ... }
        >>> dataset = FiftyOne(
        ...     name="custom",
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     transform=transform
        ... )
    """

    def __init__(
        self,
        name: str,
        samples: fod.Dataset,
        augmentations: Transform | None = None,
        root: str | Path | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self._name = name
        self.split = split
        self.root = root
        self.samples = make_fiftyone_dataset(
            samples=samples,
            root=self.root,
            split=self.split,
        )

    @property
    def name(self) -> str:
        """Get dataset name.

        Returns:
            str: Name of the dataset
        """
        return self._name
    
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __getitem__(self, index: int) -> DatasetItem:
        """Get dataset item for the given index.

        Args:
            index (int): Index to get the item.

        Returns:
            DatasetItem: Dataset item containing image and ground truth (if available).

        Example:
            >>> dataset = AnomalibDataset()
            >>> item = dataset[0]
            >>> isinstance(item.image, torch.Tensor)
            True
        """
        image_path = self.samples.iloc[index].image_path
        mask_path = self.samples.iloc[index].mask_path
        label_index = self.samples.iloc[index].label_index

        # Read the image
        image = read_image(image_path, as_tensor=True)

        # Initialize mask as None
        gt_mask = None

        # Process based on task type
        if self.task == TaskType.SEGMENTATION:
            if label_index == LabelName.NORMAL:
                # Create zero mask for normal samples
                gt_mask = Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
            elif label_index == LabelName.ABNORMAL:
                if mask_path:
                    gt_mask = read_mask(mask_path, as_tensor=True)
                else:
                    gt_mask = Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
            # For UNKNOWN, gt_mask remains None

        # Apply augmentations if available
        if self.augmentations:
            if self.task == TaskType.CLASSIFICATION:
                image = self.augmentations(image)
            elif self.task == TaskType.SEGMENTATION:
                # For augmentations that require both image and mask:
                # - Use a temporary zero mask for UNKNOWN samples
                # - But preserve the final gt_mask as None for UNKNOWN
                temp_mask = gt_mask if gt_mask is not None else Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
                image, augmented_mask = self.augmentations(image, temp_mask)
                # Only update gt_mask if it wasn't None before augmentations
                if gt_mask is not None:
                    gt_mask = augmented_mask

        # Create gt_label tensor (None for UNKNOWN)
        gt_label = None if label_index == LabelName.UNKNOWN else torch.tensor(label_index)

        imageItem = ImageItem(
            image=image,
            gt_mask=gt_mask,
            gt_label=gt_label,
            image_path=image_path,
            mask_path=mask_path,
        )

        imageItem.id = self.samples.iloc[index].id

        # Return the dataset item
        return imageItem

def make_fiftyone_dataset(
    samples: fod.Dataset,
    root: str | Path | None = None,
    split: str | Split | None = None,
) -> pd.DataFrame:
    """Create a dataset from a fiftyone dataset.

    Args:
        samples (Dataset): FityOne ``Dataset``
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            Choose from ``Split.FULL``, ``Split.TRAIN``, ``Split.TEST``.
            Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns for image paths, labels, splits
            and mask paths (for segmentation).

    """
    ######################
    ### Pre-processing ###
    ######################

    # Convert to pandas DataFrame if dictionary or list is given

    # Check if samples contain image_path column
    columns:Set[str] = set()
    if "filepath" in samples.get_field_schema():
        columns.add("image_path")
    else:
        msg = "The _samples must each contain an 'filepath' field."
        logger.error(msg)
        raise ValueError(msg)
    if "label_index" in samples.get_field_schema():
        columns.add("label_index")
    else:
        msg = "The samples must each contain a 'label_index' field. Either 0:('normal'), 1:('abnormal') or -1('unknown')."
        logger.error(msg)
        raise ValueError(msg)
    if "split" in samples.get_field_schema():
        columns.add("split")
    else:
        msg = "The samples must each contain a 'split' field. Either 'train', 'val', 'test'."
        logger.error(msg)
        raise ValueError(msg)
    # if "mask_path" in samples.get_field_schema():
    columns.add("mask_path")
    columns.add("label")
    columns.add("id")

    _samples = pd.DataFrame(columns=list(columns))

    for sample in samples:
        sampleDict = {column: None for column in columns}
        if not hasattr(sample, "filepath"):
            msg = "The samples must each contain a filepath that is not empty."
            raise ValueError(msg)
        else:
            sample["image_path"] = sample.filepath
            sampleDict["image_path"] = sample["image_path"] 
            
        if sample.get_field_schema().get("mask_path", None) is not None:
            sample["mask_path"] = sample.mask_path
            sampleDict["mask_path"] = sample["mask_path"] 
        else:
            sample["mask_path"] = ""
            sampleDict["mask_path"] = ""

        # if sample.tags;
        if sample.get_field_schema().get("label_index", None) is None:
            msg = "The samples must each contain a 'label_index' field. Either 0:('normal'), 1:('abnormal') or -1('unknown')."
            raise ValueError(msg)
        else:
            sampleDict["label_index"] = sample["label_index"] 

        if sample.get_field_schema().get("split", None) is None:
            msg = "The samples must each contain a 'split' field. Either 'train', 'val', 'test'."
            raise ValueError(msg)
        else:
            sampleDict["split"] = sample["split"] 

        if sample.label_index == LabelName.NORMAL:
            if sample.split == Split.TRAIN:
                sample["label"] = DirType.NORMAL
            elif sample.split == Split.TEST:
                sample["label"] = DirType.NORMAL_TEST
            elif sample.split == Split.VAL:
                sample["label"] = None
        elif sample.label_index == LabelName.ABNORMAL:
            sample["label"] = DirType.ABNORMAL
        elif sample.label_index == LabelName.UNKNOWN:
            sample["label"] = "unknown" # TODO: does this work?
            # raise ValueError("sample.label_index can`t be UNKNOWN.")
        else:
            raise ValueError(f"sample.label_index {sample.label_index} not known.")
        sampleDict["label"] = sample["label"] 
        sampleDict["id"] = sample["id"]
        # newDict = {column: sample[column] for column in columns} # TODO Fix if mask are not available
        _samples.loc[len(_samples)] = sampleDict

    #######################
    ### Post-processing ###
    #######################

    # Add root to paths
    if "mask_path" in _samples:
        _samples["mask_path"] = _samples["mask_path"].fillna("")
        if root:
            _samples.loc[
                _samples["mask_path"] != "",
                "mask_path",
            ] = _samples.loc[_samples["mask_path"] != "", "mask_path"].map(lambda x: Path(root, x))
        _samples = _samples.astype({"image_path": "str", "mask_path": "str", "label": "str"})
    else:
        if root:
            _samples["image_path"] = _samples["image_path"].map(lambda x: Path(root, x))
        _samples = _samples.astype({"image_path": "str", "label": "str"})

    # Check if anomalous _samples are in training set
    if ((_samples.label_index == LabelName.ABNORMAL) & (_samples.split == Split.TRAIN)).any():
        msg = "Training set must not contain anomalous samples."
        raise MisMatchError(msg)

    # Check for None or NaN values
    if _samples.isna().any().any():
        msg = "The samples table contains None or NaN values."
        raise ValueError(msg)

    # Infer the task type
    if "mask_path" in _samples:
        _samples.attrs["task"] = "classification" if (_samples["mask_path"] == "").all() else "segmentation"
    else:
        _samples.attrs["task"] = "classification" 

    # Get the dataframe for the split.
    if split:
        _samples = _samples[_samples.split == split]
        _samples = _samples.reset_index(drop=True)

    return _samples

def _to_list(arg):
    if arg is None:
        return None

    if etau.is_container(arg):
        return list(arg)

    return [arg]

def exportDataset(dataset:FODataset, path:Path, overwrite:bool=True):
    exporter = AnomalyImageTreeExporter(path)
    for sample in dataset:
        _ = exporter.export_sample(sample)
    exporter.close()

def importDataset(path:Path, name:str, overwrite:bool=True, split: Tuple[str,...] = ("train", "test")) -> Tuple[fo.Dataset,dict[str,Any]|None]:
    """For importing training and/or test datasets. Not meant for unknown data which should be predicted on. Use
    importPredictDataset for that.

    Args:
        path (_type_): _description_
        name (_type_): _description_
        overwrite (bool, optional): _description_. Defaults to True.
        split (Union[str, Tuple[str, str], Tuple[str]], optional): _description_. Defaults to ("train", "test").

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if "train" in split and "test" in split:
        importer = TrainTestDataImporter(
            dataset_dir=path,
            compute_metadata=True,
            classes=None,
            shuffle=False,
            seed=None,
            max_samples=None,
        )

    elif "test" in split and "train" not in split:
        importer = TestDataImporter(
            dataset_dir=path,
            compute_metadata=True,
            classes=None,
            shuffle=False,
            seed=None,
            max_samples=None,
        )
    elif "train" in split and "test" not in split:
        return loadTrainingDataFolder(path, name, overwrite=overwrite)
    elif "pred" in split:
        return loadPredictDataset(path, name, overwrite=overwrite)
    else:
        raise ValueError("split must be 'train', 'test', ('train', 'test') or pred")

    dataset = fod.Dataset(name=name, overwrite=overwrite)
    info = None
    splits:set[Split] = set()

    with importer:
        for paths, image_metadata, category, split, anomalyType, label_index in importer:
            sample = fo.Sample(filepath=paths["image"], metadata=image_metadata, tags=[split])

            sample["anomalyType"] = anomalyType
            sample["category"] = category
            sample["split"] = split
            sample["label_index"] = label_index
            if paths.get("ground_truth", None) is not None:
                sample["ground_truth"] = fol.Segmentation(mask_path=paths["ground_truth"])
                sample["mask_path"] = paths["ground_truth"]
            # else:
            #     sample["mask_path"] = ""

            dataset.add_sample(sample)
            splits.add(split)
        
        if importer.has_dataset_info:
            info = importer.get_dataset_info()
            # parse_info(dataset, info)

    if Split.TRAIN.value in splits:
            dataset.tags.append(Split.TRAIN.value)
    if Split.TRAIN.TEST.value in splits:
            dataset.tags.append(Split.TEST.value)

    return dataset, info

def loadTrainingDataFolder(path:Path, name:str, overwrite:bool=False) -> Tuple[fo.Dataset, None]:
    split = "train"
    dataset = fo.Dataset.from_dir(
        dataset_dir=path,
        dataset_type=fot.ImageDirectory,
        name=name,
        overwrite=overwrite
    )

    for sample in dataset:
        sample["split"] = split
        sample["label_index"] = LabelName.NORMAL
        sample["anomalyType"] = "good"
        sample["category"] = fol.Classification(label=sample.filepath.split(os.sep)[-4])
        sample["mask_path"] = ""
        sample.tags.append("train")
        sample.save()
    dataset.tags.append(Split.TRAIN.value)    
    info = None
    return dataset, info

def loadPredictDataset(path:Path, name:str="pred", overwrite:bool=False):
    split = "pred"
    dataset = fo.Dataset.from_dir(
        dataset_dir=path,
        dataset_type=fot.ImageDirectory,
        name=name,
        overwrite=overwrite
    )
    for sample in dataset:
        sample["split"] = split
        sample["label_index"] = LabelName.UNKNOWN
        sample["anomalyType"] = fol.Classification(label="unknown")
        sample["category"] = fol.Classification(label=sample.filepath.split(os.sep)[-3])
        sample["mask_path"] = ""
        sample.tags.append("pred")
        sample.save()
    dataset.tags.append("pred")    
    info = None
    return dataset, info

def importPredictDataset(path:Path, name:str="prediction", overwrite:bool=False) -> Tuple[fo.Dataset, PredictDataset]:
    fo_dataset:fo.Dataset
    fo_dataset, info = loadPredictDataset(
        path=path,
        name=name,
        overwrite=overwrite
    )
    anomalib_Dataset:PredictDataset = PredictDataset(
        path=path,
    )
    return fo_dataset, anomalib_Dataset

if __name__ == "__main__":

    # Test import function
    dataset, info = importDataset(path="datasets/traintest/bottle", name="Dataset", overwrite=True, split=("train", "test"))
    DM = FODataModule("FODatamodule", dataset, train_batch_size=1, eval_batch_size=1)
    DM.setup()
    dataset1, info1 = importDataset(path="datasets/train/bottle", name="Dataset1", overwrite=True, split="train")
    DM1 = FODataModule("FODatamodule1", dataset1, train_batch_size=1, eval_batch_size=1)
    DM1.setup()
    dataset2, info2 = importDataset(path="datasets/test/bottle", name="Dataset2", overwrite=True, split="test")
    DM2 = FODataModule("FODatamodule2", dataset2, train_batch_size=1, eval_batch_size=1)
    DM2.setup()
    dataset3, anomalibDataset = importPredictDataset(path="datasets/prediction/bottle")
    # The anomalib PredictDataset does not support being part of a Datamodule and is usually used on it's own

    # Validate dataset with fiftyone launcher
    # session = fo.launch_app(dataset)
    # session.wait()

    # session = fo.launch_app(dataset1)
    # session.wait()

    # session = fo.launch_app(dataset2)
    # session.wait()

    session = fo.launch_app(dataset3)
    session.wait()