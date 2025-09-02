from datetime import datetime
import inspect
import itertools
import logging
import os
import random

from bson import json_util
from mongoengine.base import get_document
import pydash

import eta.core.datasets as etad
import eta.core.image as etai
import eta.core.serial as etas
import eta.core.utils as etau
import eta.core.video as etav

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
from fiftyone.core.sample import Sample
import fiftyone.core.storage as fos
import fiftyone.core.utils as fou
import fiftyone.migrations as fomi
import fiftyone.types as fot

# from .parsers import (
#     FiftyOneImageClassificationSampleParser,
#     FiftyOneTemporalDetectionSampleParser,
#     FiftyOneImageDetectionSampleParser,
#     FiftyOneImageLabelsSampleParser,
#     FiftyOneVideoLabelsSampleParser,
# )

from fiftyone.utils.data.importers import LabeledImageDatasetImporter
from fiftyone.utils.data.exporters import LabeledImageDatasetExporter


logger = logging.getLogger(__name__)

class AnomalyImageTreeImporter(LabeledImageDatasetImporter):
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
        unlabeled ("_unlabeled"): the name of the subdirectory containing
            unlabeled images
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
        unlabeled="_unlabeled",
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
        self.unlabeled = unlabeled
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
        paths, split, anomalyType = next(self._iter_samples)

        if self.compute_metadata:
            image_metadata = fom.ImageMetadata.build_for(paths["image"])
        else:
            image_metadata = None
        if anomalyType is not None:
            anomalyType = fol.Classification(label=anomalyType)

        return paths, image_metadata, split, anomalyType

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
            chunks = relpath.split(os.path.sep, 1)

            paths = {}

            if len(chunks) == 1:
                continue

            split, anomalyType, file = relpath.split(os.path.sep, 2)

            if anomalyType.startswith("."):
                continue
            if whitelist is not None and anomalyType not in whitelist:
                continue
            if split == "ground_truth":
                continue
            if split == "test":
                # Look for ground truth segmentation masks
                nr, ext = file.split(".")
                gtPath = os.path.join("ground_truth", anomalyType, nr + "_mask." + ext)
                gtPath = os.path.join(self.dataset_dir, gtPath)
                if os.path.exists(gtPath):
                    paths["ground_truth"] = gtPath                
            path = os.path.join(self.dataset_dir, relpath)
            paths["image"] = path
            if anomalyType == self.unlabeled:
                anomalyType = None
            else:
                classes.add(anomalyType)

            samples.append((paths, split, anomalyType))

        samples = self._preprocess_list(samples)

        if whitelist is not None:
            classes = self.classes
        else:
            classes = sorted(classes)

        self._classes = classes
        self._samples = samples
        self._num_samples = len(samples)
            
        # samples = []
        # classes = set()
        # whitelist = set(self.classes) if self.classes is not None else None

        # for relpath in etau.list_files(self.dataset_dir, recursive=True):
        #     chunks = relpath.split(os.path.sep, 1)
        #     if len(chunks) == 1:
        #         continue

        #     label = chunks[0]
        #     if label.startswith("."):
        #         continue

        #     if whitelist is not None and label not in whitelist:
        #         continue

        #     if label == self.unlabeled:
        #         label = None
        #     else:
        #         classes.add(label)

        #     path = os.path.join(self.dataset_dir, relpath)
        #     samples.append((path, label))

        # samples = self._preprocess_list(samples)

        # if whitelist is not None:
        #     classes = self.classes
        # else:
        #     classes = sorted(classes)

        # self._classes = classes
        # self._samples = samples
        # self._num_samples = len(samples)

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
    

import csv
import os

import fiftyone as fo
import fiftyone.utils.data as foud


class AnomalyImageTreeExporter(LabeledImageDatasetExporter):
    """Exporter for image classification datasets whose labels and image
    metadata are stored on disk in a CSV file.

    Datasets of this type are exported in the following format:

        <dataset_dir>/
            data/
                <filename1>.<ext>
                <filename2>.<ext>
                ...
            labels.csv

    where ``labels.csv`` is a CSV file in the following format::

        filepath,size_bytes,mime_type,width,height,num_channels,label
        <filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
        <filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
        ...

    Args:
        export_dir: the directory to write the export
    """

    def __init__(self, export_dir):
        super().__init__(export_dir=export_dir)
        self._data_dir = None
        self._labels = None
        self._labels_path = None
        self._image_exporter = None

    @property
    def requires_image_metadata(self):
        """Whether this exporter requires
        :class:`fiftyone.core.metadata.ImageMetadata` instances for each sample
        being exported.
        """
        return True

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) exported by this
        exporter.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            exporter directly exports labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the exporter can export a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the exporter can handle label dictionaries with
            value-types specified by this dictionary. Not all keys need be
            present in the exported label dicts
        -   ``None``. In this case, the exporter makes no guarantees about the
            labels that it can export
        """
        return fo.Classification

    def setup(self):
        """Performs any necessary setup before exporting the first sample in
        the dataset.

        This method is called when the exporter's context manager interface is
        entered, :func:`DatasetExporter.__enter__`.
        """
        self._data_dir = os.path.join(self.export_dir, "data")
        self._labels_path = os.path.join(self.export_dir, "labels.csv")
        self._labels = []

        # The `ImageExporter` utility class provides an `export()` method
        # that exports images to an output directory with automatic handling
        # of things like name conflicts
        self._image_exporter = foud.ImageExporter(
            True, export_path=self._data_dir, default_ext=".png",
        )
        self._image_exporter.setup()

    def export_sample(self, image_or_path, label, metadata=None):
        """Exports the given sample to the dataset.

        Args:
            image_or_path: an image or the path to the image on disk
            label: an instance of :meth:`label_cls`, or a dictionary mapping
                field names to :class:`fiftyone.core.labels.Label` instances,
                or ``None`` if the sample is unlabeled
            metadata (None): a :class:`fiftyone.core.metadata.ImageMetadata`
                instance for the sample. Only required when
                :meth:`requires_image_metadata` is ``True``
        """
        out_image_path, _ = self._image_exporter.export(image_or_path)

        if metadata is None:
            metadata = fo.ImageMetadata.build_for(image_or_path)

        self._labels.append((
            out_image_path,
            metadata.size_bytes,
            metadata.mime_type,
            metadata.width,
            metadata.height,
            metadata.num_channels,
            split,
            label.label,  # here, `label` is a `Classification` instance  
        ))

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
                "split",
                "label"                
            ])
            for row in self._labels:
                writer.writerow(row)

def _to_list(arg):
    if arg is None:
        return None

    if etau.is_container(arg):
        return list(arg)

    return [arg]

import fiftyone as fo

importer = AnomalyImageTreeImporter(
    dataset_dir="datasets/MVTecAD/bottle",
    compute_metadata=True,
    classes=None,
    unlabeled="_unlabeled",
    shuffle=False,
    seed=None,
    max_samples=None,
)

dataset = fod.Dataset(name="bottle", overwrite=True)

split_field = "split"
anomalyType_field = "anomalyType"

with importer:
    for paths, image_metadata, split, anomalyType in importer:
        sample = fo.Sample(filepath=paths["image"], metadata=image_metadata, tags=[split])

        sample[anomalyType_field] = anomalyType
        if paths.get("ground_truth", None) is not None:
            sample["ground_truth"] = fol.Segmentation(mask_path=paths["ground_truth"])

        dataset.add_sample(sample)

    if importer.has_dataset_info:
        info = importer.get_dataset_info()
        # parse_info(dataset, info)

samples = dataset.limit(10)

exporter = AnomalyImageTreeExporter(export_dir="datasets/MVTecAD_export/bottle")
samples.export(dataset_exporter=exporter)
#dataset.persistent = True

session = fo.launch_app(dataset)

session.wait()