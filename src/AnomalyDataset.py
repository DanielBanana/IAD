from datetime import datetime
import inspect
import itertools
import logging
import os
import random

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from bson import json_util
from mongoengine.base import get_document
import pydash

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
from fiftyone.utils.data.exporters import LabeledImageDatasetExporter, GenericSampleDatasetExporter


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
        paths, category, split, anomalyType = next(self._iter_samples)

        if self.compute_metadata:
            image_metadata = fom.ImageMetadata.build_for(paths["image"])
        else:
            image_metadata = None
        if anomalyType is not None:
            anomalyType = fol.Classification(label=anomalyType)
        if category is not None:
            category = fol.Classification(label=category)

        return paths, image_metadata, category, split, anomalyType

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
            if anomalyType == self.unlabeled:
                anomalyType = None
            else:
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

    def export_sample(self, sample):
        """Exports the given sample to the dataset.

        Args:
            sample: a :class:`fiftyone.core.sample.Sample`
        """
        file = sample.filepath.split(os.path.sep)[-1]
        
        category =sample["category"].label
        anomalyType = sample["anomalyType"].label

        outpath = os.path.join(self._data_dir, category, anomalyType, file)
        out_image_path, _ = self._image_exporter.export(sample.filepath, outpath=outpath)

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
            anomalyType,
            sample.tags[0]
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
                "category",
                "anomalyType",
                "split"                
            ])
            for row in self._labels:
                writer.writerow(row)

def _to_list(arg):
    if arg is None:
        return None

    if etau.is_container(arg):
        return list(arg)

    return [arg]

def importAnomalyDataset(dataset, importer):
    with importer:
        for paths, image_metadata, category, split, anomalyType in importer:
            sample = fo.Sample(filepath=paths["image"], metadata=image_metadata, tags=[split])

            sample["anomalyType"] = anomalyType
            sample["category"] = category
            if paths.get("ground_truth", None) is not None:
                sample["ground_truth"] = fol.Segmentation(mask_path=paths["ground_truth"])

            dataset.add_sample(sample)

        if importer.has_dataset_info:
            info = importer.get_dataset_info()
            # parse_info(dataset, info)
    return dataset, info

if __name__ == "__main__":

    importer = AnomalyImageTreeImporter(
        dataset_dir="datasets/MVTecAD",
        compute_metadata=True,
        classes=None,
        unlabeled="_unlabeled",
        shuffle=False,
        seed=None,
        max_samples=None,
    )

    dataset = fod.Dataset(name="MVTec", overwrite=True)
    #dataset = fod.load_dataset("MVTec")

    dataset, info = importAnomalyDataset(dataset, importer)

    samples = dataset.limit(10)

    exporter = AnomalyImageTreeExporter(export_dir="datasets/MVTecAD_export")
    samples.export(dataset_exporter=exporter)

    session = fo.launch_app(dataset)

    session.wait()