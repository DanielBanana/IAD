# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble training pipeline."""

from typing import TYPE_CHECKING

from anomalib.data.utils import ValSplitMode

import logging

import torch

from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner

from anomalib.pipelines.tiled_ensemble.components import (
    MergeJobGenerator,
    PredictJobGenerator,
    SmoothingJobGenerator,
    StatisticsJobGenerator
)
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage, PredictData
from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_engine import TiledEnsembleEngine

from collections.abc import Generator
from itertools import product
from pathlib import Path
from anomalib.data import AnomalibDataModule
from anomalib.models import AnomalibModule
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT

# from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import (
#     get_ensemble_datamodule,
#     get_ensemble_engine,
#     get_ensemble_model,
#     get_ensemble_tiler,
# )

from typing import Any
from lightning import seed_everything
from torch.utils.data import DataLoader
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions

logger = logging.getLogger(__name__)

from anomalib.data.utils import TestSplitMode
from anomalib.pipelines.tiled_ensemble.components import (
    MetricsCalculationJobGenerator,
    NormalizationJobGenerator,
    ThresholdingJobGenerator,
    VisualizationJobGenerator,
)
from anomalib.pipelines.tiled_ensemble.components.utils import ThresholdingStage


import json
from jsonargparse import ArgumentParser, Namespace
from lightning import Trainer
from torchvision.transforms.v2 import Compose, Resize, Transform
from anomalib.data import AnomalibDataModule, ImageBatch, get_datamodule
from anomalib.models import AnomalibModule, get_model
from anomalib.pre_processing.utils.transform import get_exportable_transform
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics.evaluator import Evaluator
from anomalib.metrics import F1Score, AUPR, AUROC, F1AdaptiveThreshold
from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler, TileCollater

"""Tiled ensemble - pipelines"""

class TrainTiledEnsemble(Pipeline):
    """Tiled ensemble training pipeline."""

    def __init__(self) -> None:
        self.root_dir: Path

    def _setup_runners(self, args: dict) -> list[Runner]:
        """Setup the runners for the pipeline.

        This pipeline consists of training and validation steps:
        Training models > prediction on val data > merging val data >
        > (optionally) smoothing seams > calculation of post-processing statistics

        Returns:
            list[Runner]: List of runners executing tiled ensemble train + val jobs.
        """
        runners: list[Runner] = []
        self.root_dir = TiledEnsembleEngine.setup_ensemble_workspace(args)

        seed = args["seed"]
        accelerator = args["accelerator"]
        tiling_args = args["tiling"]
        data_args = args["data"] # TODO: Example pipeline takes config and loads from disk, we want to supply the data as a loaded datamodule
        normalization_stage = NormalizationStage(args["normalization_stage"])
        model_args = args["TrainModels"]["model"] # TODO: Example pipeline is really simple in the amount and type of model arguments

        train_job_generator = TrainModelJobGenerator(
            seed=seed,
            accelerator=accelerator,
            root_dir=self.root_dir,
            tiling_args=tiling_args,
            data_args=data_args,
            normalization_stage=normalization_stage,
        )

        predict_job_generator = PredictJobGenerator(
            data_source=PredictData.VAL,
            seed=seed,
            accelerator=accelerator,
            root_dir=self.root_dir,
            tiling_args=tiling_args,
            data_args=data_args,
            model_args=model_args,
            normalization_stage=normalization_stage,
        )

        # 1. train
        if accelerator == "cuda":
            runners.append(
                ParallelRunner(
                    train_job_generator,
                    n_jobs=torch.cuda.device_count(),
                ),
            )
        else:
            runners.append(
                SerialRunner(
                    train_job_generator,
                ),
            )

        if data_args["init_args"]["val_split_mode"] == ValSplitMode.NONE:
            logger.warning("No validation set provided, skipping statistics calculation.")
            return runners

        # 2. predict using validation data
        if accelerator == "cuda":
            runners.append(
                ParallelRunner(predict_job_generator, n_jobs=torch.cuda.device_count()),
            )
        else:
            runners.append(
                SerialRunner(predict_job_generator),
            )

        # 3. merge predictions
        runners.append(SerialRunner(MergeJobGenerator(tiling_args=tiling_args, data_args=data_args)))

        # 4. (optional) smooth seams
        if args["SeamSmoothing"]["apply"]:
            runners.append(
                SerialRunner(
                    SmoothingJobGenerator(accelerator=accelerator, tiling_args=tiling_args, data_args=data_args),
                ),
            )

        # 5. calculate statistics used for inference
        runners.append(SerialRunner(StatisticsJobGenerator(self.root_dir)))

        return runners


class EvalTiledEnsemble(Pipeline):
    """Tiled ensemble evaluation pipeline.

    Args:
        root_dir (Path): Path to root dir of run that contains checkpoints.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)

    def _setup_runners(self, args: dict) -> list[Runner]:
        """Set up the runners for the pipeline.

        This pipeline consists of jobs used to test/evaluate tiled ensemble:
        Prediction on test data > merging of predictions > (optional) seam smoothing
        > (optional) Normalization > (optional) Thresholding
        > Visualisation of predictions > Metrics calculation.

        Returns:
            list[Runner]: List of runners executing tiled ensemble testing jobs.
        """
        runners: list[Runner] = []

        if args["data"]["init_args"]["test_split_mode"] == TestSplitMode.NONE:
            logger.info("Test split mode set to `none`, skipping test phase.")
            return runners

        seed = args["seed"]
        accelerator = args["accelerator"]
        tiling_args = args["tiling"]
        data_args = args["data"]
        normalization_stage = NormalizationStage(args["normalization_stage"])
        thresholding_stage = ThresholdingStage(args["thresholding_stage"])
        model_args = args["TrainModels"]["model"]

        predict_job_generator = PredictJobGenerator(
            PredictData.TEST,
            seed=seed,
            accelerator=accelerator,
            root_dir=self.root_dir,
            tiling_args=tiling_args,
            data_args=data_args,
            model_args=model_args,
            normalization_stage=normalization_stage,
        )
        # 1. predict using test data
        if accelerator == "cuda":
            runners.append(
                ParallelRunner(
                    predict_job_generator,
                    n_jobs=torch.cuda.device_count(),
                ),
            )
        else:
            runners.append(
                SerialRunner(
                    predict_job_generator,
                ),
            )
        # 2. merge predictions
        runners.append(SerialRunner(MergeJobGenerator(tiling_args=tiling_args, data_args=data_args)))

        # 3. (optional) smooth seams
        if args["SeamSmoothing"]["apply"]:
            runners.append(
                SerialRunner(
                    SmoothingJobGenerator(accelerator=accelerator, tiling_args=tiling_args, data_args=data_args),
                ),
            )

        # 4. (optional) normalize
        if normalization_stage == NormalizationStage.IMAGE:
            runners.append(SerialRunner(NormalizationJobGenerator(self.root_dir)))
        # 5. (optional) threshold to get labels from scores
        if thresholding_stage == ThresholdingStage.IMAGE:
            runners.append(SerialRunner(ThresholdingJobGenerator(self.root_dir, normalization_stage)))

        # 6. visualize predictions
        runners.append(
            SerialRunner(VisualizationJobGenerator(self.root_dir, data_args=data_args)),
        )
        # calculate metrics
        runners.append(
            SerialRunner(
                MetricsCalculationJobGenerator(
                    accelerator=accelerator,
                    root_dir=self.root_dir,
                    model_args=model_args,
                ),
            ),
        )

        return runners

"""Tiled ensemble - ensemble training job."""

class TrainModelJob(Job):
    """Job for training of individual models in the tiled ensemble.

    Args:
        accelerator (str): Accelerator (device) to use.
        seed (int): Random seed for reproducibility.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        tile_index (tuple[int, int]): Index of tile that this model processes.
        normalization_stage (str): Normalization stage flag.
        metrics (dict): metrics dict with pixel and image metric names.
        trainer_args (dict| None): Additional arguments to pass to the trainer class.
        model (AnomalyModule): Model to train.
        datamodule (AnomalibDataModule): Datamodule with all dataloaders.

    """

    name = "TrainModels"

    def __init__(
        self,
        accelerator: str,
        seed: int,
        root_dir: Path,
        tile_index: tuple[int, int],
        normalization_stage: str,
        trainer_args: dict | None,
        model: AnomalibModule,
        datamodule: AnomalibDataModule,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.seed = seed
        self.root_dir = root_dir
        self.tile_index = tile_index
        self.normalization_stage = normalization_stage
        self.trainer_args = trainer_args
        self.model = model
        self.datamodule = datamodule

    def run(
        self,
        task_id: int | None = None,
    ) -> TiledEnsembleEngine:
        """Run train job that fits the model for given tile location.

        Args:
            task_id: Passed when job is ran in parallel.

        Returns:
            TiledEnsembleEngine: Engine containing trained model.
        """
        devices: str | list[int] = "auto"
        if task_id is not None:
            devices = [task_id]
            logger.info(f"Running job {self.model.__class__.__name__} with device {task_id}")

        logger.info("Start of training for tile at position %s,", self.tile_index)
        seed_everything(self.seed)

        # create engine for specific tile location and fit the model
        engine = get_ensemble_engine(
            tile_index=self.tile_index,
            accelerator=self.accelerator,
            devices=devices,
            root_dir=self.root_dir,
            trainer_args=self.trainer_args,
        )
        engine.fit(model=self.model, datamodule=self.datamodule)
        # move model to cpu to avoid memory issues as the engine is returned to be used in validation phase
        engine.model.cpu()

        return engine

    @staticmethod
    def collect(results: list[TiledEnsembleEngine]) -> dict[tuple[int, int], TiledEnsembleEngine]:
        """Collect engines from each tile location into a dict.

        Returns:
            dict[tuple[int, int], TiledEnsembleEngine]: Dict has form {tile_index: TiledEnsembleEngine}
        """
        return {r.tile_index: r for r in results}

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Skip as checkpoints are already saved by callback."""


class TrainModelJobGenerator(JobGenerator):
    """Generator for training job that train model for each tile location.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    def __init__(
        self,
        seed: int,
        accelerator: str,
        root_dir: Path,
        tiling_args: dict,
        data_args: dict,
        normalization_stage: NormalizationStage,
    ) -> None:
        self.seed = seed
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.tiling_args = tiling_args
        self.data_args = data_args
        self.normalization_stage = normalization_stage

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return TrainModelJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[TrainModelJob, None, None]:
        """Generate training jobs for each tile location.

        Args:
            args (dict): Dict with config passed to training.
            prev_stage_result (None): Not used here.

        Returns:
            Generator[TrainModelJob, None, None]: TrainModelJob generator
        """
        del prev_stage_result  # Not needed for this job
        if args is None:
            msg = "TrainModels job requires config args"
            raise ValueError(msg)

        # tiler used for splitting the image and getting the tile count
        tiler = get_ensemble_tiler(self.tiling_args)

        logger.info(
            f"Tiled ensemble training started. Separate models will be trained for {tiler.num_tiles} tile locations.",
        )
        # go over all tile positions
        for tile_index in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
            # prepare datamodule with custom collate function that only provides specific tile of image
            datamodule = get_ensemble_datamodule(
                data_config=self.data_args,
                image_size=self.tiling_args["image_size"],
                tiler=tiler,
                tile_index=tile_index,
            )
            model = get_ensemble_model(
                model_args=args["model"],
                normalization_stage=self.normalization_stage,
                input_size=self.tiling_args["tile_size"],
            )

            # pass root_dir to engine so all models in ensemble have the same root dir
            yield TrainModelJob(
                accelerator=self.accelerator,
                seed=self.seed,
                root_dir=self.root_dir,
                tile_index=tile_index,
                normalization_stage=self.normalization_stage,
                trainer_args=args.get("trainer", {}),
                model=model,
                datamodule=datamodule,
            )

"""Tiled ensemble - ensemble prediction job."""

class PredictJob(Job):
    """Job for generating predictions with individual models in the tiled ensemble.

    Args:
        accelerator (str): Accelerator (device) to use.
        seed (int): Random seed for reproducibility.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        tile_index (tuple[int, int]): Index of tile that this model processes.
        normalization_stage (str): Normalization stage flag.
        dataloader (DataLoader): Dataloader to use for training (either val or test).
        model (AnomalyModule): Model to train.
        engine (TiledEnsembleEngine | None):
            engine from train job. If job is used standalone, instantiate engine and model from checkpoint.
        ckpt_path (Path | None): Path to checkpoint to be loaded if engine doesn't contain correct weights.

    """

    name = "Predict"

    def __init__(
        self,
        accelerator: str,
        seed: int,
        root_dir: Path,
        tile_index: tuple[int, int],
        normalization_stage: str,
        dataloader: DataLoader,
        model: AnomalibModule | None,
        engine: TiledEnsembleEngine | None,
        ckpt_path: Path | None,
    ) -> None:
        super().__init__()
        if engine is None and ckpt_path is None:
            msg = "Either engine or checkpoint must be provided to predict job."
            raise ValueError(msg)

        self.accelerator = accelerator
        self.seed = seed
        self.root_dir = root_dir
        self.tile_index = tile_index
        self.normalization_stage = normalization_stage
        self.dataloader = dataloader
        self.model = model
        self.engine = engine
        self.ckpt_path = ckpt_path

    def run(
        self,
        task_id: int | None = None,
    ) -> tuple[tuple[int, int], Any | None]:
        """Predict job that predicts the data with specific model for given tile location.

        Args:
            task_id: Passed when job is ran in parallel.

        Returns:
            tuple[tuple[int, int], list[Any]]: Tile index, List of predictions.
        """
        devices: str | list[int] = "auto"
        if task_id is not None:
            devices = [task_id]
            logger.info(f"Running job {self.model.__class__.__name__} with device {task_id}")

        logger.info("Start of predicting for tile at position %s,", self.tile_index)
        seed_everything(self.seed)

        if self.engine is None:
            # in case predict is invoked separately from train job, make new engine instance
            self.engine = get_ensemble_engine(
                tile_index=self.tile_index,
                accelerator=self.accelerator,
                devices=devices,
                root_dir=self.root_dir,
            )

        predictions = self.engine.predict(model=self.model, dataloaders=self.dataloader, ckpt_path=self.ckpt_path)

        # also return tile index as it's needed in collect method
        return self.tile_index, predictions

    @staticmethod
    def collect(results: list[tuple[tuple[int, int], list[Any]]]) -> EnsemblePredictions:
        """Collect predictions from each tile location into the predictions class.

        Returns:
            EnsemblePredictions: Object containing all predictions in form ready for merging.
        """
        storage = EnsemblePredictions()

        for tile_index, predictions in results:
            storage.add_tile_prediction(tile_index, predictions)

        return storage

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """This stage doesn't save anything."""


class PredictJobGenerator(JobGenerator):
    """Generator for predict job that uses individual models to predict for each tile location.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
        data_source (PredictData): Whether to predict on validation set. If false use test set.
    """

    def __init__(
        self,
        data_source: PredictData,
        seed: int,
        accelerator: str,
        root_dir: Path,
        tiling_args: dict,
        data_args: dict,
        model_args: dict,
        normalization_stage: NormalizationStage,
    ) -> None:
        self.data_source = data_source
        self.seed = seed
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.tiling_args = tiling_args
        self.data_args = data_args
        self.model_args = model_args
        self.normalization_stage = normalization_stage

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return PredictJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[PredictJob, None, None]:
        """Generate predict jobs for each tile location.

        Args:
            args (dict): Dict with config passed to training.
            prev_stage_result (dict[tuple[int, int], TiledEnsembleEngine] | None):
                if called after train job this contains engines with individual models, otherwise load from checkpoints.

        Returns:
            Generator[PredictJob, None, None]: PredictJob generator.
        """
        del args  # args not used here

        # tiler used for splitting the image and getting the tile count
        tiler = get_ensemble_tiler(self.tiling_args)

        logger.info(
            "Tiled ensemble predicting started using %s data.",
            self.data_source.value,
        )
        # go over all tile positions
        for tile_index in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
            # prepare datamodule with custom collate function that only provides specific tile of image
            datamodule = get_ensemble_datamodule(
                data_config=self.data_args,
                image_size=self.tiling_args["image_size"],
                tiler=tiler,
                tile_index=tile_index,
            )

            # check if predict step is positioned after training
            if prev_stage_result and tile_index in prev_stage_result:
                engine = prev_stage_result[tile_index]
                # model is inside engine in this case
                model = engine.model
                ckpt_path = None
            else:
                # any other case - predict is called standalone
                engine = None
                # we need to make new model instance as it's not inside engine
                model = get_ensemble_model(
                    model_args=self.model_args,
                    normalization_stage=self.normalization_stage,
                    input_size=self.tiling_args["tile_size"],
                )
                tile_i, tile_j = tile_index
                # prepare checkpoint path for model on current tile location
                ckpt_path = self.root_dir / "weights" / "lightning" / f"model{tile_i}_{tile_j}.ckpt"

            # pick the dataloader based on predict data
            dataloader = datamodule.test_dataloader()
            if self.data_source == PredictData.VAL:
                dataloader = datamodule.val_dataloader()

            # pass root_dir to engine so all models in ensemble have the same root dir
            yield PredictJob(
                accelerator=self.accelerator,
                seed=self.seed,
                root_dir=self.root_dir,
                tile_index=tile_index,
                normalization_stage=self.normalization_stage,
                model=model,
                dataloader=dataloader,
                engine=engine,
                ckpt_path=ckpt_path,
            )

"""Helper functions for the tiled ensemble training."""

def get_ensemble_datamodule(
    data_config: dict,
    image_size: int | tuple[int, int],
    tiler: EnsembleTiler,
    tile_index: tuple[int, int],
) -> AnomalibDataModule:
    """Get Anomaly Datamodule adjusted for use in ensemble.

    Datamodule collate function gets replaced by TileCollater in order to tile all images before they are passed on.

    Args:
        data_config: tiled ensemble data configuration.
        image_size (int | tuple[int, int]): full effective image size of tiled ensemble.
        tiler (EnsembleTiler): Tiler used to split the images to tiles for use in ensemble.
        tile_index (tuple[int, int]): Index of the tile in the split image.

    Returns:
        AnomalibDataModule: Anomalib Lightning DataModule
    """
    datamodule = get_datamodule(data_config)
    datamodule.setup()

    # add tiled ensemble image_size transform to datamodule
    setup_transforms(datamodule, image_size=image_size)
    datamodule.external_collate_fn = TileCollater(tiler, tile_index, default_collate_fn=ImageBatch.collate)
    # manually set setup, so later setup doesn't override the transforms...
    datamodule._is_setup = True  # noqa: SLF001

    return datamodule


def setup_transforms(datamodule: AnomalibDataModule, image_size: int | tuple[int, int]) -> None:
    """Modify datamodule resize transforms so the effective ensemble image_size is correct.

    Args:
        datamodule: datamodule where resize transform will be setup.
        image_size (int | tuple[int, int]): tiled ensemble input image size

    """
    resize_transform = Resize(image_size)

    for subset_name in ["train", "val", "test"]:
        default_aug = getattr(datamodule, f"{subset_name}_augmentations", None)

        if isinstance(default_aug, Resize):
            msg = f"Conflicting resize shapes found between dataset augmentations and tiled ensemble size. \
                You are using a Resize transform in your input data augmentations. Please be aware that the \
                tiled ensemble image size is determined by tiling config. The final effective input size as \
                seen by individual model will be determined by the tile_size. To change \
                the effective ensemble input size, please change the image_size in the tiling config. \
                Augmentations: {default_aug.size}, Tiled ensemble base size: {image_size}"
            logger.warning(msg)
            augmentations = resize_transform
        elif isinstance(default_aug, Compose):
            augmentations = Compose([*default_aug.transforms, resize_transform])
        elif isinstance(default_aug, Transform):
            augmentations = Compose([default_aug, resize_transform])
        else:
            augmentations = resize_transform
        # add augmentations with resize to datamodule and datasets, ensuring that output images match effective size
        setattr(datamodule, f"{subset_name}_augmentations", augmentations)
        data_subset = getattr(datamodule, f"{subset_name}_data", None)
        if data_subset is not None:
            data_subset.augmentations = augmentations


def get_ensemble_model(
    model_args: dict,
    input_size: int | tuple[int, int],
    normalization_stage: NormalizationStage,
) -> AnomalibModule:
    """Get model prepared for ensemble training.

    Args:
        model_args (dict): tiled ensemble model configuration.
        input_size (int | tuple[int, int]): individual model input size.
        normalization_stage (NormalizationStage): stage when normalization performed.

    Returns:
        AnomalyModule: model with input_size setup
    """
    # first make temporary model to get object
    temp_model = get_model(model_args)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    # create custom pre_proc with correct input size
    # since we can't modify input_size directly (needed during instantiation by some models like FastFlow)
    pre_processor = temp_model.configure_pre_processor(input_size)
    # make actual model with correct input size
    # image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    # pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    # evaluator = Evaluator(val_metrics=[image_auroc, pixel_auroc])
    model: AnomalibModule = get_model(model_args, pre_processor=pre_processor, visualizer=False, evaluator=True)
    if model.pre_processor is not None:
        model_pre_processor: PreProcessor = model.pre_processor

        # drop Resize in all cases since it gets copied to datamodule, and we don't want that!
        pre_transforms = model_pre_processor.transform
        if isinstance(pre_transforms, Resize):
            update_transform = []
        elif isinstance(pre_transforms, Compose):
            update_transform = Compose([
                transform for transform in pre_transforms.transforms if not isinstance(transform, Resize)
            ])
        elif pre_transforms is not None:
            update_transform = pre_transforms
        else:
            update_transform = []

        model_pre_processor.transform = update_transform
        model_pre_processor.export_transform = get_exportable_transform(update_transform)

    if model.post_processor is not None:
        model_post_processor: PostProcessor = model.post_processor
        # set model normalisation only if the stage is set to tile level (but thresholding is always applied)
        model_post_processor.enable_normalization = normalization_stage == NormalizationStage.TILE

    return model


def get_ensemble_tiler(tiling_args: dict) -> EnsembleTiler:
    """Get tiler used for image tiling and to obtain tile dimensions.

    Args:
        tiling_args: tiled ensemble tiling configuration.

    Returns:
        EnsembleTiler: tiler object.
    """
    tiler = EnsembleTiler(
        tile_size=tiling_args["tile_size"],
        stride=tiling_args["stride"],
        image_size=tiling_args["image_size"],
    )

    return tiler  # noqa: RET504


def parse_trainer_kwargs(trainer_args: dict | None) -> Namespace | dict:
    """Parse trainer args and instantiate all needed elements.

    Transforms config into kwargs ready for Trainer, including instantiation of callback etc.

    Args:
        trainer_args (dict): Trainer args dictionary.

    Returns:
        dict: parsed kwargs with instantiated elements.
    """
    if not trainer_args:
        return {}

    # try to get trainer args, if not present return empty
    parser = ArgumentParser()

    parser.add_class_arguments(Trainer, fail_untyped=False, instantiate=False, sub_configs=True)
    config = parser.parse_object(trainer_args)
    objects = parser.instantiate_classes(config)

    return objects  # noqa: RET504


def get_ensemble_engine(
    tile_index: tuple[int, int],
    accelerator: str,
    devices: list[int] | str | int,
    root_dir: Path,
    trainer_args: dict | None = None,
) -> TiledEnsembleEngine:
    """Prepare engine for ensemble training or prediction.

    This method makes sure correct normalization is used, prepares metrics and additional trainer kwargs..

    Args:
        tile_index (tuple[int, int]): Index of tile that this model processes.
        accelerator (str): Accelerator (device) to use.
        devices (list[int] | str | int): device IDs used for training.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        trainer_args (dict): Trainer args dictionary. Empty dict if not present.

    Returns:
        TiledEnsembleEngine: set up engine for ensemble training/prediction.
    """
    # parse additional trainer args and callbacks if present in config
    trainer_kwargs = parse_trainer_kwargs(trainer_args)
    # remove keys that we already have
    trainer_kwargs.pop("accelerator", None)
    trainer_kwargs.pop("default_root_dir", None)
    trainer_kwargs.pop("devices", None)

    # create engine for specific tile location
    engine = TiledEnsembleEngine(
        tile_index=tile_index,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=root_dir,
        **trainer_kwargs,
    )

    return engine  # noqa: RET504


def get_threshold_values(normalization_stage: NormalizationStage, root_dir: Path) -> tuple[float, float]:
    """Get threshold values for image and pixel level predictions.

    If normalization is not used, get values based on statistics obtained from validation set.
    If normalization is used, both image and pixel threshold are 0.5

    Args:
        normalization_stage (NormalizationStage): ensemble run args, used to get normalization stage.
        root_dir (Path): path to run root where stats file is saved.

    Returns:
        tuple[float, float]: image and pixel threshold.
    """
    if normalization_stage == NormalizationStage.NONE:
        stats_path = root_dir / "weights" / "lightning" / "stats.json"
        with stats_path.open("r") as f:
            stats = json.load(f)
        image_threshold = stats["image_threshold"]
        pixel_threshold = stats["pixel_threshold"]
    else:
        # normalization transforms the scores so that threshold is at 0.5
        image_threshold = 0.5
        pixel_threshold = 0.5

    return image_threshold, pixel_threshold