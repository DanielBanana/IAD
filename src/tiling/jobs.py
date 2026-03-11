# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Changed by Daniel Pommer, TH Nuremberg, 2026

"""Tiled ensemble - post-processing statistics calculation job."""

import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS
from anomalib.post_processing import PostProcessor
from anomalib.visualization import ImageVisualizer

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Any, Tuple, Dict, List
from anomalib.pipelines.tiled_ensemble.components.stats_calculation import StatisticsJob, StatisticsJobGenerator
from anomalib.pipelines.tiled_ensemble.components.thresholding import ThresholdingJob
from AnomalyDataset import FODataModule, FODataset



class AOIStatisticsJob(StatisticsJob):
    def __init__(self, predictions: List[Any] | None, root_dir: Path) -> None:
        super().__init__(predictions, root_dir)

    def run(self, task_id: int | None = None) -> Tuple[List[Any], Dict[str,Any], ]:
        """Run job that calculates statistics needed in post-processing steps.

        Args:
            task_id: Not used in this case

        Returns:
            dict: Statistics dict with min, max and threshold values.
        """
        del task_id  # not needed here

        post_processor = PostProcessor()

        logger.info("Starting post-processing statistics calculation.")

        for data in tqdm(self.predictions, desc="Stats calculation"):
            # update minmax and thresholds
            post_processor.on_validation_batch_end(None, None, outputs=data)

        post_processor.on_validation_epoch_end(None, None)

        # return stats with save path that is later used to save statistics.
        return self.predictions, {
            "minmax": {
                "pred_score": {
                    "min": post_processor.image_min.item(),
                    "max": post_processor.image_max.item(),
                },
                "anomaly_map": {
                    "min": post_processor.pixel_min.item(),
                    "max": post_processor.pixel_max.item(),
                },
            },
            "image_threshold": post_processor.image_threshold.item(),
            "pixel_threshold": post_processor.pixel_threshold.item(),
            "save_path": (self.root_dir / "stats.json"),
        }
    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            dict: statistics dictionary.
        """
        return results

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Save statistics to file system."""

        if isinstance(results[0], tuple):
            res = results[0][1] # the run job is returning something else than the statistics, assume the statistics are the first thing
        else:
            res = results[0] # take the first element as result is list of lists here

        # get and remove path from stats dict
        stats_path: Path = res.pop("save_path")
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        # save statistics next to weights
        with stats_path.open("w", encoding="utf-8") as stats_file:
            json.dump(res, stats_file, ensure_ascii=False, indent=4)
    
class AOIStatisticsJobGenerator(JobGenerator):
    """Generate StatisticsJob.

    Args:
        root_dir (Path): Root directory where statistics file will be saved (in weights folder).
    """

    def __init__(
        self,
        root_dir: Path,
    ) -> None:
        self.root_dir = root_dir

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIStatisticsJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[AOIStatisticsJob, None, None]:
        """Return a generator producing a single stats calculating job.

        Args:
            args: Not used here.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[StatisticsJob, None, None]: StatisticsJob generator.
        """
        del args  # not needed here

        yield AOIStatisticsJob(
            predictions=prev_stage_result,
            root_dir=self.root_dir,
        )



# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - prediction merging job."""

import logging
from collections.abc import Generator
from typing import Any

from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS

from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler
from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import get_ensemble_tiler
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
# from anomalib.pipelines.tiled_ensemble.components.utils.prediction_merging import PredictionMergingMechanism

logger = logging.getLogger(__name__)

class AOIMergeJob(Job):
    """Job for merging tile-level predictions into image-level predictions.

    Args:
        predictions (EnsemblePredictions): Object containing ensemble predictions.
        tiler (EnsembleTiler): Ensemble tiler used for untiling.
    """

    name = "Merge"

    def __init__(self, predictions: EnsemblePredictions, tiler: EnsembleTiler) -> None:
        super().__init__()
        self.predictions = predictions
        self.tiler = tiler

    def run(self, task_id: int | None = None) -> list[Any]:
        """Run merging job that merges all batches of tile-level predictions into image-level predictions.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: List of merged predictions.
        """
        del task_id  # not needed here

        merger = PredictionMergingMechanism(self.predictions, self.tiler)

        logger.info("Merging predictions.")

        # merge all batches
        merged_predictions = [
            merger.merge_tile_predictions(batch_idx)
            for batch_idx in tqdm(range(merger.num_batches), desc="Prediction merging")
        ]

        return merged_predictions  # noqa: RET504

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: List of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing to save in this job."""


class AOIMergeJobGenerator(JobGenerator):
    """Generate MergeJob."""

    def __init__(self, tiling_args: dict, data_args: dict) -> None:
        super().__init__()
        self.tiling_args = tiling_args
        self.data_args = data_args

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIMergeJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: EnsemblePredictions | None = None,
    ) -> Generator[AOIMergeJob, None, None]:
        """Return a generator producing a single merging job.

        Args:
            args (dict): Tiled ensemble pipeline args.
            prev_stage_result (EnsemblePredictions): Ensemble predictions from predict step.

        Returns:
            Generator[AOIMergeJob, None, None]: AOIMergeJob generator
        """
        del args  # args not used here

        tiler = get_ensemble_tiler(self.tiling_args)
        if prev_stage_result is not None:
            yield AOIMergeJob(prev_stage_result, tiler)
        else:
            msg = "Merging job requires tile level predictions from previous step."
            raise ValueError(msg)


# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class used as mechanism to merge ensemble predictions from each tile into complete whole-image representation."""

import torch
from torch import Tensor

from anomalib.data import ImageBatch

from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions


class PredictionMergingMechanism:
    """Class used for merging the data predicted by each separate model of tiled ensemble.

    Tiles are stacked in one tensor and untiled using Ensemble Tiler.
    Boxes from tiles are either stacked or generated anew from anomaly map.
    Labels are combined with OR operator, meaning one anomalous tile -> anomalous image.
    Scores are averaged across all tiles.

    Args:
        ensemble_predictions (EnsemblePredictions): Object containing predictions on tile level.
        tiler (EnsembleTiler): Tiler used to transform tiles back to image level representation.

    Example:
        >>> from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler
        >>> from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
        >>>
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> data = EnsemblePredictions()
        >>> merger = PredictionMergingMechanism(data, tiler)
        >>>
        >>> # we can then start merging procedure for each batch
        >>> merger.merge_tile_predictions(0)
    """

    def __init__(self, ensemble_predictions: EnsemblePredictions, tiler: EnsembleTiler) -> None:
        assert ensemble_predictions.num_batches > 0, "There should be at least one batch for each tile prediction."
        assert (0, 0) in ensemble_predictions.get_batch_tiles(
            0,
        ), "Tile prediction dictionary should always have at least one tile"

        self.ensemble_predictions = ensemble_predictions
        self.num_batches = self.ensemble_predictions.num_batches

        self.tiler = tiler

    def merge_tiles(self, batch_data: dict, tile_key: str) -> Tensor:
        """Merge tiles back into one tensor and perform untiling with tiler.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.
            tile_key (str): Key used in prediction dictionary for tiles that we want to merge.

        Returns:
            Tensor: Tensor of tiles in original (stitched) shape.
        """
        # batch of tiles with index (0, 0) always exists, so we use it to get some basic information
        first_tiles = getattr(batch_data[0, 0], tile_key)
        batch_size = first_tiles.shape[0]
        device = first_tiles.device

        single_channel = False
        if len(first_tiles.shape) == 3:
            single_channel = True
            # in some cases, we don't have channels but just B, H, W
            merged_size = [
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            ]
        else:
            # some tiles also have channels
            num_channels = first_tiles.shape[1]
            merged_size = [
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                int(num_channels),
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            ]

        # create new empty tensor for merged tiles
        merge_buffer = torch.zeros(size=merged_size, device=device)

        # insert tile into merged tensor at right locations
        for (tile_i, tile_j), tile_data in batch_data.items():
            merge_buffer[tile_i, tile_j, ...] = getattr(tile_data, tile_key)

        if single_channel:
            # add channel as tiler needs it
            merge_buffer = merge_buffer.unsqueeze(3)

        # stitch tiles back into whole, output is [B, C, H, W]
        merged_output = self.tiler.untile(merge_buffer)

        if single_channel:
            # remove previously added channels
            merged_output = merged_output.squeeze(1)

        return merged_output

    def merge_labels_and_scores(self, batch_data: dict) -> dict[str, Tensor]:
        """Join scores and their corresponding label predictions from all tiles for each image.

        Label merging is done by rule where one anomalous tile in image results in whole image being anomalous.
        Scores are averaged over tiles.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.

        Returns:
            dict[str, Tensor]: Dictionary with "pred_labels" and "pred_scores"
        """
        # create accumulator with same shape as original
        labels = torch.zeros(batch_data[0, 0].pred_label.shape, dtype=torch.bool)
        scores = torch.zeros(batch_data[0, 0].pred_score.shape)

        for curr_tile_data in batch_data.values():
            curr_labels = curr_tile_data.pred_label
            curr_scores = curr_tile_data.pred_score

            labels = labels.logical_or(curr_labels)
            scores += curr_scores

        scores /= self.tiler.num_tiles

        return {"pred_label": labels, "pred_score": scores}

    def merge_tile_predictions(self, batch_index: int) -> ImageBatch:
        """Join predictions from ensemble into whole image level representation for batch at index batch_index.

        Args:
            batch_index (int): Index of current batch.

        Returns:
            dict[str, Tensor | list]: List of merged predictions for specified batch.
        """
        current_batch_data = self.ensemble_predictions.get_batch_tiles(batch_index)

        # take first tile as base prediction, keep items that are the same over all tiles:
        # image_path, label, mask_path
        merged_predictions = {
            "image_path": current_batch_data[0, 0].image_path,
            "gt_label": current_batch_data[0, 0].gt_label,
        }
        if hasattr(current_batch_data[0, 0], "mask_path"):
            merged_predictions["mask_path"] = current_batch_data[0, 0].mask_path
        
        tiled_data = ["image"]

        # What if the batch does not have a ground truth (gt) mask??
        if hasattr(current_batch_data[0, 0], "gt_mask") and current_batch_data[0,0].gt_mask is not None:
            tiled_data += ["gt_mask"]
        if hasattr(current_batch_data[0, 0], "anomaly_map") and current_batch_data[0, 0].anomaly_map is not None:
            tiled_data += ["anomaly_map"]
        if hasattr(current_batch_data[0, 0], "pred_mask") and current_batch_data[0, 0].pred_mask is not None:
            tiled_data += ["pred_mask"]

        # merge all tiled data
        for t_key in tiled_data:
            if hasattr(current_batch_data[0, 0], t_key):
                print(t_key)
                merged_predictions[t_key] = self.merge_tiles(current_batch_data, t_key)

        # label and score merging
        merged_scores_and_labels = self.merge_labels_and_scores(current_batch_data)
        merged_predictions["pred_label"] = merged_scores_and_labels["pred_label"]
        merged_predictions["pred_score"] = merged_scores_and_labels["pred_score"]

        return ImageBatch(**merged_predictions)


# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - normalization job."""

import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS
from anomalib.utils.normalization.min_max import normalize

logger = logging.getLogger(__name__)


class AOINormalizationJob(Job):
    """Job for normalization of predictions.

    Args:
        predictions (list[Any]): List of predictions.
        root_dir (Path): Root directory containing statistics needed for normalization.
    """

    name = "Normalize"

    def __init__(self, predictions: list[Any] | None, root_dir: Path) -> None:
        super().__init__()
        if isinstance(predictions[0], tuple):
            self.predictions = predictions[0][0]
            self.rest = predictions[0][1:]
        else:
            self.predictions = predictions
            self.rest = None
        self.root_dir = root_dir

    def run(self, task_id: int | None = None) -> list[Any] | None:
        """Run normalization job which normalizes image, pixel and box scores.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: List of normalized predictions.
        """
        del task_id  # not needed here

        # load all statistics needed for normalization
        stats_path = self.root_dir / "stats.json"
        with stats_path.open("r") as f:
            stats = json.load(f)
        minmax = stats["minmax"]
        image_threshold = stats["image_threshold"]
        pixel_threshold = stats["pixel_threshold"]

        logger.info("Starting normalization.")

        for data in tqdm(self.predictions, desc="Normalizing"):
            data.pred_score = normalize(
                data.pred_score,
                image_threshold,
                minmax["pred_score"]["min"],
                minmax["pred_score"]["max"],
            )
            if hasattr(data, "anomaly_map"):
                data.anomaly_map = normalize(
                    data.anomaly_map,
                    pixel_threshold,
                    minmax["anomaly_map"]["min"],
                    minmax["anomaly_map"]["max"],
                )

        return self.predictions, self.rest

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: List of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing is saved in this job."""


class AOINormalizationJobGenerator(JobGenerator):
    """Generate NormalizationJob.

    Args:
        root_dir (Path): Root directory where statistics are saved.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOINormalizationJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[AOINormalizationJob, None, None]:
        """Return a generator producing a single normalization job.

        Args:
            args: not used here.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[NormalizationJob, None, None]: NormalizationJob generator.
        """
        del args  # not needed here

        yield AOINormalizationJob(prev_stage_result, self.root_dir)

# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - metrics calculation job."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from anomalib.metrics import Evaluator
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT, RUN_RESULTS

# from anomalib.utils import NormalizationStage
# from anomalib.utils.helper_functions import get_ensemble_model

logger = logging.getLogger(__name__)


class AOIMetricsCalculationJob(Job):
    """Job for image and pixel metrics calculation.

    Args:
        accelerator (str): Accelerator (device) to use.
        predictions (list[Any]): List of batch predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    name = "Metrics"

    def __init__(
        self,
        accelerator: str,
        predictions: list[Any] | None,
        root_dir: Path,
        evaluator: Evaluator,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        if isinstance(predictions[0], tuple):
            self.predictions = predictions[0][0]
            self.rest = predictions[0][1:]
        else:
            self.predictions = predictions
            self.rest = None
        self.root_dir = root_dir
        self.evaluator = evaluator

    def run(self, task_id: int | None = None) -> dict:
        """Run a job that calculates image and pixel level metrics.

        Args:
            task_id: Not used in this case.

        Returns:
            dict[str, float]: Dictionary containing calculated metric values.
        """
        del task_id  # not needed here

        logger.info("Starting metrics calculation.")

        # add predicted data to metrics
        for data in tqdm(self.predictions, desc="Calculating metrics"):
            # on_test_batch_end updates test metrics
            self.evaluator.on_test_batch_end(None, None, None, batch=data, batch_idx=0)

        # compute all metrics on specified accelerator
        metrics_dict = {}
        for metric in self.evaluator.test_metrics:
            metric.to(self.accelerator)
            metrics_dict[metric.name] = metric.compute().item()
            metric.cpu()

        for name, value in metrics_dict.items():
            print(f"{name}: {value:.4f}")

        # save path used in `save` method
        metrics_dict["save_path"] = self.root_dir / "metric_results.csv"

        return metrics_dict

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: list of predictions.
        """
        # take the first element as result is list of dict here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Save metrics values to csv."""
        logger.info("Saving metrics to csv.")

        # get and remove path from stats dict
        results_path: Path = results.pop("save_path")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        df_dict = {k: [v] for k, v in results.items()}
        metrics_df = pd.DataFrame(df_dict)
        metrics_df.to_csv(results_path, index=False)


class AOIMetricsCalculationJobGenerator(JobGenerator):
    """Generate MetricsCalculationJob.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    def __init__(
        self,
        accelerator: str,
        root_dir: Path,
        model_args: dict,
    ) -> None:
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.model_args = model_args

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIMetricsCalculationJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[AOIMetricsCalculationJob, None, None]:
        """Make a generator that yields a single metrics calculation job.

        Args:
            args: ensemble run config.
            prev_stage_result: ensemble predictions from previous step.

        Returns:
            Generator[MetricsCalculationJob, None, None]: MetricsCalculationJob generator
        """
        del args  # args not used here

        model = get_ensemble_model(self.model_args, normalization_stage=NormalizationStage.IMAGE, input_size=(10, 10))

        if model.evaluator is not None:
            yield AOIMetricsCalculationJob(
                accelerator=self.accelerator,
                predictions=prev_stage_result,
                root_dir=self.root_dir,
                evaluator=model.evaluator,
            )
        else:
            msg = "Model passed to tiled ensemble has no evaluator module which is required to calculate metrics."
            raise RuntimeError(msg)
        


from anomalib.pre_processing import PreProcessor
from anomalib.metrics import AUROC
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage
from anomalib.models import AnomalibModule, get_model
from torchvision.transforms.v2 import Compose, Resize, Transform
from anomalib.pre_processing.utils.transform import get_exportable_transform



def get_ensemble_model(
    model_args: dict[str,dict[str,Any]],
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

    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    # Take evaluator out of model args if possible
    post_processor:PostProcessor|bool = model_args["init_args"].pop("post_processor", False)
    pre_processor:PreProcessor|bool = model_args["init_args"].pop("pre_processor", False)        # TODO Find a way to preserve settings while also overwriting input size with tiling size
    evaluator:Evaluator|bool = model_args["init_args"].pop("evaluator", False)
    visualizer = model_args["init_args"].pop("visualizer", False) 
    if isinstance(visualizer, ImageVisualizer):
        visualizer.field_size = input_size

    if not isinstance(evaluator, Evaluator):
        image_auroc_val = AUROC(fields=["pred_score", "gt_label"], prefix="image_val_")
        pixel_auroc_val = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_val_")
        image_auroc_test = AUROC(fields=["pred_score", "gt_label"], prefix="image_test_")
        pixel_auroc_test = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_test_")
        evaluator = Evaluator(val_metrics=[image_auroc_val, pixel_auroc_val], test_metrics=[image_auroc_test, pixel_auroc_test])

    if isinstance(post_processor, PostProcessor):
        # set model normalisation only if the stage is set to tile level (but thresholding is always applied)
        post_processor.enable_normalization = normalization_stage == NormalizationStage.TILE
        
    # first make temporary model to get object
    temp_model = get_model(model_args)

    # create custom pre_proc with correct input size
    # since we can't modify input_size directly (needed during instantiation by some models like FastFlow)
    _pre_processor = temp_model.configure_pre_processor(input_size)
    
    name = model_args["class_path"]

    model: AnomalibModule = get_model(name, pre_processor=_pre_processor, visualizer=visualizer, evaluator=evaluator, post_processor=post_processor, **model_args["init_args"])
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

    model_args["init_args"]["post_processor"] = post_processor
    model_args["init_args"]["pre_processor"] = pre_processor
    model_args["init_args"]["evaluator"] = evaluator
    model_args["init_args"]["visualizer"] = visualizer

    return model

# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - visualization job."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS
from anomalib.utils.path import generate_output_filename
from anomalib.visualization import visualize_image_item
from anomalib.visualization.image.item_visualizer import (
    DEFAULT_FIELDS_CONFIG,
    DEFAULT_OVERLAY_FIELDS_CONFIG,
    DEFAULT_TEXT_CONFIG,
)

class AOIVisualizationJob(Job):
    """Job for visualization of predictions.

    Args:
        predictions (list[Any]): list of image-level predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        data_args (Dict): data args used to get data name and category name.
    """

    name = "Visualize"

    def __init__(self, predictions: list[Any], root_dir: Path, data_args: dict[str,Any], visualisation_args:dict[str,Any], pred_mask_image:bool=False) -> None:
        super().__init__()
        self.predictions = predictions
        self.predMaskImage = pred_mask_image # If this is the prediction mask is saved as a standalone image
        self.root_dir = root_dir / "images"

        self.fields = visualisation_args.get("fields", None)
        if self.fields is None:
            self.fields = ["image", "pred_mask", "gt_mask"]
        self.overlay_fields = visualisation_args.get("overlay_fields", None)
        if self.overlay_fields is None:
            self.overlay_fields = [("image", ["pred_mask"]), ("image", ["anomaly_map"])]
        self.field_size = visualisation_args.get("field_size", None)
        if self.field_size is None:
            self.field_size = (256,256)
            logger.warning(f"Field size was not given for VisualisationJob. The Visualisation is probaly not right; size of (256,256) is assumed")

        self.fields_config = visualisation_args.get("fields_config", None)
        if self.fields_config is None:
            self.fields_config = DEFAULT_FIELDS_CONFIG

        self.overlay_fields_config = visualisation_args.get("overlay_fields_config", None)
        if self.overlay_fields_config is None:
            self.overlay_fields_config = DEFAULT_OVERLAY_FIELDS_CONFIG

        self.text_config = visualisation_args.get("text_config", None)
        if self.text_config is None:
            self.text_config = DEFAULT_TEXT_CONFIG

        self.dataset_name = data_args["init_args"].get("name", None)
        if self.dataset_name is None:
            # if not specified, take class name
            self.dataset_name = data_args["class_path"].split(".")[-1]
        self.category = data_args["init_args"].get("category", "")

    def run(self, task_id: int | None = None) -> list[Any]:
        """Run job that visualizes all prediction data.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: Unchanged predictions.
        """
        del task_id  # not needed here

        logger.info("Starting visualization.")

        for batch in tqdm(self.predictions, desc="Visualizing"):
            for item in batch:
                image = visualize_image_item(
                    item,
                    fields=self.fields,
                    overlay_fields=self.overlay_fields,
                    field_size=self.field_size,
                    fields_config=self.fields_config,
                    overlay_fields_config=self.overlay_fields_config,
                    text_config=self.text_config,
                )

                # Get the dataset name and category to save the image
                filename = generate_output_filename(
                    input_path=item.image_path or "",
                    output_path=self.root_dir,
                    dataset_name=self.dataset_name,
                    category=self.category,
                )

                if image is not None:
                    # Save the image to the specified filename
                    image.save(filename)

                if self.predMaskImage:
                    predMask = visualize_image_item(
                        item,
                        fields=["pred_mask"],
                        overlay_fields=None,
                        field_size=self.field_size,
                        fields_config=self.fields_config,
                        overlay_fields_config=self.overlay_fields_config,
                        text_config={"enable": False},
                    )
                    if predMask is not None:
                        # Save the image to the specified filename
                        newStem = f"{filename.stem}_predMask"
                        predMask.save(filename.with_stem(newStem))

        return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: Unchanged list of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """This job doesn't save anything."""


class AOIVisualizationJobGenerator(JobGenerator):
    """Generate VisualizationJob.

    Args:
        root_dir (Path): Root directory where images will be saved (root/images).
    """

    def __init__(self, root_dir: Path, data_args: dict[str,Any], visualisation_args:dict[str,Any], pred_mask_image:bool=False) -> None:
        self.root_dir = root_dir
        self.data_args = data_args
        self.visualisation_args = visualisation_args
        self.pred_mask_image = pred_mask_image

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIVisualizationJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[AOIVisualizationJob, None, None]:
        """Return a generator producing a single visualization job.

        Args:
            args: Ensemble run args.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[VisualizationJob, None, None]: VisualizationJob generator
        """
        del args  # args not used here

        if prev_stage_result is not None:
            yield AOIVisualizationJob(prev_stage_result, self.root_dir, data_args=self.data_args, visualisation_args=self.visualisation_args, pred_mask_image=self.pred_mask_image)
        else:
            msg = "Visualization job requires tile level predictions from previous step."
            raise ValueError(msg)
        

import fiftyone as fo
from fiftyone import ViewField as F
import numpy as np

class AOIFiftyOneVisJob(Job):
    """Job for visualization of predictions.

    Args:
        predictions (list[Any]): list of image-level predictions.
        root_dir (Path): Root directory for images.
        data_args (Dict): data args used to get data name and category name.
    """

    name = "Visualize"

    def __init__(self, predictions: list[Any], dataset:FODataset, dataArgs: dict[str,Any], modelName:str) -> None:
        super().__init__()
        if isinstance(predictions[0], tuple):
            self.predictions = predictions[0][0]
            self.rest = predictions[0][1:]
        else:
            self.predictions = predictions
            self.rest = None
        self.dataset:FODataset = dataset
        self.dataArgs:dict[str,Any] = dataArgs
        self.modelName:str = modelName

        self.dataset_name = dataArgs["init_args"].get("name", None)
        if self.dataset_name is None:
            # if not specified, take class name
            self.dataset_name = dataArgs["class_path"].split(".")[-1]
        self.category:str = dataArgs["init_args"].get("category",dataset.first().category.label)
        if self.category == "":
            self.category = dataset.first().category.label

    def run(self, task_id: int | None = None) -> list[Any]:
        """Run job that visualizes all prediction data.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: Unchanged predictions.
        """
        del task_id  # not needed here

        logger.info("Starting visualisation with Fiftyone.")

        for batch in self.predictions:
            for pred in batch:
                path = pred.image_path
                sample = self.dataset[path]
                conf = pred.pred_score.item()
                anomaly = "anomaly" if pred.pred_label else "normal"

                sample[f"pred_anomaly_score_{self.modelName}"] = conf
                sample[f"pred_anomaly_{self.modelName}"] = fo.Classification(label=anomaly)
                heatmap = pred.anomaly_map.to("cpu")
                sample[f"pred_anomaly_map_{self.modelName}"] = fo.Heatmap(map=heatmap.data.numpy().squeeze()*255, range=[0,255])
                if pred.__getattribute__("pred_mask") is not None:
                    mask = pred.pred_mask.to("cpu")
                    sample[f"pred_defect_mask_{self.modelName}"] = fo.Segmentation(mask=mask.data.numpy().squeeze().astype(np.uint8)*255)
                sample.save()
        return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: Unchanged list of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """This job doesn't save anything."""


class AOIFiftyOneVisJobGenerator(JobGenerator):
    """Generate VisualizationJob.

    Args:
        root_dir (Path): Root directory where images will be saved (root/images).
    """

    def __init__(self, dataset, data_args: dict[str,dict[str,Any]], modelName:str) -> None:
        self.data_args = data_args
        self.dataset = dataset
        self.modelName = modelName

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIFiftyOneVisJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[AOIFiftyOneVisJob, None, None]:
        """Return a generator producing a single visualization job.

        Args:
            args: Ensemble run args.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[VisualizationJob, None, None]: VisualizationJob generator
        """
        del args  # args not used here

        if prev_stage_result is not None:
            yield AOIFiftyOneVisJob(prev_stage_result, self.dataset, dataArgs=self.data_args, modelName=self.modelName)
        else:
            msg = "Visualization job requires tile level predictions from previous step."
            raise ValueError(msg)
        
        
from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import get_threshold_values

class AOIThresholdingJob(ThresholdingJob):
    """Job used to threshold predictions, producing labels from scores.

    Args:
        predictions (list[Any]): List of predictions.
        image_threshold (float): Threshold used for image-level thresholding.
        pixel_threshold (float): Threshold used for pixel-level thresholding.
    """

    name = "Threshold"

    def __init__(self, predictions: list[Any] | None, image_threshold: float, pixel_threshold: float) -> None:
        if isinstance(predictions[0], tuple):
            _predictions = predictions[0][0]
            self.rest = predictions[0][1:]
        else:
            _predictions = predictions[0]
        super().__init__(_predictions, image_threshold, pixel_threshold)
        # self.image_threshold = image_threshold
        # self.pixel_threshold = pixel_threshold

    def run(self, task_id: int | None = None) -> list[Any] | None:
        """Run job that produces prediction labels from scores.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: List of thresholded predictions.
        """
        del task_id  # not needed here

        logger.info("Starting thresholding.")

        for data in tqdm(self.predictions, desc="Thresholding"):
            if hasattr(data, "pred_score") and data.pred_score is not None:
                data.pred_label = data.pred_score >= self.image_threshold
            if hasattr(data, "anomaly_map") and data.anomaly_map is not None:
                data.pred_mask = data.anomaly_map >= self.pixel_threshold

        return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: List of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing is saved in this job."""


class AOIThresholdingJobGenerator(JobGenerator):
    """Generate ThresholdingJob.

    Args:
        root_dir (Path): Root directory containing post-processing stats.
    """

    def __init__(self, root_dir: Path, normalization_stage: NormalizationStage) -> None:
        self.root_dir = root_dir
        self.normalization_stage = normalization_stage

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIThresholdingJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[AOIThresholdingJob, None, None]:
        """Return a generator producing a single thresholding job.

        Args:
            args: ensemble run args.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[AOIThresholdingJob, None, None]: AOIThresholdingJob generator.
        """
        del args  # args not used here

        # get threshold values base on normalization
        image_threshold, pixel_threshold = get_threshold_values(self.normalization_stage, self.root_dir)

        yield AOIThresholdingJob(
            predictions=prev_stage_result,
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
        )
# for sample in sample_collection.iter_samples(autosave=True, progress=True):
#         output = engine.predict(data_path=sample.filepath, model=model)[0]
        
#         conf = output.pred_score.item()
#         anomaly = "anomaly" if output.pred_label else "normal"

#         sample[f"pred_anomaly_score_{key}"] = conf
#         sample[f"pred_anomaly_{key}"] = fo.Classification(label=anomaly)
#         sample[f"pred_anomaly_map_{key}"] = fo.Heatmap(map=output.anomaly_map.data.numpy().squeeze()*255, range=[0,255])
#         sample[f"pred_defect_mask_{key}"] = fo.Segmentation(mask=output.pred_mask.data.numpy().squeeze().astype(np.int16)*255)