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

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Any
from anomalib.pipelines.tiled_ensemble.components.stats_calculation import StatisticsJob, StatisticsJobGenerator

class AOIStatisticsJob(StatisticsJob):
    def __init__(self, predictions: list[Any] | None, root_dir: Path) -> None:
        super().__init__(predictions, root_dir)

    def run(self, task_id: int | None = None) -> dict:
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
        return {
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
            "save_path": (self.root_dir / "checkpoints" / "stats.json"),
        }
    
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

        tiled_data = ["image", "gt_mask"]
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
