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
