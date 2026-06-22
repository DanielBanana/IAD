"""
Tiled ensemble training pipeline.
"""
# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# GENERAL
import fiftyone as fo # base library and app
import logging
import torch
import json
import os

from typing import TYPE_CHECKING, List, Any
from collections.abc import Generator
from itertools import product
from pathlib import Path
from lightning import seed_everything, Trainer
from torch.utils.data import DataLoader
from jsonargparse import ArgumentParser, Namespace
from torchvision.transforms.v2 import Compose, Resize, Transform


# ANOMALIB
from anomalib.data import AnomalibDataModule, ImageBatch, get_datamodule, AnomalibDataModule, PredictDataset
from anomalib.data.utils import ValSplitMode, TestSplitMode
from anomalib.models import AnomalibModule, get_model
from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform
from anomalib.post_processing import PostProcessor
from anomalib.metrics import F1Score, AUPR, AUROC, F1AdaptiveThreshold
from anomalib.metrics.evaluator import Evaluator
from anomalib.utils.logging import redirect_logs

# ANOMALIB PIPELINES
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage, PredictData, ThresholdingStage
from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_engine import TiledEnsembleEngine
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
from anomalib.pipelines.tiled_ensemble.components import (
    # MetricsCalculationJobGenerator,
    NormalizationJobGenerator,
    ThresholdingJobGenerator,
    VisualizationJobGenerator,
    StatisticsJobGenerator
)
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT

# OWN FILES
# try:
#     base = Path(__file__).parent
#     ad_base = Path(__file__).parent.parent
# except NameError:
#     base = Path.cwd()  # fallback for notebooks/REPL
from src.data.anomaly_datasets import importDataset, FODataModule, FODataset, AnomalibDataset
from src.tiling.ensemble_engine import AOITiledEnsembleEngine
from src.tiling.ensemble_tiling import EnsembleTiler, TileCollater
from src.tiling.jobs import (
    AOIStatisticsJobGenerator,
    AOIMergeJobGenerator,
    AOINormalizationJobGenerator,
    AOIMetricsCalculationJobGenerator,
    AOIVisualizationJobGenerator,
    AOIFiftyOneVisJobGenerator,
    AOIThresholdingJobGenerator,
    AOISmoothingJobGenerator,
    get_ensemble_model
)

logger = logging.getLogger(__name__)

class TrainTiledEnsemble(Pipeline):
    """Tiled ensemble training pipeline."""

    def __init__(self, rootDir:Path, datamodule:FODataModule|None=None, datamoduleArgs:dict[str,Any]|None=None, FO_Dataset:fo.Dataset|None=None, gtAvail:bool=False) -> None:
        self.rootDir:Path = rootDir
        if datamodule is not None:
            self.setDatamodule(datamodule=datamodule)
        else:
            self.datamodule = datamodule
            self.datamoduleArgs:dict[str,Any]|None = datamoduleArgs
        self.FO_Dataset:fo.Dataset|None = FO_Dataset
        self.gtAvail:bool = gtAvail #TODO add function that sets this value

    def _setup_runners(self, args: dict[str,Any]) -> List[Runner]:
        """Setup the runners for the pipeline.

        This pipeline consists of training and validation steps:
        Training models > prediction on val data > merging val data >
        > (optionally) smoothing seams > calculation of post-processing statistics

        Returns:
            List[Runner]: List of runners executing tiled ensemble train + val jobs.
        """

        seed:int = args.get("seed", 42)
        accelerator:str = args.get("accelerator", "cpu")
        normalization_stage = NormalizationStage(args.get("normalization_stage", "none"))
        thresholding_stage = ThresholdingStage(args.get("thresholding_stage", "none"))
        tArgs:dict[str,Any]|None = args.get("tiling", None)
        if tArgs is None:
            raise AttributeError(f"tiling in {self.__class__} arguments missing")
        else:
            tilingArgs:dict[str,Any] = tArgs
        modelArgs:dict[str,Any]|None = args.get("model",None)
        if modelArgs is None:
            raise AttributeError(f"model arguments missing from config for {self.__class__} pipeline")

        visualisation_args:dict[str,Any] = {
            "field_size": tilingArgs["image_size"],
            "fields": ["image", "pred_mask"] if not self.gtAvail else ["image", "gt_mask", "pred_mask"],
            "overlay_fields": [("image", ["anomaly_map"]), ("image", ["pred_mask"])] if not self.gtAvail else [("image", ["anomaly_map"]), ("image", ["gt_mask"]), ("image", ["pred_mask"])]
        }

        if self.datamoduleArgs is not None:
            # Overwrite data arguments with given datamodule args (Assuming if they are given that they are more important)
            args["data"] = self.datamoduleArgs
            dataArgs = self.datamoduleArgs
        else:
            dataArgs = args.get("data", None) # TODO: Example pipeline takes config and loads from disk, we want to supply the data as a loaded datamodule
        if dataArgs is None:
            raise AttributeError(f"Neither data_args nor datamodule given in {self.__class__} ; Quitting.")
        else:
            assert "init_args" in dataArgs.keys()
            assert "val_split_mode" in dataArgs["init_args"]
        
        runners: List[Runner] = []
        valSplitMode:ValSplitMode = dataArgs["init_args"]["val_split_mode"]

        # 1. train
        train_job_generator = TrainModelJobGenerator(
            seed=seed,
            accelerator=accelerator,
            root_dir=self.rootDir,
            tiling_args=tilingArgs,
            data_args=dataArgs,
            model_args=modelArgs,
            datamodule=self.datamodule,
            normalization_stage=normalization_stage,
        )
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

        if valSplitMode == ValSplitMode.NONE:
            logger.warning("No validation set provided, skipping statistics calculation.")
            return runners

        # 2. predict using validation data
        predict_job_generator = PredictJobGenerator(
            data_source=PredictData.VAL,
            seed=seed,
            accelerator=accelerator,
            root_dir=self.rootDir,
            tiling_args=tilingArgs,
            data_args=dataArgs,
            datamodule=self.datamodule,
            model_args=modelArgs,
            normalization_stage=normalization_stage,
            predictionDataset=None
        )

        if accelerator == "cuda":
            runners.append(
                ParallelRunner(predict_job_generator, n_jobs=torch.cuda.device_count()),
            )
        else:
            runners.append(
                SerialRunner(predict_job_generator),
            )

        # 3. merge predictions
        runners.append(SerialRunner(AOIMergeJobGenerator(tiling_args=tilingArgs, data_args=dataArgs)))

        # 4. (optional) smooth seams
        if args["SeamSmoothing"]["apply"]:
            runners.append(
                SerialRunner(
                    AOISmoothingJobGenerator(accelerator="cpu", tiling_args=tilingArgs, data_args=dataArgs),
                ),
            )

        # 5. calculate statistics used for inference
        runners.append(SerialRunner(AOIStatisticsJobGenerator(self.rootDir)))

        # 6. (optional) normalize
        if normalization_stage == NormalizationStage.IMAGE:
            runners.append(SerialRunner(AOINormalizationJobGenerator(self.rootDir)))
            
        # 7. (optional) threshold to get labels from scores
        if thresholding_stage == ThresholdingStage.IMAGE:
            runners.append(SerialRunner(AOIThresholdingJobGenerator(self.rootDir, normalization_stage)))
        
        # 8 (optional) Associate the results back with the fiftyone dataset where they come from so they can be visualised
        if self.FO_Dataset is not None:
            runners.append(SerialRunner(AOIFiftyOneVisJobGenerator(FO_Dataset=self.FO_Dataset, data_args=dataArgs, modelName=modelArgs["class_path"])))

        return runners

    def setDatamodule(self, datamodule:FODataModule) -> None:
        self.datamodule = datamodule
        self.datamoduleArgs = {
            "init_args": {
                "name": datamodule.name,
                "root": datamodule.root,
                "category": datamodule.category,
                "train_batch_size": datamodule.train_batch_size,
                "eval_batch_size": datamodule.eval_batch_size,
                "num_workers": datamodule.num_workers,
                "train_augmentations": datamodule.train_augmentations,
                "val_augmentations": datamodule.val_augmentations,
                "test_augmentations": datamodule.test_augmentations,
                "augmentations": None,
                "test_split_mode": datamodule.test_split_mode,
                "test_split_ratio": datamodule.test_split_ratio,
                "val_split_mode": datamodule.val_split_mode,
                "val_split_ratio": datamodule.val_split_ratio,
            }
        }

    def setFODataset(self, dataset:FODataset) -> None:
        self.dataset = dataset

    def run(self, args: dict[str,Any], logFile:str|Path|None) -> None:
        """Run the pipeline.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.
        """
        runners:List[Runner] = self._setup_runners(args)
        # redirect_logs(logFile) # dont know what it does
        previous_results: PREV_STAGE_RESULT = None

        for runner in runners:
            try:
                job_args = args.get(runner.generator.job_class.name)
                previous_results = runner.run(job_args or {}, previous_results)
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the runner.")
                print(
                    f"There were some errors when running {runner.generator.job_class.name} with"
                    f" {runner.__class__.__name__}."
                    f" Please check {logFile} for more details.",
                )

class EvalTiledEnsemble(Pipeline):
    """Tiled ensemble evaluation pipeline.

    Args:
        root_dir (Path): Path to root dir of run that contains checkpoints.
    """

    def __init__(self, rootDir:Path, datamodule:AnomalibDataModule|None=None, datamoduleArgs:dict[str,Any]|None=None, FO_Dataset:fo.Dataset|None=None, gtAvail:bool=False, ckptPath:Path|None=None) -> None:
        self.rootDir:Path = rootDir
        self.datamodule = datamodule
        self.datamoduleArgs:dict[str,Any]|None = datamoduleArgs
        if self.datamoduleArgs is None and self.datamodule is not None:
            self.datamoduleArgs = {
                "init_args": {
                    "name": datamodule.name,
                    "root": datamodule.root,
                    "category": datamodule.category,
                    "train_batch_size": datamodule.train_batch_size,
                    "eval_batch_size": datamodule.eval_batch_size,
                    "num_workers": datamodule.num_workers,
                    "train_augmentations": datamodule.train_augmentations,
                    "val_augmentations": datamodule.val_augmentations,
                    "test_augmentations": datamodule.test_augmentations,
                    "augmentations": None,
                    "test_split_mode": datamodule.test_split_mode,
                    "test_split_ratio": datamodule.test_split_ratio,
                    "val_split_mode": datamodule.val_split_mode,
                    "val_split_ratio": datamodule.val_split_ratio,
                }
            }
        
        self.FO_Dataset = FO_Dataset
        self.gtAvail:bool = gtAvail #TODO add function that sets this value
        self.ckptPath = ckptPath

    def _setup_runners(self, args: dict[str,Any]) -> List[Runner]:
        """Set up the runners for the pipeline.

        This pipeline consists of jobs used to test/evaluate tiled ensemble:
        Prediction on test data > merging of predictions > (optional) seam smoothing
        > (optional) Normalization > (optional) Thresholding
        > Visualisation of predictions > Metrics calculation.

        Returns:
            List[Runner]: List of runners executing tiled ensemble testing jobs.
        """
        runners: List[Runner] = []

        seed:int = args.get("seed", 42)
        if self.ckptPath is None:
            logger.info(f"No ckptPath given in arguments to {self.__class__}. Going over rootDir: {self.rootDir}/checkpoints")
            ckptPath:Path = self.rootDir / "checkpoints"
        else:
            ckptPath:Path = self.ckptPath
            logger.info(f"Loading Checkpoints from path: {ckptPath}")
        accelerator:str = args.get("accelerator", "cpu")
        normalization_stage = NormalizationStage(args.get("normalization_stage", "none"))
        thresholding_stage = ThresholdingStage(args.get("thresholding_stage", "none"))
        tArgs:dict[str,Any]|None = args.get("tiling", None)
        if tArgs is None:
            raise AttributeError(f"tiling in {self.__class__} arguments missing")
        else:
            tilingArgs:dict[str,Any] = tArgs

        mArgs:dict[str,Any]|None = args.get("model",None)
        if mArgs is None:
            raise AttributeError(f"model arguments missing from config for {self.__class__} pipeline")
        else:
            modelArgs = mArgs

        visualisation_args:dict[str,Any] = {
            "field_size": tilingArgs["image_size"],
            "fields": ["image", "pred_mask"] if not self.gtAvail else ["image", "gt_mask", "pred_mask"],
            "overlay_fields": [("image", ["anomaly_map"]), ("image", ["pred_mask"])] if not self.gtAvail else [("image", ["anomaly_map"]), ("image", ["gt_mask"]), ("image", ["pred_mask"])]
        }

        if self.datamoduleArgs is not None:
            # Overwrite data arguments with given datamodule args (Assuming if they are given that they are more important)
            args["data"] = self.datamoduleArgs
            dataArgs = self.datamoduleArgs
        else:
            dataArgs = args.get("data", None) # TODO: Example pipeline takes config and loads from disk, we want to supply the data as a loaded datamodule
        if dataArgs is None:
            raise AttributeError(f"Neither data_args nor datamodule given in {self.__class__} ; Quitting.")
        else:
            assert "init_args" in dataArgs.keys()
            assert "val_split_mode" in dataArgs["init_args"]
        
        valSplitMode:ValSplitMode = dataArgs["init_args"]["val_split_mode"]
        if valSplitMode == TestSplitMode.NONE:
            logger.info("Test split mode set to `none`, skipping test phase.")
            return runners
#############

        predict_job_generator = PredictJobGenerator(
            data_source=PredictData.TEST,
            seed=seed,
            accelerator=accelerator,
            root_dir=self.rootDir,
            tiling_args=tilingArgs,
            data_args=dataArgs,
            datamodule=self.datamodule,
            model_args=modelArgs,
            normalization_stage=normalization_stage,
            ckptPath=ckptPath,
            predictionDataset=None
        )

        # fo_predict_job_generator = FOPredictJobGenerator(
        #     PredictData.TEST,
        #     seed=seed,
        #     accelerator=accelerator,
        #     root_dir=self.root_dir,
        #     tiling_args=tiling_args,
        #     data_args=data_args,
        #     datamodule=self.datamodule,
        #     model_args=model_args,
        #     normalization_stage=normalization_stage,
        #     dataset = self.dataset
        # )
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
        runners.append(SerialRunner(AOIMergeJobGenerator(tiling_args=tilingArgs, data_args=dataArgs)))

        # 3. (optional) smooth seams
        if args["SeamSmoothing"]["apply"]:
            runners.append(
                SerialRunner(
                    AOISmoothingJobGenerator(accelerator="cpu", tiling_args=tilingArgs, data_args=dataArgs),
                ),
            )

        # 4. (optional) normalize
        if normalization_stage == NormalizationStage.IMAGE:
            logger.info(f"Taking stats for Nomalization from: {self.rootDir if self.ckptPath is None else self.ckptPath.parent}")
            runners.append(SerialRunner(AOINormalizationJobGenerator(self.rootDir if self.ckptPath is None else self.ckptPath.parent)))

        # 5. (optional) threshold to get labels from scores
        if thresholding_stage == ThresholdingStage.IMAGE:
            logger.info(f"Taking stats for Thresholding from: {self.rootDir if self.ckptPath is None else self.ckptPath.parent}")
            runners.append(SerialRunner(AOIThresholdingJobGenerator(self.rootDir if self.ckptPath is None else self.ckptPath.parent, normalization_stage)))

        # # 6. visualize predictions
        if self.FO_Dataset is not None:
            runners.append(SerialRunner(AOIFiftyOneVisJobGenerator(FO_Dataset=self.FO_Dataset, data_args=dataArgs, modelName=modelArgs["class_path"])))

        # calculate metrics
        runners.append(
            SerialRunner(
                AOIMetricsCalculationJobGenerator(
                    accelerator=accelerator,
                    root_dir=self.rootDir,
                    model_args=modelArgs,
                ),
            ),
        )

        return runners
    
    def setDatamodule(self, datamodule: FODataModule):
        self.datamodule = datamodule
        self.datamoduleArgs = {
            "init_args": {
                "name": datamodule.name,
                "root": datamodule.root,
                "category": datamodule.category,
                "train_batch_size": datamodule.train_batch_size,
                "eval_batch_size": datamodule.eval_batch_size,
                "num_workers": datamodule.num_workers,
                "train_augmentations": datamodule.train_augmentations,
                "val_augmentations": datamodule.val_augmentations,
                "test_augmentations": datamodule.test_augmentations,
                "augmentations": None,
                "test_split_mode": datamodule.test_split_mode,
                "test_split_ratio": datamodule.test_split_ratio,
                "val_split_mode": datamodule.val_split_mode,
                "val_split_ratio": datamodule.val_split_ratio,
            }
        }

    def setFODataset(self, dataset:FODataset):
        self.dataset = dataset

    def run(self, args: dict[str,Any], logFile:str|Path|None) -> None:
        """Run the pipeline.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.
        """
        runners:List[Runner] = self._setup_runners(args)
        # redirect_logs(logFile) # dont know what it does
        previous_results: PREV_STAGE_RESULT = None

        for runner in runners:
            try:
                job_args = args.get(runner.generator.job_class.name)
                # Start the runner (serial or parallel) that contains a job/multiple jobs. This calls the
                # run(), collect() and save() methods of the Jobs to calculate the result of the job, collect the
                # important bits for the next job (previous_results) and save something to disk if necessary.
                previous_results = runner.run(job_args or {}, previous_results)
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the runner.")
                print(
                    f"There were some errors when running {runner.generator.job_class.name} with"
                    f" {runner.__class__.__name__}."
                    f" Please check {logFile} for more details.",
                )

class PredTiledEnsemble(Pipeline):
    """Tiled ensemble evaluation pipeline.

    Args:
        root_dir (Path): Path to root dir of run that contains checkpoints.
    """

    def __init__(self,
                 root_dir: Path,
                 trainingDir:Path,
                 ckptDir:Path,
                 predictDataset:PredictDataset,
                 dataset:fo.Dataset,
                 datamodule:FODataModule|None=None,
                 datamoduleArgs:dict[str,Any]|None=None,
                 gtAvail:bool=False) -> None:
        self.root_dir = Path(root_dir)                                          # Where this pipeline stores results from
        self.trainingDir = Path(trainingDir)   
        self.ckptDir = ckptDir
        logger.debug(f"Current working directory (cwd): {os.getcwd()}")                                 # Where the training pipeline stored results like threshold and normalization stats
        logger.info(f"Root directory for Eval Pipeline: {root_dir}")
        logger.info(f"Checkpoint directory: {ckptDir}")
        logger.info(f"Stats directory: {trainingDir}")
        self.dataset:fo.Dataset = dataset
        self.predictDataset = predictDataset
        self.datamodule:AnomalibDataModule|None = datamodule
        self.datamoduleArgs:dict[str,Any]|None = datamoduleArgs
        self.gtAvail = gtAvail
        if self.datamoduleArgs is None:
            self.datamoduleArgs = {
                "init_args": {
                    "name": datamodule.name,
                    "root": datamodule.root,
                    "category": datamodule.category,
                    "train_batch_size": datamodule.train_batch_size,
                    "eval_batch_size": datamodule.eval_batch_size,
                    "num_workers": datamodule.num_workers,
                    "train_augmentations": datamodule.train_augmentations,
                    "val_augmentations": datamodule.val_augmentations,
                    "test_augmentations": datamodule.test_augmentations,
                    "augmentations": None,
                    "test_split_mode": datamodule.test_split_mode,
                    "test_split_ratio": datamodule.test_split_ratio,
                    "val_split_mode": datamodule.val_split_mode,
                    "val_split_ratio": datamodule.val_split_ratio,
                }
            }

    def _setup_runners(self, args: dict[str,Any]) -> List[Runner]:
        """Set up the runners for the pipeline.

        This pipeline consists of jobs used to test/evaluate tiled ensemble:
        Prediction on test data > merging of predictions > (optional) seam smoothing
        > (optional) Normalization > (optional) Thresholding
        > Visualisation of predictions > Metrics calculation.

        Returns:
            List[Runner]: List of runners executing tiled ensemble testing jobs.
        """

        logger.info("Setting up runners")

        runners: List[Runner] = []
   
        seed:int = int(args.get("seed", 0))
        ckptPath:Path|str|None = args.get("ckptPath",None)
        if ckptPath is not None:
            ckptPath = Path(ckptPath)
            logger.info(f"Checkpoint path: {ckptPath}")
        else:
            ValueError(f"ckptPath missing for {self.__class__} pipeline")

        accelerator:str = args.get("accelerator", "cpu")

        tiling_args = args.get("tiling", None)
        if tiling_args is None:
            raise ValueError(f"No Tiling args given to {self} pipeline")
        
        if self.datamoduleArgs is not None:
            # Overwrite data arguments with given datamodule args (Assuming if they are given that they are more important)
            args["data"] = self.datamoduleArgs
            dataArgs = self.datamoduleArgs
        else:
            dataArgs = args.get("data", None) # TODO: Example pipeline takes config and loads from disk, we want to supply the data as a loaded datamodule
        if dataArgs is None:
            raise AttributeError(f"Neither data_args nor datamodule given in {self.__class__} ; Quitting.")
        else:
            assert "init_args" in dataArgs.keys()
            assert "val_split_mode" in dataArgs["init_args"]
        # valSplitMode:ValSplitMode = dataArgs["init_args"]["val_split_mode"]
        # if valSplitMode == TestSplitMode.NONE:
        #     logger.info("Test split mode set to `none`, skipping test phase.")
        #     return runners
        
        nStage:str|None = args.get("normalization_stage", None)
        if nStage is None:
            logger.warning(f"Normalization-Stage not given to {self} pipeline")
            normalization_stage = NormalizationStage("none")
        else:
            normalization_stage = NormalizationStage(nStage)

        tStage:str|None = args.get("thresholding_stage", None)
        if tStage is None:
            logger.warning(f"ThresholdingStage not given to {self} pipeline")
            thresholding_stage = ThresholdingStage("image")
        else:
            thresholding_stage = ThresholdingStage(tStage)
        
        mArgs:dict[str,Any]|None = args.get("model",None)
        if mArgs is None:
            raise AttributeError(f"model arguments missing from config for {self.__class__} pipeline")
        else:
            modelArgs = mArgs        

        visualisation_args:dict[str,Any] = {
            "field_size": tiling_args["image_size"],
            "fields": ["image", "pred_mask"] if not self.gtAvail else ["image", "gt_mask", "pred_mask"],
            "overlay_fields": [("image", ["anomaly_map"]), ("image", ["pred_mask"])] if not self.gtAvail else [("image", ["anomaly_map"]), ("image", ["gt_mask"]), ("image", ["pred_mask"])]
        }

        logger.debug(self.dataset)

        logger.debug("Setting up JobGenerators")

        predict_job_generator = PredictJobGenerator(
            PredictData.TEST,
            seed=seed,
            accelerator=accelerator,
            root_dir=self.root_dir,
            tiling_args=tiling_args,
            data_args=dataArgs,
            model_args=modelArgs,
            normalization_stage=normalization_stage,
            datamodule=self.datamodule,
            predictionDataset=self.predictDataset,
            ckptPath=ckptPath
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
        runners.append(SerialRunner(AOIMergeJobGenerator(tiling_args=tiling_args, data_args=dataArgs)))

        # 3. (optional) smooth seams
        if args["SeamSmoothing"]["apply"]:
            runners.append(
                SerialRunner(
                    AOISmoothingJobGenerator(accelerator="cpu", tiling_args=tiling_args, data_args=dataArgs),
                ),
            )

        # 4. (optional) normalize
        if normalization_stage == NormalizationStage.IMAGE:
             runners.append(SerialRunner(AOINormalizationJobGenerator(self.trainingDir)))

        # 5. (optional) threshold to get labels from scores
        if thresholding_stage == ThresholdingStage.IMAGE:
            runners.append(SerialRunner(AOIThresholdingJobGenerator(self.trainingDir, normalization_stage)))

        runners.append(SerialRunner(AOIVisualizationJobGenerator(self.root_dir, data_args=dataArgs, visualisation_args=visualisation_args, pred_mask_image=True)))

        # # 6. visualize predictions
        # if self.dataset is not None:
        runners.append(SerialRunner(AOIFiftyOneVisJobGenerator(FO_Dataset=self.dataset, data_args=dataArgs, modelName=modelArgs["class_path"])))

        # # calculate metrics
        # runners.append(
        #     SerialRunner(
        #         AOIMetricsCalculationJobGenerator(
        #             accelerator=accelerator,
        #             root_dir=self.root_dir,
        #             model_args=modelArgs,
        #         ),
        #     ),
        # )

        logger.debug("Finished JobGenerator setup!")

        return runners
    
    def setDatamodule(self, datamodule: FODataModule):
        self.datamodule = datamodule
        self.datamoduleArgs = {
            "init_args": {
                "name": datamodule.name,
                "root": datamodule.root,
                "category": datamodule.category,
                "train_batch_size": datamodule.train_batch_size,
                "eval_batch_size": datamodule.eval_batch_size,
                "num_workers": datamodule.num_workers,
                "train_augmentations": datamodule.train_augmentations,
                "val_augmentations": datamodule.val_augmentations,
                "test_augmentations": datamodule.test_augmentations,
                "augmentations": None,
                "test_split_mode": datamodule.test_split_mode,
                "test_split_ratio": datamodule.test_split_ratio,
                "val_split_mode": datamodule.val_split_mode,
                "val_split_ratio": datamodule.val_split_ratio,
            }
        }

    def setFODataset(self, dataset:FODataset):
        self.dataset = dataset

    def run(self, args: dict[str,Any], logFile:str|Path|None) -> None:
        """Run the pipeline.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.
        """
        runners:List[Runner] = self._setup_runners(args)
        # redirect_logs(logFile) # dont know what it does
        previous_results: PREV_STAGE_RESULT = None

        for runner in runners:
            try:
                job_args = args.get(runner.generator.job_class.name)
                previous_results = runner.run(job_args or {}, previous_results)
                # logger.info(f"Finished {runner.generator.job_class.name}.")
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the runner.")
                print(
                    f"There were some errors when running {runner.generator.job_class.name} with"
                    f" {runner.__class__.__name__}."
                    f" Please check {logFile} for more details.",
                )


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
        trainer_args: dict[str,Any] | None,
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
    ) -> AOITiledEnsembleEngine:
        """Run train job that fits the model for given tile location.

        Args:
            task_id: Passed when job is ran in parallel.

        Returns:
            AOITiledEnsembleEngine: Engine containing trained model.
        """
        devices: str | List[int] = "auto"
        if task_id is not None:
            devices = [task_id]
            logger.info(f"Running job {self.model.__class__.__name__} with device {task_id}")

        logger.info(f"Running {self.__class__}")
        logger.info("Training for tile at position %s,", self.tile_index)
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
    def collect(results: List[AOITiledEnsembleEngine]) -> dict[tuple[int, int], AOITiledEnsembleEngine]:
        """Collect engines from each tile location into a dict.

        Returns:
            dict[tuple[int, int], AOITiledEnsembleEngine]: Dict has form {tile_index: AOITiledEnsembleEngine}
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
        tiling_args: dict[str,Any],
        data_args: dict[str,Any],
        model_args: dict[str,Any],
        datamodule: AnomalibDataModule|None,
        normalization_stage: NormalizationStage,
    ) -> None:
        self.seed = seed
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.tiling_args = tiling_args
        self.data_args = data_args,
        self.model_args = model_args
        self.datamodule = datamodule
        self.normalization_stage = normalization_stage

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return TrainModelJob

    def generate_jobs(
        self,
        args: dict[str,Any] | None = None,
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
                data_config=args,
                image_size=self.tiling_args["image_size"],
                tiler=tiler,
                tile_index=tile_index,
                datamodule=self.datamodule
            )
            model = get_ensemble_model(
                model_args=self.model_args,
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
        engine (AOITiledEnsembleEngine | None):
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
        dataloader: DataLoader[FODataset|PredictDataset],
        model: AnomalibModule | None,
        engine: AOITiledEnsembleEngine | None,
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
            tuple[tuple[int, int], List[Any]]: Tile index, List of predictions.
        """
        devices: str | List[int] = "auto"
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

        logger.info(f"Length of dataloader: {len(self.dataloader)}")
        predictions:list[Any]|list[list[Any]]|None = self.engine.predict(model=self.model, dataloaders=self.dataloader, ckpt_path=self.ckpt_path)

        # also return tile index as it's needed in collect method

        return self.tile_index, predictions

    @staticmethod
    def collect(results: List[tuple[tuple[int, int], List[Any]]]) -> EnsemblePredictions:
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
        tiling_args: dict[str,Any],
        data_args: dict[str,Any],
        model_args: dict[str,Any],
        normalization_stage: NormalizationStage,
        datamodule:AnomalibDataModule|None,
        predictionDataset:PredictDataset|None,
        ckptPath:Path|None=None
    ) -> None:
        self.data_source = data_source
        self.seed = seed
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.tiling_args = tiling_args
        self.data_args = data_args
        self.model_args = model_args
        self.normalization_stage = normalization_stage
        self.datamodule = datamodule
        self.ckptPath = ckptPath
        self.predictionDataset = predictionDataset

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return PredictJob

    def generate_jobs(
        self,
        args: dict[str,Any] | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[PredictJob, None, None]:
        """Generate predict jobs for each tile location.

        Args:
            args (dict): Dict with config passed to training.
            prev_stage_result (dict[tuple[int, int], AOITiledEnsembleEngine] | None):
                if called after train job this contains engines with individual models, otherwise load from checkpoints.

        Returns:
            Generator[PredictJob, None, None]: PredictJob generator.
        """
        # del args  # args not used here

        # tiler used for splitting the image and getting the tile count
        tiler = get_ensemble_tiler(self.tiling_args)

        logger.info(
            "Tiled ensemble predicting started using Using ckpt_path%s data.",
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
                datamodule=self.datamodule
            )

            # check if predict step is positioned after training
            if prev_stage_result and tile_index in prev_stage_result:
                engine = prev_stage_result[tile_index]
                # model is inside engine in this case
                if isinstance(engine, AOITiledEnsembleEngine):
                    model = engine.model
                else:
                    raise AttributeError(f"prev_stage_result does not contain engine")
                ckpt_path = None
                logger.info(f"Using model from previous training job. No loading from checkpoint")
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
                
                # ckpt_path = self.root_dir / "weights" / "lightning" / f"model{tile_i}_{tile_j}.ckpt"
                if self.ckptPath is None:
                    ckpt_path = self.root_dir / "checkpoints" / f"model{tile_i}_{tile_j}.ckpt"
                else:
                    ckpt_path = self.ckptPath / f"model{tile_i}_{tile_j}.ckpt"
                logger.info(f"Loading checkpoint from ckpt_path: {ckpt_path}. No Model from previous training job available.")
                

            # pick the dataloader based on predict data
            if self.predictionDataset:
                logger.info(f"Dataset for dataloader: {self.predictionDataset}")
                dataloader:DataLoader[PredictDataset] = DataLoader(self.predictionDataset, collate_fn=datamodule.external_collate_fn, pin_memory=True)
            else:
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

class FOPredictJob(Job):
    """Job for generating predictions with individual models in the tiled ensemble.

    Args:
        accelerator (str): Accelerator (device) to use.
        seed (int): Random seed for reproducibility.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        tile_index (tuple[int, int]): Index of tile that this model processes.
        normalization_stage (str): Normalization stage flag.
        dataloader (DataLoader): Dataloader to use for training (either val or test).
        model (AnomalyModule): Model to train.
        engine (AOITiledEnsembleEngine | None):
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
        dataloader:DataLoader[FODataset],
        foDataset:FODataset,
        model: AnomalibModule | None,
        engine: AOITiledEnsembleEngine | None,
        ckpt_path: Path | None,
        key: str,
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
        self.foDataset = foDataset
        self.model = model
        self.engine = engine
        self.ckpt_path = ckpt_path
        self.key = key

    def run(
        self,
        task_id: int | None = None,
    ) -> tuple[tuple[int, int], Any | None]:
        """Predict job that predicts the data with specific model for given tile location.

        Args:
            task_id: Passed when job is ran in parallel.

        Returns:
            tuple[tuple[int, int], List[Any]]: Tile index, List of predictions.
        """
        devices: str | List[int] = "auto"
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

        for sample in self.foDataset.iter_samples(autosave=True, progress=True):
            predictions = self.engine.predict(model=self.model, data_path=sample.filepath, ckpt_path=self.ckpt_path)

            output = predictions[0]

            conf = output.pred_score.item()
            anomaly = "anomaly" if output.pred_label else "normal"

            sample[f"pred_anomaly_score_{self.key}"] = conf
            sample[f"pred_anomaly_{self.key}"] = fo.Classification(label=anomaly)
            sample[f"pred_anomaly_map_{self.key}"] = fo.Heatmap(map=output.anomaly_map.data.numpy().squeeze()*255, range=[0,255])
            sample[f"pred_defect_mask_{self.key}"] = fo.Segmentation(mask=output.pred_mask.data.numpy().squeeze().astype(np.int16)*255)

        # also return tile index as it's needed in collect method
        return self.tile_index, predictions

    @staticmethod
    def collect(results: List[tuple[tuple[int, int], List[Any]]]) -> EnsemblePredictions:
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

class FOPredictJobGenerator(JobGenerator):
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
        datamodule,
        dataset
    ) -> None:
        self.data_source = data_source
        self.seed = seed
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.tiling_args = tiling_args
        self.data_args = data_args
        self.model_args = model_args
        self.normalization_stage = normalization_stage
        self.datamodule_ = datamodule
        self.dataset = dataset

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return FOPredictJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[FOPredictJob, None, None]:
        """Generate predict jobs for each tile location.

        Args:
            args (dict): Dict with config passed to training.
            prev_stage_result (dict[tuple[int, int], AOITiledEnsembleEngine] | None):
                if called after train job this contains engines with individual models, otherwise load from checkpoints.

        Returns:
            Generator[PredictJob, None, None]: PredictJob generator.
        """
        # del args  # args not used here

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
                datamodule=self.datamodule_
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
                # ckpt_path = self.root_dir / "weights" / "lightning" / f"model{tile_i}_{tile_j}.ckpt"
                ckpt_path = self.root_dir / "checkpoints" / f"model{tile_i}_{tile_j}.ckpt"

            # pick the dataloader based on predict data
            dataloader = datamodule.test_dataloader()
            if self.data_source == PredictData.VAL:
                dataloader = datamodule.val_dataloader()

            # pass root_dir to engine so all models in ensemble have the same root dir
            yield FOPredictJob(
                accelerator=self.accelerator,
                seed=self.seed,
                root_dir=self.root_dir,
                tile_index=tile_index,
                normalization_stage=self.normalization_stage,
                model=model,
                dataloader=dataloader,
                engine=engine,
                ckpt_path=ckpt_path,
                foDataset=self.dataset,
                key=self.model_args["class_path"]
            )

"""Helper functions for the tiled ensemble training."""

def get_ensemble_datamodule(
    data_config: dict[str,Any],
    image_size: int | tuple[int, int],
    tiler: EnsembleTiler,
    tile_index: tuple[int, int],
    datamodule: AnomalibDataModule|None = None,
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
    if datamodule == None:
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
    devices: List[int] | str | int,
    root_dir: Path,
    trainer_args: dict | None = None,
) -> AOITiledEnsembleEngine:
    """Prepare engine for ensemble training or prediction.

    This method makes sure correct normalization is used, prepares metrics and additional trainer kwargs..

    Args:
        tile_index (tuple[int, int]): Index of tile that this model processes.
        accelerator (str): Accelerator (device) to use.
        devices (List[int] | str | int): device IDs used for training.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        trainer_args (dict): Trainer args dictionary. Empty dict if not present.

    Returns:
        AOITiledEnsembleEngine: set up engine for ensemble training/prediction.
    """
    # parse additional trainer args and callbacks if present in config
    trainer_kwargs = parse_trainer_kwargs(trainer_args)
    # remove keys that we already have
    trainer_kwargs.pop("accelerator", None)
    trainer_kwargs.pop("default_root_dir", None)
    trainer_kwargs.pop("devices", None)

    # create engine for specific tile location
    engine = AOITiledEnsembleEngine(
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
        stats_path = root_dir /"checkpoints" / "stats.json"
        with stats_path.open("r") as f:
            stats = json.load(f)
        image_threshold = stats["image_threshold"]
        pixel_threshold = stats["pixel_threshold"]
    else:
        # normalization transforms the scores so that threshold is at 0.5
        image_threshold = 0.5
        pixel_threshold = 0.5

    return image_threshold, pixel_threshold

