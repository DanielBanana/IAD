# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run tiled ensemble training."""

# from anomalib.pipelines.tiled_ensemble import EvalTiledEnsemble, TrainTiledEnsemble
from tiling.tiled_ensemble import TrainTiledEnsemble, EvalTiledEnsemble
from AnomalyDataset import importDataset, FODataModule
import argparse
import os
import logging
from rich import traceback
from pathlib import Path
import yaml
traceback.install()
log_file = "runs/pipeline.log"
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run tiled ensemble training and evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/TiledEnsemble.yaml",
        help="Path to the configuration file or config string."
    )
    args = parser.parse_args()

    datamoduleParams = {
        "root":"tileTest",
        "train_batch_size":32,
        "eval_batch_size":32,
        "num_workers":8,
        "train_augmentations":None,
        "val_augmentations":None,
        "test_augmentations":None,
        "augmentations":None,
        "test_split_mode":"from_dir",
        "test_split_ratio":0.2,
        "val_split_mode":"same_as_test",
        "val_split_ratio":0.5
    }

    datasetName = "MVTecADShort"
    dataDir = os.path.join("datasets", datasetName)

    dataset, _ = importDataset(
        path=dataDir,
        name=datasetName + "_tiled",
        overwrite=True,
        split=("train", "test"),
    )

    datamodule = FODataModule(name="Train", samples=dataset, **datamoduleParams)

    with Path(args.config).open(encoding="utf-8") as file:
        parsedArgs = yaml.safe_load(file)

    print("Running tiled ensemble train pipeline")
    train_pipeline = TrainTiledEnsemble()
    train_pipeline.setDatamodule(datamodule=datamodule)
    train_pipeline.setFODataset(dataset=dataset)
    train_pipeline.run(args)

    print("Running tiled ensemble test pipeline.")
    # pass the root dir from train run to load checkpoints
    test_pipeline = EvalTiledEnsemble(train_pipeline.root_dir)
    test_pipeline.setDatamodule(datamodule=datamodule)
    test_pipeline.run(args)
