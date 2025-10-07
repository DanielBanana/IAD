# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run tiled ensemble training."""

from anomalib.pipelines.tiled_ensemble import EvalTiledEnsemble, TrainTiledEnsemble
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run tiled ensemble training and evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/TiledEnsemble.yaml",
        help="Path to the configuration file or config string."
    )
    args = parser.parse_args()

    print("Running tiled ensemble train pipeline")
    train_pipeline = TrainTiledEnsemble()
    # run training
    train_pipeline.run(args)

    print("Running tiled ensemble test pipeline.")
    # pass the root dir from train run to load checkpoints
    test_pipeline = EvalTiledEnsemble(train_pipeline.root_dir)
    test_pipeline.run(args)

    