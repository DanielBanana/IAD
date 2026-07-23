"""
Amended version of the TiledEnsembleEngine from anomalib. Changed setup of callbacks and workspace for different file access path.
"""
#
# Created on Fri Feb 06 2026
#
# Copyright (c) 2026 TH Nuernberg - Daniel Pommer
#

from pathlib import Path

from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_engine import TiledEnsembleEngine
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.utils.path import create_versioned_dir
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback



class AOITiledEnsembleEngine(TiledEnsembleEngine):

    def __init__(self, tile_index: tuple[int, int], **kwargs) -> None:
        super().__init__(tile_index, **kwargs)

    @staticmethod
    def setup_ensemble_workspace(args: dict, versioned_dir: bool = True) -> Path:
        """Set up the workspace at the beginning of tiled ensemble training.

        Args:
            args (dict): Tiled ensemble config dict.
            versioned_dir (bool, optional): Whether to create a versioned directory.
                Defaults to ``True``.

        Returns:
            Path: path to new workspace root dir
        """

        return create_versioned_dir(args["rootDir"]) if versioned_dir else args["rootDir"] / "latest"
    
    def _setup_anomalib_callbacks(self) -> None:
        """Modified method to enable individual model training. It's called when Trainer is being set up."""
        callbacks: list[Callback] = []
        tile_i, tile_j = self.tile_index
        checkpoint_dir = self._cache.args["default_root_dir"] / "checkpoints"
        checkpoint_filename = f"model{tile_i}_{tile_j}"

        # If the user configured their own ModelCheckpoint (e.g. via the yaml `callbacks:`
        # list, same mechanism as EarlyStopping), reuse its behaviour settings (monitor,
        # mode, save_top_k, ...) but force dirpath/filename to this tile-aware scheme.
        # A user-supplied dirpath/filename would collide across tiles - every tile in the
        # ensemble shares the same default_root_dir, and the merge/predict/thresholding
        # stages locate each tile's checkpoint by this exact `model{i}_{j}` filename.
        remaining_callbacks = list(self._cache.args["callbacks"])
        existing = next((c for c in remaining_callbacks if isinstance(c, ModelCheckpoint)), None)
        if existing is not None:
            remaining_callbacks = [c for c in remaining_callbacks if c is not existing]
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=checkpoint_filename,
                auto_insert_metric_name=False,
                monitor=existing.monitor,
                mode=existing.mode,
                save_top_k=existing.save_top_k,
                save_last=existing.save_last,  # type: ignore[arg-type]  # runtime value is always bool | "link" | None, same constraint as the constructor
                every_n_epochs=existing.every_n_epochs,
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=checkpoint_filename,
                auto_insert_metric_name=False,
            )
        callbacks.append(checkpoint_callback)
        callbacks.append(TimerCallback())

        # Combine the callbacks, and update the trainer callbacks.
        self._cache.args["callbacks"] = callbacks + remaining_callbacks
