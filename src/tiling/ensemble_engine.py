#
# Created on Fri Feb 06 2026
#
# Copyright (c) 2026 TH Nuernberg - Daniel Pommer
#

from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_engine import TiledEnsembleEngine
from pathlib import Path
from anomalib.utils.path import create_versioned_dir
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback

from anomalib.callbacks import ModelCheckpoint, TimerCallback


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

        # Add ModelCheckpoint if it is not in the callbacks list.
        has_checkpoint_callback = any(isinstance(c, ModelCheckpoint) for c in self._cache.args["callbacks"])
        if not has_checkpoint_callback:
            tile_i, tile_j = self.tile_index
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self._cache.args["default_root_dir"] / "checkpoints",
                    filename=f"model{tile_i}_{tile_j}",
                    auto_insert_metric_name=False,
                ),
            )
        callbacks.append(TimerCallback())

        # Combine the callbacks, and update the trainer callbacks.
        self._cache.args["callbacks"] = callbacks + self._cache.args["callbacks"]
