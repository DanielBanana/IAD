"""
Contains functions that help with managing the checkpoints of the tiled ensemble.
"""
# GENERAL
import math
import re
from pathlib import Path
from typing import List, Tuple

# OWN CODE
from setup import TilingPipelineConfig

def compute_tile_grid(
    image_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> Tuple[int, int]:
    """Number of (rows, cols) tiles produced by the tiler for a given image size.

    NOTE: this must match whatever formula anomalib's Tiler actually uses internally.
    Verify against `anomalib.pipelines.tiled_ensemble` (or the Tiler class itself) once
    and adjust here if it differs — treat this as the single place that assumption lives,
    rather than re-deriving it wherever tile counts are needed.
    """
    n_rows = math.ceil((image_size[0] - tile_size[0]) / stride[0]) + 1
    n_cols = math.ceil((image_size[1] - tile_size[1]) / stride[1]) + 1
    return n_rows, n_cols


def expected_tiled_checkpoint_paths(
    ckptDir: Path,
    tilingPipelineConfig: "TilingPipelineConfig",
) -> List[Path]:
    """All checkpoint file paths expected to exist for a fully trained tiled ensemble."""
    n_rows, n_cols = compute_tile_grid(
        tilingPipelineConfig.image_size,
        tilingPipelineConfig.tile_size,
        tilingPipelineConfig.stride,
    )
    return [
        ckptDir / f"model{r}_{c}.ckpt"
        for r in range(n_rows)
        for c in range(n_cols)
    ]


def checkTiledCheckpointsExist(
    ckptDir: Path,
    tilingPipelineConfig: "TilingPipelineConfig",
) -> Tuple[bool, List[Path]]:
    """Returns (all_present, missing_paths). Empty missing_paths means fully trained."""
    if not ckptDir.exists():
        expected = expected_tiled_checkpoint_paths(ckptDir, tilingPipelineConfig)
        return False, expected

    expected = expected_tiled_checkpoint_paths(ckptDir, tilingPipelineConfig)
    missing = [p for p in expected if not p.exists()]
    return (len(missing) == 0), missing

_CKPT_NAME_RE = re.compile(r"^model(\d+)_(\d+)\.ckpt$")

def find_unexpected_checkpoint_files(ckptDir: Path, tilingPipelineConfig: "TilingPipelineConfig") -> List[Path]:
    """Checkpoint-looking files present that fall outside the current config's tile grid —
    usually leftovers from a previous tiling config in the same directory."""
    if not ckptDir.exists():
        return []
    n_rows, n_cols = compute_tile_grid(
        tilingPipelineConfig.image_size, tilingPipelineConfig.tile_size, tilingPipelineConfig.stride,
    )
    unexpected:List[Path] = []
    for f in ckptDir.glob("model*_*.ckpt"):
        m = _CKPT_NAME_RE.match(f.name)
        if not m or int(m.group(1)) >= n_rows or int(m.group(2)) >= n_cols:
            unexpected.append(f)
    return unexpected