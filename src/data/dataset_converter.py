"""
Convert a fabric-defect dataset into the MVTec-style anomalib directory format.

Source layout:
    datasetName/
        Defect_images/    nnnn_ddd_ff.png
        Mask_images/      nnnn_ddd_ff_mask.png
        NODefect_Images/  nnnn_000_ff.png

Target layout:
    dest/datasetName/categoryName/
        train/good/           000.png, 001.png, ...
        test/good/            000.png, 001.png, ...
        test/<defect_name>/   000.png, 001.png, ...      (e.g. broken_end, nep)
        ground_truth/<defect_name>/  000_mask.png, 001_mask.png, ...
"""

import re
import shutil
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFECT_RE = re.compile(r"^(\d{4})_(\d{3})_(\w+)\.png$", re.IGNORECASE)
_MASK_RE = re.compile(r"^(\d{4})_(\d{3})_(\w+)_mask\.png$", re.IGNORECASE)

# Maps 3-digit defect code strings to human-readable folder names.
DEFECT_CODE_MAP: dict[str, str] = {
    "002": "broken_end",
    "006": "broken_yarn",
    "010": "broken_pick",
    "016": "weft_curling",
    "019": "fuzzyball",
    "022": "cut_selvage",
    "023": "crease",
    "025": "warp_ball",
    "027": "knots",
    "029": "contamination",
    "030": "nep",
    "036": "weft_crack",
}


def convert_fabric_dataset(
    source_dir: str | Path,
    dest_dir: str | Path,
    dataset_name: str,
    category_name: str,
    train_ratio: float = 0.8,
    fabric_code: str | None = None,
) -> None:
    """Reformat a fabric-defect dataset into the MVTec-style anomalib format.

    Args:
        source_dir: Root of the source dataset (contains Defect_images/, etc.).
        dest_dir: Root where the reformatted dataset will be written.
        dataset_name: Folder name for the dataset inside dest_dir.
        category_name: Category subfolder name (e.g. a fabric type label).
        train_ratio: Fraction of defect-free images used for training. The
            remainder goes into test/good. Defaults to 0.8.
        fabric_code: If given, only process images whose filename fabric code
            matches this string (e.g. "01"). Defaults to None (all fabrics).
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    category_root = dest_dir / dataset_name / category_name

    defect_src = source_dir / "Defect_images"
    mask_src = source_dir / "Mask_images"
    good_src = source_dir / "NODefect_Images"

    for folder, label in [(defect_src, "Defect_images"), (mask_src, "Mask_images"), (good_src, "NODefect_Images")]:
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected folder not found: {folder}  (looked for '{label}')")

    # --- collect defective images grouped by defect code ---
    defect_images: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    for f in sorted(defect_src.glob("*.png")):
        m = _DEFECT_RE.match(f.name)
        if not m:
            logger.warning("Skipping unrecognised file: %s", f.name)
            continue
        img_num, defect_code, fc = m.group(1), m.group(2), m.group(3)
        if fabric_code and fc != fabric_code:
            continue
        defect_images[defect_code].append((img_num, fc, f))

    # --- build mask lookup: (img_num, defect_code, fabric_code) -> Path ---
    mask_lookup: dict[tuple[str, str, str], Path] = {}
    for f in sorted(mask_src.glob("*.png")):
        m = _MASK_RE.match(f.name)
        if not m:
            continue
        img_num, defect_code, fc = m.group(1), m.group(2), m.group(3)
        if fabric_code and fc != fabric_code:
            continue
        mask_lookup[(img_num, defect_code, fc)] = f

    # --- collect defect-free images ---
    good_images: list[tuple[str, str, Path]] = []
    for f in sorted(good_src.glob("*.png")):
        m = _DEFECT_RE.match(f.name)
        if not m:
            logger.warning("Skipping unrecognised file: %s", f.name)
            continue
        img_num, defect_code, fc = m.group(1), m.group(2), m.group(3)
        if fabric_code and fc != fabric_code:
            continue
        if defect_code != "000":
            logger.warning("NODefect_Images contains non-zero defect code, skipping: %s", f.name)
            continue
        good_images.append((img_num, fc, f))

    if not good_images:
        logger.warning("No defect-free images found (fabric_code filter: %s)", fabric_code)

    # --- split good images into train / test ---
    split_idx = max(1, round(len(good_images) * train_ratio))
    train_good = good_images[:split_idx]
    test_good = good_images[split_idx:]

    # --- write files ---
    _copy(train_good, category_root / "train" / "good")
    _copy(test_good, category_root / "test" / "good")

    for defect_code, images in defect_images.items():
        folder_name = DEFECT_CODE_MAP.get(defect_code, defect_code)
        if folder_name == defect_code:
            logger.warning("Unknown defect code '%s' — using raw code as folder name", defect_code)
        _copy(images, category_root / "test" / folder_name)
        mask_targets: list[tuple[str, str, Path]] = []
        for img_num, fc, _ in images:
            mask_path = mask_lookup.get((img_num, defect_code, fc))
            if mask_path is None:
                logger.warning("No mask found for %s_%s_%s", img_num, defect_code, fc)
            else:
                mask_targets.append((img_num, fc, mask_path))
        _copy(mask_targets, category_root / "ground_truth" / folder_name, mask_suffix=True)

    resolved = {DEFECT_CODE_MAP.get(c, c) for c in defect_images}
    logger.info(
        "Done. train/good: %d  test/good: %d  defect types: %s",
        len(train_good),
        len(test_good),
        sorted(resolved),
    )


def _copy(
    images: list[tuple[str, str, Path]],
    out_dir: Path,
    mask_suffix: bool = False,
) -> None:
    if not images:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (_, _, src) in enumerate(images):
        stem = f"{i:03d}"
        name = f"{stem}_mask.png" if mask_suffix else f"{stem}.png"
        shutil.copy2(src, out_dir / name)

if __name__ == "__main__":

    FABRIC_CODES = ["00", "01", "02", "03", "04", "05", "06", "08"]

    for fc in FABRIC_CODES:
        convert_fabric_dataset(
            source_dir="datasets/AITEX",
            dest_dir="datasets",
            dataset_name="AITEX_",
            category_name=f"fabric_{fc}",
            train_ratio=0.8,
            fabric_code=fc,
        )