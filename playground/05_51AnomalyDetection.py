import fiftyone as fo # base library and app
from fiftyone import ViewField as F # helper for defining views
# 1. Import required modules
from anomalib.data import MVTecAD
from anomalib import TaskType
from anomalib.data import Folder
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine

import numpy as np
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms.v2 import Resize

OBJECT = "bottle" ## object to train on
ROOT_DIR = Path("datasets/MVTecAD") ## root directory to store data for anomalib
TASK = TaskType.SEGMENTATION ## task type for the model
IMAGE_SIZE = (256, 256) ## preprocess image size for uniformity

# 1. Import the MVTecAD dataset as a FiftyOne dataset
dataset = fo.Dataset('MVTecAD')



# def create_datamodule(object_type, dataset, transform=None):
#     ## Build transform
#     if transform is None:
#         transform = Resize(IMAGE_SIZE, antialias=True)

#     normal_data = dataset.match(F("category.label") == object_type).match(
#         F("split") == "train"
#     )
#     abnormal_data = (
#         dataset.match(F("category.label") == object_type)
#         .match(F("split") == "test")
#         .match(F("defect.label") != "good")
#     )

#     normal_dir = Path(ROOT_DIR) / object_type / "normal"
#     abnormal_dir = ROOT_DIR / object_type / "abnormal"
#     mask_dir = ROOT_DIR / object_type / "mask"

#     # create directories if they do not exist
#     os.makedirs(normal_dir, exist_ok=True)
#     os.makedirs(abnormal_dir, exist_ok=True)
#     os.makedirs(mask_dir, exist_ok=True)

#     if not os.path.exists(str(normal_dir)):
#         normal_data.export(
#             export_dir=str(normal_dir),
#             dataset_type=fo.types.ImageDirectory,
#             export_media="symlink",
#         )

#     for sample in abnormal_data.iter_samples():
#         base_filename = sample.filename
#         dir_name = os.path.dirname(sample.filepath).split("/")[-1]
#         new_filename = f"{dir_name}_{base_filename}"
#         if not os.path.exists(str(abnormal_dir / new_filename)):
#             os.symlink(sample.filepath, str(abnormal_dir / new_filename))

#         if not os.path.exists(str(mask_dir / new_filename)):
#             os.symlink(sample.defect_mask.mask_path, str(mask_dir / new_filename))

#     datamodule = Folder(
#         name=object_type,
#         root=ROOT_DIR,
#         normal_dir=normal_dir,
#         abnormal_dir=abnormal_dir,
#         mask_dir=mask_dir,
#         task=TASK,
#         transform=transform
#     )
#     datamodule.setup()
#     return datamodule
