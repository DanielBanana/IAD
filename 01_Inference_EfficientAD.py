# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Inference using the Python API.

This example shows how to perform inference on a trained model
using the Anomalib Python API.
"""

# 1. Import required modules
from pathlib import Path

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import EfficientAd
import glob
import os


# 2. Initialize the model and load weights
model = EfficientAd()
engine = Engine()

# 3. Prepare test data
# You can use a single image or a folder of image

def get_folder_paths(directory_path):
    # Use glob to get all items in the directory
    all_items = glob.glob(os.path.join(directory_path, '*'))

    # Filter out the directories
    folder_paths = [item for item in all_items if os.path.isdir(item)]

    return folder_paths

# Example usage
directory_path = "./datasets/MVTecAD/cable/test/"
folders = get_folder_paths(directory_path)
print(folders)

for folder in folders:
    print(f"Testing: {folder}")
    dataset = PredictDataset(
        path=Path(folder),
        image_size=(256, 256),
    )

    # 4. Get predictions
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path="results/checkpoints/best.ckpt",
    )

    # 5. Access the results
    if predictions is not None:
        for prediction in predictions:
            image_path = prediction.image_path
            anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
            pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
            pred_score = prediction.pred_score  # Image-level anomaly score