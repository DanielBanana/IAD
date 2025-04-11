# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Inference using the Python API.

This example shows how to perform inference on a trained model
using the Anomalib Python API.
"""

# 1. Import required modules
from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.data.datamodules import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
import torchvision
import glob
import os
import cv2
from anomalib.visualization import ImageVisualizer
from anomalib.callbacks import LoadModelCallback

def get_folder_paths(directory_path):
    # Use glob to get all items in the directory
    all_items = glob.glob(os.path.join(directory_path, '*'))

    # Filter out the directories
    folder_paths = [item for item in all_items if os.path.isdir(item)]

    return folder_paths

print("#############################################################################################")
print("#############################################################################################")
print("######### STARTING INFERENCE ################################################################")
print("#############################################################################################")
print("#############################################################################################")

print(f"Current Directory: {os.getcwd()}")
print(f"Folder in direcotry {os.getcwd()}: {get_folder_paths(os.getcwd())}")
prediction_path = "./results/MVTecAD/cable/predictions"
#callbacks = [LoadModelCallback(weights_path="results/checkpoints/EfficientAD_best.ckpt")]

# 2. Initialize the model and load weights
visualizer = ImageVisualizer(output_dir=prediction_path)
model = EfficientAd(visualizer=visualizer, model_size="small")
#engine = Engine(callbacks=callbacks)
engine = Engine()
category = "leather"
datamodule = MVTecAD("./datasets/MVTecAD", category=category)

# 3. Prepare test data
# You can use a single image or a folder of image

# Example usage


# for folder in folders:
#     print("#############################################################################################")
#     print(f"######### Testing: {folder}")
#     print("#############################################################################################")
    
#     test_tag = folder.split(os.sep)[-1]
    
#     dataset = PredictDataset(
#         path=Path(folder),
#         image_size=(256, 256),
#     )

# 4. Get predictions
# The path where the images are stored comes in part from the optional parameter "datamodule"
# datamodule is a alternative to dataset, it comes from the pytorch lightning library
# datamodule should have a variable named "name" and "category"
# predictions = engine.predict(
#     model=model,
#     dataset=dataset
# )

predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=f"results/checkpoints/EfficientAD_{category}_best.ckpt"
)

# 5. Access the results
if predictions is not None:
    for i, prediction in enumerate(predictions):
        
        image_path = prediction.image_path
        anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
        pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
        pred_score = prediction.pred_score  # Image-level anomaly score
        
        # torchvision.utils.save_image(prediction.image, os.path.join(prediction_path, f"cable_image_{i}.png"))
        # torchvision.utils.save_image(prediction.anomaly_map, os.path.join(prediction_path, f"cable_anomaly_map_{i}.png"))
        
print("Finished")