# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Inference using the Python API.

This example shows how to perform inference on a trained model
using the Anomalib Python API.
"""

# 1. Import required modules
from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.data import MVTecAD, BTech, Visa, Kolektor
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Dsr, ReverseDistillation, Fastflow, Patchcore, Stfpm
import torchvision
import glob
import os
import cv2
from anomalib.visualization import ImageVisualizer
from anomalib.callbacks import LoadModelCallback
import argparse

DATASETS = ["mvtecad", "kolektor", "visa", "btech"]     # TODO. "isp-ad", "wfdd", (not in anomalib)
CATEGORIES = {
    "mvtecad": ["all",
                "bottle",
                "cable",
                "capsule",
                "hazelnut",
                "metal nut",
                "pill",
                "screw",
                "toothbrush",
                "transistor",
                "zipper",
                "carpet",
                "grid",
                "leather",
                "tile",
                "wood"],
    "kolektor": ["none"],
    "visa": ["candle",
            "capsules",
            "cashew",
            "chewinggum",
            "fryum",
            "macaroni1",
            "macaroni2",
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4",
            "pipe_fryum"],
    "btech": ["01",
              "02",
              "03"]}
MODELS = ["efficientad-s", "efficientad-m", "patchcore", "fastflow", "dsr", "reverse_distillation/rd", "stfpm"]     # TODO GLASS(not in anomalib)

def get_folder_paths(directory_path):
    # Use glob to get all items in the directory
    all_items = glob.glob(os.path.join(directory_path, '*'))

    # Filter out the directories
    folder_paths = [item for item in all_items if os.path.isdir(item)]

    return folder_paths

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train an anomaly detection model.")
    parser.add_argument("--dataset", type=str, default="mvtecad", help="Which dataset to train on")
    parser.add_argument("--category", type=str, default="leather", help="Which category of the dataset to train on")
    parser.add_argument("--modelName", type=str, default="EfficientAD-S", help="Which Anomaly Detection Model to train")

    # Parse the arguments
    args = parser.parse_args()
    
    dataset = args.dataset.lower()
    category = args.category.lower()
    modelName = args.modelName.lower()
    
    if dataset in DATASETS:
        print(f"Dataset {dataset} found!")
        if dataset == "kolektor":
            print(f"INFO: {dataset} does not have categories")
            category = "none"
    else:
        print(f"Dataset {args.dataset} not found! \n Available models are: {', '.join(MODELS)}")
        exit(1)
        
    if category in CATEGORIES[dataset]:
        print(f"Category {category} found!")
    else:
        print(f"Category {category} not found! \n Available categories are: {', '.join(CATEGORIES[dataset])}")
        exit(1)
        
    if modelName in MODELS:
        print(f"Model {modelName} found!")
    else:
        print(f"Model {modelName} not found! \n Available models are: {', '.join(MODELS)}")
        exit(1)

    print("#############################################################################################")
    print("#############################################################################################")
    print("######### STARTING INFERENCE ################################################################")
    print("#############################################################################################")
    print("#############################################################################################")
    
    prediction_path = f"results/{dataset}/{category}/predictions"
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
        print(f"Directory created for predictions at: {prediction_path}")
    else:
        print(f"Directory for predictions already exists at: {prediction_path}")
        
    # 3. Initialize the model
    visualizer = ImageVisualizer(output_dir=prediction_path)
    if modelName == "efficientad-s":
        model = EfficientAd(visualizer=visualizer, model_size="small")
    elif modelName == "efficientad-m":
        model = EfficientAd(visualizer=visualizer, model_size="medium")
    elif modelName == "dsr":
        model = Dsr(visualizer=visualizer)
    elif modelName == "reversedistillation":
        model = ReverseDistillation(visualizer=visualizer)
    elif modelName == "reverse_distillation":
        model = ReverseDistillation(visualizer=visualizer)
    elif modelName == "rd":
        model = ReverseDistillation(visualizer=visualizer)
    elif modelName == "stfpm":
        model = Stfpm(visualizer=visualizer)
    elif modelName == "fastflow":
        model = Fastflow(visualizer=visualizer)
    elif modelName == "fast_flow":
        model = Fastflow(visualizer=visualizer)
    elif modelName == "patchcore":
        model = Patchcore(visualizer=visualizer)
    
    checkpointDir = "results/checkpoints"
    checkpointFile = f'{modelName}_{category}_best.ckpt'
    checkpointPath = os.path.join(checkpointDir, dataset ,checkpointFile)
    
    print(f"Searching for checkpoint file at: {checkpointPath}")

    if os.path.exists(checkpointPath):
        print(f"Found Checkpoint!")
    else:
        print(f"No checkpoint found!")
        exit(1)


    # 2. Initialize the model and load weights
    #engine = Engine(callbacks=callbacks)
    engine = Engine()
        # 2. Create a dataset
    if dataset == "mvtecad":
        datasetPath = f"datasets/MVTecAD"
        print(f"Searching for dataset at: {datasetPath}")
        datamodule = MVTecAD(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # MVTecAD category to use
        )
    elif dataset == "visa":
        datasetPath = f"datasets/visa"
        print(f"Searching for dataset at: {datasetPath}")
        datamodule = Visa(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # Visa category to use
        )
    elif dataset == "kolektor":
        datasetPath = f"datasets/kolektor"
        print(f"Searching for dataset at: {datasetPath}")
        datamodule = Kolektor(
            root=datasetPath,  # Path to download/store the dataset
        )
    elif dataset == "btech":
        datasetPath = f"datasets/btech"
        print(f"Searching for dataset at: {datasetPath}")
        datamodule = BTech(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # BTech category to use
        )

    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=checkpointPath
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