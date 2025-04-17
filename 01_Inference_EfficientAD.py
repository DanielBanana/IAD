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
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import F1Score, AUPR, AUROC, Evaluator
from anomalib.callbacks import ModelCheckpoint

import torchvision
import glob
import os
import cv2
from anomalib.visualization import ImageVisualizer
from anomalib.callbacks import LoadModelCallback
import argparse
from torchvision.transforms import Compose, Normalize, Resize

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

DEFAULT_FIELDS_CONFIG = {
    "image": {},
    "gt_mask": {},
    "pred_mask": {},
    "anomaly_map": {"colormap": True, "normalize": False},
}

DEFAULT_OVERLAY_FIELDS_CONFIG = {
    "gt_mask": {"color": (255, 255, 255), "alpha": 1.0, "mode": "contour"},
    "pred_mask": {"color": (255, 0, 0), "alpha": 1.0, "mode": "contour"},
}

DEFAULT_TEXT_CONFIG = {
    "enable": True,
    "font": None,
    "size": None,
    "color": "white",
    "background": (0, 0, 0, 128),
}


def get_folder_paths(directory_path):
    # Use glob to get all items in the directory
    all_items = glob.glob(os.path.join(directory_path, '*'))

    # Filter out the directories
    folder_paths = [item for item in all_items if os.path.isdir(item)]

    return folder_paths

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train an anomaly detection model.")
    parser.add_argument("--dataset", type=str, default="kolektor", help="Which dataset to train on")
    parser.add_argument("--category", type=str, default="none", help="Which category of the dataset to train on")
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
    # preProcessor = PreProcessor(transform = Compose([Resize((224, 224)), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    preProcessor = True
    
    visualizer = ImageVisualizer(output_dir=prediction_path,
                                 fields=["image", "gt_mask"],
                                 overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],
                                 field_size=(256,256),
                                 fields_config=DEFAULT_FIELDS_CONFIG,
                                 overlay_fields_config=DEFAULT_OVERLAY_FIELDS_CONFIG,
                                 text_config=DEFAULT_TEXT_CONFIG)
    
    postProcessor = PostProcessor(enable_normalization=True,
                                  enable_threshold_matching=True,
                                  enable_thresholding=True,
                                  image_sensitivity=0.01,
                                  pixel_sensitivity=0.01)
    postProcessor=True
    f1_score = F1Score(fields=["pred_label", "gt_label"])
    auroc = AUROC(fields=["pred_score", "gt_label"])
    aupr = AUPR(fields=["pred_score", "gt_label"])
    # evaluator = Evaluator(val_metrics=[f1_score, auroc, aupr], test_metrics=[f1_score, auroc, aupr])
    evaluator = Evaluator(test_metrics=[f1_score, auroc, aupr])
    evaluator=True
    
    if modelName == "efficientad-s":
        # model = EfficientAd(visualizer=visualizer, model_size="small", post_processor=postProcessor)
        model = EfficientAd(pre_processor=preProcessor,
                            post_processor=postProcessor,
                            visualizer=visualizer,
                            evaluator=evaluator,
                            model_size="small")

    elif modelName == "efficientad-m":
        model = EfficientAd(pre_processor=preProcessor,
                            post_processor=postProcessor,
                            visualizer=visualizer,
                            evaluator=evaluator,
                            model_size="medium")
    elif modelName == "dsr":
        model = Dsr(pre_processor=preProcessor,
                    post_processor=postProcessor,
                    visualizer=visualizer,
                    evaluator=evaluator)
    elif modelName == "reversedistillation":
        model = ReverseDistillation(pre_processor=preProcessor,
                                    post_processor=postProcessor,
                                    visualizer=visualizer,
                                    evaluator=evaluator)
    elif modelName == "reverse_distillation":
        model = ReverseDistillation(pre_processor=preProcessor,
                                    post_processor=postProcessor,
                                    visualizer=visualizer,
                                    evaluator=evaluator)
    elif modelName == "rd":
        model = ReverseDistillation(pre_processor=preProcessor,
                                    post_processor=postProcessor,
                                    visualizer=visualizer,
                                    evaluator=evaluator)
    elif modelName == "stfpm":
        model = Stfpm(pre_processor=preProcessor,
                      post_processor=postProcessor,
                      visualizer=visualizer,
                      evaluator=evaluator)
    elif modelName == "fastflow":
        model = Fastflow(pre_processor=preProcessor,
                         post_processor=postProcessor,
                         visualizer=visualizer,
                         evaluator=evaluator)
    elif modelName == "fast_flow":
        model = Fastflow(pre_processor=preProcessor,
                         post_processor=postProcessor,
                         visualizer=visualizer,
                         evaluator=evaluator)
    elif modelName == "patchcore":
        model = Patchcore(pre_processor=preProcessor,
                          post_processor=postProcessor,
                          visualizer=visualizer,
                          evaluator=evaluator)

    checkpointDir = "results/checkpoints"

    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)
        print(f"Directory created: {checkpointDir}")
    else:
        print(f"Directory already exists: {checkpointDir}")
    
    checkpointDatasetFolder = os.path.join(checkpointDir, dataset.lower())
    
    if not os.path.exists(checkpointDatasetFolder):
        os.makedirs(checkpointDatasetFolder)
        print(f"Directory created: {checkpointDatasetFolder}")
    else:
        print(f"Directory already exists: {checkpointDatasetFolder}")
        
    
    checkpointFile = f'{modelName}_{category}_best'
    
    checkpointPath = os.path.join(checkpointDatasetFolder, checkpointFile+'.ckpt')

    if os.path.exists(checkpointPath):
        print(f"Found Checkpoint!")
    else:
        print(f"No checkpoint found!")
        exit(1)

    # 2. Initialize the model and load weights
    checkpointCallback = ModelCheckpoint(
        dirpath=checkpointDatasetFolder,
        filename=checkpointFile,
        monitor="train_loss",  # val_loss not found?
        verbose=True,
    )
    
    engine = Engine(
        max_epochs=1,
        default_root_dir='results',
        callbacks=[checkpointCallback],
        accelerator="cpu",
        devices=1
    )
    
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
        
    # predictions = engine.validate(
    #     model=model,
    #     datamodule=datamodule,
    #     ckpt_path=checkpointPath
    # )
    
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=checkpointPath
    )

    # 5. Access the results
    iO = 0
    niO = 0
    if predictions is not None:
        for i, batch in enumerate(predictions):
            for prediction in batch:
                image_path = prediction.image_path
                anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
                pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
                if pred_label:
                    niO+=1
                    print(f"Predicted image {i} to be anomalous.")
                else:
                    iO+=1
                    print(f"Predicted image {i} to be normal.")
                pred_score = prediction.pred_score  # Image-level anomaly score
                print(f"Anomaly score: {pred_score}")
    print(f"Number of predicted anomalous samples: {niO}")
    print(f"Number of predicted normal samples: {iO}")
            
            # torchvision.utils.save_image(prediction.image, os.path.join(prediction_path, f"cable_image_{i}.png"))
            # torchvision.utils.save_image(prediction.anomaly_map, os.path.join(prediction_path, f"cable_anomaly_map_{i}.png"))
            
    print("Finished")