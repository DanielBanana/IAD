# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Inference using the Python API.

This example shows how to perform inference on a trained model
using the Anomalib Python API.
"""

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import glob
import os
import yaml
import cv2
import argparse

# 1. Import required modules
from pathlib import Path
from time import time
from anomalib.data import PredictDataset
from anomalib.data import MVTecAD, BTech, Visa, Kolektor, Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Dsr, ReverseDistillation, Fastflow, Patchcore, Stfpm
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import F1Score, AUPR, AUROC, Evaluator
from anomalib.callbacks import ModelCheckpoint, GraphLogger, TimerCallback, TilerConfigurationCallback
from anomalib.visualization import ImageVisualizer
from anomalib.visualization.image.item_visualizer import visualize_image_item
from anomalib.callbacks import LoadModelCallback
from anomalib.loggers import AnomalibTensorBoardLogger, AnomalibWandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torchvision.transforms import Compose, Normalize, Resize

from settings import DATASETS, CATEGORIES, MODELS, DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG
from setup import create_datamodule, create_model, setupTensorboardLoggingAndCallbacks, setupLogging, define_metrics
from results import createConfusionMatrixDisplay, logConfusionMatrix, logMetrics


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
    parser.add_argument("--modelName", type=str, default="patchcore", help="Which Anomaly Detection Model to train")
    parser.add_argument("--checkpointVersion", type=int, default=0, help="Version of the training run, used for loading the checkpoint")
    parser.add_argument("--version", type=int, default=2, help="Version of the run, used for logging")

    # Parse the arguments
    args = parser.parse_args()
    
    dataset = args.dataset.lower()
    category = args.category.lower()
    modelName = args.modelName.lower()
    version = args.version
    
    eval_batch_size = 1  # Set the batch size for evaluation, online inference
    
    logger = setupLogging(version, modelName, dataset, category)

    
    if dataset in DATASETS:
        logger.info(f"Dataset {dataset} found!")
        if dataset == "kolektor":
            logger.warning(f"INFO: {dataset} does not have categories")
            category = "none"
    else:
        logger.error(f"Dataset {dataset} not found! \n Available models are: {', '.join(MODELS)}")
        exit(1)
        
    if category in CATEGORIES[dataset]:
        logger.info(f"Category {category} found!")
    else:
        logger.error(f"Category {category} not found! \n Available categories are: {', '.join(CATEGORIES[dataset])}")
        exit(1)
        
    if modelName in MODELS:
        logger.info(f"Model {modelName} found!")
    else:
        logger.error(f"Model {modelName} not found! \n Available models are: {', '.join(MODELS)}")
        exit(1)
        
    

    logger.info("##### STARTING INFERENCE ################################################################")

    
    # resultsDir = os.path.join("results", dataset)
    # prediction_path = os.path.join("results", dataset, category)

    # logAndResultsDir = "logs"
    # runName = f"{modelName}-{dataset}-{category}"
    # checkpointDir = os.path.join(logAndResultsDir, runName, f"version_{args.checkpointVersion}", "checkpoints")
    
    # if not os.path.exists(prediction_path):
    #     os.makedirs(prediction_path)
    #     print(f"Directory created for predictions at: {prediction_path}")
    # else:
    #     print(f"Directory for predictions already exists at: {prediction_path}")
    
    # checkpointFile ="best"
    # fileExtension = ".pt"
    # checkpointPath = os.path.join(checkpointDir, checkpointFile + fileExtension)

    # if os.path.exists(checkpointPath):
    #     print(f"Found Checkpoint!")
    # else:
    #     print(f"No checkpoint found at {checkpointPath}. Please train the model first or provide a valid checkpoint.")
    #     exit(1)
    
    # checkpointCallback = ModelCheckpoint(
    #     dirpath=checkpointDir,
    #     filename=checkpointFile,
    #     monitor="train_loss",  # val_loss not found?
    #     verbose=True,
    #     save_top_k=1,  # Save only the best model
    #     mode="min",  # Save the model with the minimum training loss
    # )
    
    # checkpointCallback.FILE_EXTENSION = fileExtension  # Set the file extension for checkpoints
    
    # graphCallback = GraphLogger()
    # timerCallback = TimerCallback()
    # progressBar = TQDMProgressBar(refresh_rate=50)
    
    # callbacks = [progressBar, checkpointCallback, graphCallback, timerCallback]
    
    # logger = AnomalibTensorBoardLogger(
    #     save_dir=logAndResultsDir,
    #     name=runName,
    #     version=args.version)
    
    # 3. Initialize the model
    # preProcessor = PreProcessor(transform = Compose([Resize((224, 224)), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    
    tblogger, callbacks, ckptPath = setupTensorboardLoggingAndCallbacks(version)
    
    val_metrics, test_metrics = define_metrics()
    
    preProcessor = True
    
    visualizer = ImageVisualizer(# output_dir=prediction_path,
                                 fields=["image", "gt_mask"],
                                 overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],
                                 field_size=(256,256),
                                 fields_config=DEFAULT_FIELDS_CONFIG,
                                 overlay_fields_config=DEFAULT_OVERLAY_FIELDS_CONFIG,
                                 text_config=DEFAULT_TEXT_CONFIG)
    
    # visualizer = True
    postProcessor = PostProcessor(enable_normalization=True,
                                  enable_threshold_matching=True,
                                  enable_thresholding=True,
                                  image_sensitivity=0.5,
                                  pixel_sensitivity=0.5)
    
    evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
    
    engine = Engine(
        max_epochs=1,
        default_root_dir='results_testing',
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )
    
    model = create_model(modelName, preProcessor, postProcessor, visualizer, evaluator)
    
    datamodule = create_datamodule(dataset, category, eval_batch_size, eval_batch_size, 8, "from_dir", 0.2, "same_as_test", 0.5)

    # 7. Test on test set
    res = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckptPath
    )
    
    # Log throughput and time taken for testing
    takenTime = time.time() - callbacks["timer"].start
    throughput = callbacks["timer"].num_images / takenTime
    logger.info(f"Testing took {takenTime:.0f} seconds")
    logger.info(f"Throughput (batchSize = {eval_batch_size}): {throughput:.2f} images/second")
    
    # 8. Predict on test set
    logger.info("Predicting on test set...")
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckptPath
    )
    
    # 9. Calculate confusion matrix and other metrics
    logger.info("Calculating confusion matrix and other metrics...")
    confusionMatrix, trueAnomalies, predLabels = createConfusionMatrixDisplay(predictions, datamodule)
    
    logConfusionMatrix(logger, tblogger, confusionMatrix, trueAnomalies, predLabels)
    
    logMetrics(tblogger, res, confusionMatrix, takenTime, throughput)
    
    with open(os.path.join(tblogger.log_dir, "train_results.yaml"), "w") as file:
        yaml.dump(res, file, default_flow_style=False)
            
    logger.info("Finished")
        
    with open(os.path.join(logger.log_dir, "test_results.yaml"), "w") as file:
        yaml.dump(res, file, default_flow_style=False)
            
    print("Finished")