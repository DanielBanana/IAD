# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Inference using the Python API.

This example shows how to perform inference on a trained model
using the Anomalib Python API.
"""

# 1. Import required modules
from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.data import MVTecAD, BTech, Visa, Kolektor, Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Dsr, ReverseDistillation, Fastflow, Patchcore, Stfpm
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import F1Score, AUPR, AUROC, Evaluator
from anomalib.callbacks import ModelCheckpoint, GraphLogger, TimerCallback, TilerConfigurationCallback
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from anomalib.visualization import ImageVisualizer
from anomalib.visualization.image.item_visualizer import visualize_image_item
from anomalib.callbacks import LoadModelCallback
from anomalib.loggers import AnomalibTensorBoardLogger, AnomalibWandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar


import matplotlib.pyplot as plt
import numpy as np
import torchvision
import glob
import os
import yaml
import cv2
import argparse
from torchvision.transforms import Compose, Normalize, Resize

DATASETS = ["mvtecad", "kolektor", "visa", "btech"]     # TODO. "isp-ad", "wfdd", (not in anomalib)
CATEGORIES = {
    "mvtecad": ["all",
                "bottle",
                "cable",
                "capsule",
                "hazelnut",
                "metal_nut",
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
    parser.add_argument("--modelName", type=str, default="patchcore", help="Which Anomaly Detection Model to train")
    parser.add_argument("--checkpointVersion", type=int, default=0, help="Version of the training run, used for loading the checkpoint")
    parser.add_argument("--version", type=int, default=2, help="Version of the run, used for logging")

    # Parse the arguments
    args = parser.parse_args()
    
    dataset = args.dataset.lower()
    category = args.category.lower()
    modelName = args.modelName.lower()
    
    eval_batch_size = 1  # Set the batch size for evaluation, online inference
    
    if dataset in DATASETS:
        print(f"Dataset {dataset} found!")
        if dataset == "kolektor":
            print(f"INFO: {dataset} does not have categories")
            category = "none"
    else:
        print(f"Dataset {dataset} not found! \n Available models are: {', '.join(MODELS)}")
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
    
    # resultsDir = os.path.join("results", dataset)
    # prediction_path = os.path.join("results", dataset, category)

    logAndResultsDir = "logs"
    runName = f"{modelName}-{dataset}-{category}"
    checkpointDir = os.path.join(logAndResultsDir, runName, f"version_{args.checkpointVersion}", "checkpoints")
    
    # if not os.path.exists(prediction_path):
    #     os.makedirs(prediction_path)
    #     print(f"Directory created for predictions at: {prediction_path}")
    # else:
    #     print(f"Directory for predictions already exists at: {prediction_path}")
        
    # 3. Initialize the model
    # preProcessor = PreProcessor(transform = Compose([Resize((224, 224)), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    preProcessor = True
    
    checkpointFile ="best"
    fileExtension = ".pt"
    checkpointPath = os.path.join(checkpointDir, checkpointFile + fileExtension)

    if os.path.exists(checkpointPath):
        print(f"Found Checkpoint!")
    else:
        print(f"No checkpoint found at {checkpointPath}. Please train the model first or provide a valid checkpoint.")
        exit(1)
    
    checkpointCallback = ModelCheckpoint(
        dirpath=checkpointDir,
        filename=checkpointFile,
        monitor="train_loss",  # val_loss not found?
        verbose=True,
        save_top_k=1,  # Save only the best model
        mode="min",  # Save the model with the minimum training loss
    )
    
    checkpointCallback.FILE_EXTENSION = fileExtension  # Set the file extension for checkpoints
    
    graphCallback = GraphLogger()
    timerCallback = TimerCallback()
    progressBar = TQDMProgressBar(refresh_rate=50)
    
    callbacks = [progressBar, checkpointCallback, graphCallback, timerCallback]
    
    logger = AnomalibTensorBoardLogger(
        save_dir=logAndResultsDir,
        name=runName,
        version=args.version)
    
    # 3. Initialize the model
    # preProcessor = PreProcessor(transform = Compose([Resize((224, 224)), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
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
    
    # val metrics (needed for early stopping)
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    image_aupr = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_aupr = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    val_metrics = [image_auroc, pixel_auroc, image_aupr, pixel_aupr]

    # test_metrics
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    image_aupr = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_aupr = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_f1score, image_aupr, pixel_aupr]
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
        
    

    
    # 2. Create a dataset
    if dataset == "mvtecad":
        if category == "metal_nut":
            category = "metal nut"
        datasetPath = f"datasets/MVTecAD"
        print(f"Searching for dataset at: {datasetPath}")
        datamodule = MVTecAD(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # MVTecAD category to use
            eval_batch_size=eval_batch_size,  # Batch size for evaluation
            num_workers=8,  # Number of parallel processes for data loading
        )
    elif dataset == "visa":
        datasetPath = f"datasets/visa"
        print(f"Searching for dataset at: {datasetPath}")
        datamodule = Visa(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # Visa category to use
            eval_batch_size=eval_batch_size,  # Batch size for evaluation
            num_workers=8,  # Number of parallel processes for data loading
        )
    elif dataset == "kolektor":
        datasetPath = f"datasets/kolektor"
        print(f"Searching for dataset at: {datasetPath}")
        datamodule = Kolektor(
            root=datasetPath,  # Path to download/store the dataset
            eval_batch_size=eval_batch_size,  # Batch size for evaluation
            num_workers=8,  # Number of parallel processes for data loading
        )
    elif dataset == "btech":
        datasetPath = f"datasets/btech"
        print(f"Searching for dataset at: {datasetPath}")
        datamodule = BTech(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # BTech category to use
            eval_batch_size=eval_batch_size,  # Batch size for evaluation
        )
    elif dataset.lower() == "custom":
        datasetPath = f"datasets/custom"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = Folder(
            name="custom",
            root=datasetPath,  # Path to download/store the dataset
            normal_dir="simple/train/good",
            abnormal_dir="simple/test/writing",
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=8,  # Number of parallel processes for data loading
        )
        datamodule.setup()
        
    # 7. Test on test set
    res = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=checkpointPath
    )
    
    import time
    takenTime = time.time() - timerCallback.start
    throughput = timerCallback.num_images / takenTime
    print(f"Testing took {takenTime:.0f} seconds")
    print(f"Throughput (batchSize = {eval_batch_size}): {throughput:.2f} images/second")
    
    # 8. Predict on test set
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=checkpointPath
    )

    # 5. Access the results
    itemIdx = 0
    import numpy as np
    trueAnomalies = []
    predLabels = []
    if predictions is not None:
        for i, batch in enumerate(predictions):
            for j, prediction in enumerate(batch):
                trueAnomaly = datamodule.val_data.samples['label_index'][itemIdx]
                image_path = prediction.image_path
                anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
                predLabel = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
                trueAnomalies.append(trueAnomaly)
                predLabels.append(predLabel)
                itemIdx+=1
                pred_score = prediction.pred_score  # Image-level anomaly score
                print(f"Anomaly score: {pred_score}")
    trueAnomalies = np.asarray(trueAnomalies)
    predLabels = np.asarray(predLabels)
    confusionMatrix = confusion_matrix(trueAnomalies, predLabels)
    
    fig = plt.figure()
    ax = fig.subplots()
    
    CM_plot = ConfusionMatrixDisplay.from_predictions(trueAnomalies, predLabels, ax=ax)
    print(confusionMatrix)
    # CM_plot.figure_.savefig(os.path.join(prediction_path, f"{modelName}_confusion_matrix.png"))
    
    logger.add_image(CM_plot.figure_, "confusion_matrix", global_step=0)    
    
    tp = confusionMatrix[1][1]
    tn = confusionMatrix[0][0]
    fp = confusionMatrix[0][1]
    fn = confusionMatrix[1][0]
    
    positive = tp + fn
    negative = tn + fp
    tpr = tp / positive
    tnr = tn / negative
    fnr = fn / positive
    fpr = fp / negative
    f1_score = 2 * tp/(2*tp + fp + fn)
    
    res[0]["image_positive"] = int(positive)
    res[0]["image_negative"] = int(negative)
    res[0]["image_tp"] = int(tp)
    res[0]["image_tn"] = int(tn)
    res[0]["image_fp"] = int(fp)
    res[0]["image_fn"] = int(fn)
    res[0]["image_TPR"] = float(tpr)
    res[0]["image_TNR"] = float(tnr)
    res[0]["image_FNR"] = float(fnr)
    res[0]["image_FPR"] = float(fpr)
    res[0]["taken_time"] = takenTime
    res[0]["throughput"] = throughput
    
    logger.log_metrics(metrics={"image_positive": positive,
                            "image_negative": negative,
                            "image_tp": tp,
                            "image_tn": tn,
                            "image_fp": fp,
                            "image_fn": fn,
                            "image_TPR": tpr,
                            "image_TNR": tnr,
                            "image_FNR": fnr,
                            "image_FPR": fpr,
                            "taken_time": takenTime,
                            "throughput": throughput},
                    step=0)
        
    with open(os.path.join(logger.log_dir, "test_results.yaml"), "w") as file:
        yaml.dump(res, file, default_flow_style=False)
            
    print("Finished")