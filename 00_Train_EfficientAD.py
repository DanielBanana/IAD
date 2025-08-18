import sys
import argparse
import torch
import os
import time
import logging
import numpy as np
import yaml
import matplotlib.pyplot as plt

from anomalib.data import MVTecAD, BTech, Visa, Kolektor, Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Dsr, ReverseDistillation, Fastflow, Patchcore, Stfpm
from anomalib.callbacks import ModelCheckpoint, GraphLogger, TimerCallback, TilerConfigurationCallback
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import F1Score, AUPR, AUROC, Evaluator
from anomalib.visualization import ImageVisualizer
from anomalib.data.datasets.image.mvtecad import CATEGORIES
from anomalib.loggers import AnomalibTensorBoardLogger, AnomalibWandbLogger

from torchvision.transforms import Compose, Normalize, Resize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from lightning.pytorch.callbacks import TQDMProgressBar

from settings import DATASETS, CATEGORIES, MODELS, DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG
from setup import create_datamodule, create_model, setupTensorboardLoggingAndCallbacks, setupLogging, define_metrics
from results import createConfusionMatrixDisplay, logConfusionMatrix, logMetrics

# only for Cluster with NVIDIA L40S GPU
torch.set_float32_matmul_precision("high")          # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

def main(dataset, category, modelName, train_batch_size, eval_batch_size, num_workers, max_epochs, version):
   
    # 1. Set up the environment
    test_split_mode = "from_dir" # none, from_dir, synthetic, train_data
    test_split_ratio = 0.2
    val_split_mode = "same_as_test" # none, same_as_text, from_train, from_test, synthetic (from train_data)
    val_split_ratio = 0.5 # not used if same_as_text

    tblogger, callbacks, ckptPath = setupTensorboardLoggingAndCallbacks(version)
    logger = setupLogging(version, modelName, dataset, category)
    
    # 2. Setup Dataprocessing
    logger.info(f"Setting up pre-processing for model {modelName}")
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
                                  image_sensitivity=0.01,
                                  pixel_sensitivity=0.01)
    #postProcessor = True
    val_metrics, test_metrics = define_metrics()
    
    evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
    
    engine = Engine(
        max_epochs=max_epochs,
        default_root_dir='results_training',
        callbacks=list(callbacks),
        logger=tblogger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )
    
    # 3. Setup Datamodule
    logger.info(f"Creating datamodule for dataset {dataset} and category {category}")
    datamodule = create_datamodule(dataset, category, train_batch_size, eval_batch_size, num_workers, test_split_mode, test_split_ratio, val_split_mode, val_split_ratio)
    
    # 4. Setup model
    logger.info(f"Creating model {modelName} for dataset {dataset} and category {category}")
    model = create_model(modelName, preProcessor, postProcessor, visualizer, evaluator)

    # 5. Train the model
    logger.info("Starting training...")
    engine.fit(datamodule=datamodule, model=model)
    
    resultsPath = engine._cache.args["default_root_dir"]
    
    # 6. Validate on validation set. adjust thresholds
    logger.info("Validating on validation set...")
    runValidation = 'Yes' if engine._should_run_validation(engine.model, None) else 'No'
    
    logger.log(f"Should we run validation: {runValidation}")
    
    if engine._should_run_validation(engine.model, None):
        engine.validate(
            model=model,
            datamodule=datamodule,
            # ckpt_path=checkpointPath
        )

    # 7. Test on test set
    logger.info("Testing on test set...")
    res = engine.test(
        model=model,
        datamodule=datamodule,
        # ckpt_path=checkpointPath
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
        # ckpt_path=checkpointPath
    )
    
    # 9. Calculate confusion matrix and other metrics
    logger.info("Calculating confusion matrix and other metrics...")
    confusionMatrix, trueAnomalies, predLabels = createConfusionMatrixDisplay(predictions, datamodule)
    
    logConfusionMatrix(logger, tblogger, confusionMatrix, trueAnomalies, predLabels)
    
    logMetrics(tblogger, res, confusionMatrix, takenTime, throughput)        
    with open(os.path.join(tblogger.log_dir, "train_results.yaml"), "w") as file:
        yaml.dump(res, file, default_flow_style=False)
            
    logger.info("Finished")

    # engine.export(model)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train an anomaly detection model.")
    parser.add_argument("--dataset", type=str, default="kolektor", help="Which dataset to train on")
    parser.add_argument("--category", type=str, default="none", help="Which category to train on")
    parser.add_argument("--modelName", type=str, default="patchcore", help="Which method to train")
    
    parser.add_argument("--train_batch_size", type=int, default=1, help="Number of images per training batch")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Number of images per validation/test batch")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel processes for data loading")
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of epochs to train the model")
    parser.add_argument("--version", type=int, default=0, help="Version of the run, used for logging")

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
        
        

    # Call the main function with parsed arguments
    # dataset, category, model_name, train_batch_size, eval_batch_size, num_workers, max_epochs
    main(dataset, category, modelName, args.train_batch_size, args.eval_batch_size, args.num_workers, args.max_epochs, args.version)
