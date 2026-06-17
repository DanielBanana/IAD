import argparse
import torch
import os
import time
import yaml
import matplotlib.pyplot as plt

from anomalib.engine import Engine

from anomalib.post_processing import PostProcessor
from anomalib.metrics import Evaluator
from anomalib.visualization import ImageVisualizer
from anomalib.data.datasets.image.mvtecad import CATEGORIES


from src.settings import DATASETS, CATEGORIES, MODELS, DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG
from src.setup import create_datamodule, create_model, setupTensorboardLoggingAndCallbacks, setupLogging, define_metrics
from playground.results import createConfusionMatrixDisplay, logConfusionMatrix, logMetrics

# only for Cluster with NVIDIA L40S GPU
torch.set_float32_matmul_precision("high")          # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train an anomaly detection model.")
    parser.add_argument("--dataset", type=str, default="MVTecAD", help="Which dataset to train on")
    parser.add_argument("--category", type=str, default="tile", help="Which category to train on")
    parser.add_argument("--modelName", type=str, default="patchcore", help="Which method to train")
    
    parser.add_argument("--train_batch_size", type=int, default=1, help="Number of images per training batch")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Number of images per validation/test batch")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel processes for data loading")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--version", type=int, default=1, help="Version of the run, used for logging")

    # Parse the arguments
    args = parser.parse_args()
    
    dataset = args.dataset.lower()
    category = args.category.lower()
    modelName = args.modelName.lower()
    version = args.version
    max_epochs = args.max_epochs
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    num_workers = args.num_workers
    
    resultsPath = "results_training"
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
    logDir = "logs"
    runName = f"{modelName}-{dataset}-{category}"
    versionName = f"version_{version}"
    if not os.path.exists(os.path.join(logDir, runName, versionName)):
        os.makedirs(os.path.join(logDir, runName, versionName))
  
    tblogger, callbacks, ckptPath = setupTensorboardLoggingAndCallbacks(logDir=logDir, runName=runName, versionName=versionName, version=version)
    logger = setupLogging(logDir=logDir, runName=runName, versionName=versionName)

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
                
    # 1. Set up the environment
    test_split_mode = "from_dir" # none, from_dir, synthetic, train_data
    test_split_ratio = 0.2
    val_split_mode = "same_as_test" # none, same_as_text, from_train, from_test, synthetic (from train_data)
    val_split_ratio = 0.5 # not used if same_as_text

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
        default_root_dir=resultsPath,
        callbacks=list(callbacks.values()),
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
    print("Training finished")
    
    resultsPath = engine._cache.args["default_root_dir"]
    
    # 6. Validate on validation set. adjust thresholds
    logger.info("Validating on validation set...")
    runValidation = 'Yes' if engine._should_run_validation(engine.model, None) else 'No'
    
    logger.info(f"Should we run validation: {runValidation}")
    
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
    confusionMatrix, trueAnomalies, predLabels = createConfusionMatrixDisplay(logger, predictions, datamodule)
    
    logConfusionMatrix(logger, tblogger, confusionMatrix, trueAnomalies, predLabels)
    
    logMetrics(tblogger, res, confusionMatrix, takenTime, throughput)        
    with open(os.path.join(tblogger.log_dir, "train_results.yaml"), "w") as file:
        yaml.dump(res, file, default_flow_style=False)
            
    logger.info("Finished")
