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

from settings import *

# only for Cluster with NVIDIA L40S GPU
torch.set_float32_matmul_precision("high")          # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

def define_metrics():
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
    
    return val_metrics, test_metrics

def create_datamodule(dataset, category, train_batch_size, eval_batch_size, num_workers, test_split_mode, test_split_ratio, val_split_mode, val_split_ratio):
    # 2. Create a dataset
    if dataset.lower() == "mvtecad":
        if category == "metal_nut":
            category = "metal nut"
        datasetPath = f"datasets/MVTecAD"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = MVTecAD(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # MVTecAD category to use
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio
        )
    elif dataset.lower() == "visa":
        datasetPath = f"datasets/visa"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = Visa(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # Visa category to use
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio
        )
    elif dataset.lower() == "kolektor":
        datasetPath = f"datasets/kolektor"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = Kolektor(
            root=datasetPath,  # Path to download/store the dataset
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio
        )
    elif dataset.lower() == "btech":
        datasetPath = f"datasets/btech"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = BTech(
            root=datasetPath,  # Path to download/store the dataset
            category=category,  # BTech category to use
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio
        )
    elif dataset.lower() == "custom":
        datasetPath = f"datasets/custom"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = Folder(
            name="custom",
            root=datasetPath,  # Path to download/store the dataset
            normal_dir="simple/train/good",
            abnormal_dir="simple/test/writing",
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio
        )
        datamodule.setup()
        
    return datamodule

def create_model(modelName, preProcessor, postProcessor, visualizer, evaluator):
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
    return model

def setupTensorboardLoggingAndCallbacks(version, modelName, dataset, category):
    logAndResultsDir = "logs"
    runName = f"{modelName}-{dataset}-{category}"
    checkpointDir = os.path.join(logAndResultsDir, runName, f"version_{version}", "checkpoints")
    checkpointCallback = ModelCheckpoint(
        dirpath=checkpointDir,
        filename="best",
        monitor="train_loss",  # val_loss not found?
        verbose=True,
        save_top_k=1,  # Save only the best model
        mode="min",  # Save the model with the minimum training loss
    )
    
    checkpointCallback.FILE_EXTENSION = ".pt"  # Set the file extension for checkpoints
    
    graphCallback = GraphLogger()
    timerCallback = TimerCallback()
    progressBar = TQDMProgressBar(refresh_rate=50)
    
    callbacks = {
        "progress_bar": progressBar,
        "checkpoint": checkpointCallback,
        "graph": graphCallback,
        "timer": timerCallback
    }
    
    tblogger = AnomalibTensorBoardLogger(
        save_dir=logAndResultsDir,
        name=runName,
        version=version)
    
    return tblogger, callbacks

def setupLogging(version, modelName, dataset, category):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join("logs", f"{modelName}-{dataset}-{category}-v{version}.log"))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    return logger

def createConfusionMatrixDisplay(predictions, datamodule):
    itemIdx = 0
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
    return confusionMatrix, trueAnomalies, predLabels
    
def logConfusionMatrix(logger, tblogger, confusionMatrix, trueAnomalies, predLabels):
    fig = plt.figure()
    ax = fig.subplots()
    
    CM_plot = ConfusionMatrixDisplay.from_predictions(trueAnomalies, predLabels, ax=ax)
    logger.info("Confusion Matrix:")
    logger.info(confusionMatrix)
    # CM_plot.figure_.savefig(os.path.join(prediction_path, f"{modelName}_confusion_matrix.png"))
    
    tblogger.add_image(CM_plot.figure_, "confusion_matrix", global_step=0)
    
def logMetrics(tblogger, res, confusionMatrix, takenTime, throughput):
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
    
    tblogger.log_metrics(metrics={"image_positive": positive,
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



def main(dataset, category, modelName, train_batch_size, eval_batch_size, num_workers, max_epochs, version):
   
    # 1. Set up the environment
    test_split_mode = "from_dir" # none, from_dir, synthetic, train_data
    test_split_ratio = 0.2
    val_split_mode = "same_as_test" # none, same_as_text, from_train, from_test, synthetic (from train_data)
    val_split_ratio = 0.5 # not used if same_as_text

    tblogger, callbacks = setupTensorboardLoggingAndCallbacks(version)
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
