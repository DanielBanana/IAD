import sys
import argparse
from anomalib.data import MVTecAD, BTech, Visa, Kolektor, Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Dsr, ReverseDistillation, Fastflow, Patchcore, Stfpm
from anomalib.callbacks import ModelCheckpoint, GraphLogger, TimerCallback, TilerConfigurationCallback
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import F1Score, AUPR, AUROC, Evaluator
from anomalib.visualization import ImageVisualizer
from torchvision.transforms import Compose, Normalize, Resize
from anomalib.loggers import AnomalibTensorBoardLogger, AnomalibWandbLogger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from anomalib.data.datasets.image.mvtecad import CATEGORIES
import torch
import os


DATASETS = ["custom", "mvtecad", "kolektor", "visa", "btech"]     # TODO. "isp-ad", "wfdd", (not in anomalib)
CATEGORIES = {
    "custom" : ["simple"],
    "mvtecad": ["bottle",
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
MODELS = ["efficientad-s", "efficientad-m", "patchcore", "fastflow", "dsr", "reverse_distillation", "rd", "stfpm"]     # TODO GLASS(not in anomalib)

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

# only for Cluster with NVIDIA L40S GPU
torch.set_float32_matmul_precision("high")          # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

def main(dataset, category, model_name, train_batch_size, eval_batch_size, num_workers, max_epochs):
    # 0. Add anomalib folder to path (if necessary)
    # sys.path.append("anomalib")
    
    test_split_mode = "from_dir" # none, from_dir, synthetic, train_data
    test_split_ratio = 0.2
    val_split_mode = "same_as_test" # none, same_as_text, from_train, from_test, synthetic (from train_data)
    val_split_ratio = 0.5 # not used if same_as_text
    
    resultsDir = os.path.join("results", dataset)
    prediction_path = os.path.join("results", dataset, category)
    checkpointDir = os.path.join("results", dataset, category, "checkpoints")
    
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
        print(f"Directory created for predictions at: {prediction_path}")
    else:
        print(f"Directory for predictions already exists at: {prediction_path}")
    
    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)
        print(f"Directory created: {checkpointDir}")
    else:
        print(f"Directory already exists: {checkpointDir}")
    
    checkpointFile = f'{model_name}_{category}_best'
    checkpointPath = os.path.join(checkpointDir, checkpointFile + '.ckpt')
    print(f"Saving best result to: {checkpointPath}")

    # 4. Create the training engine
    checkpointCallback = ModelCheckpoint(
        #dirpath=checkpointDir,
        #filename=checkpointFile,
        monitor="train_loss",  # val_loss not found?
        verbose=True,
    )
    
    graphCallback = GraphLogger()
    timerCallback = TimerCallback()
    
    logger = AnomalibTensorBoardLogger(
        save_dir="logs",#os.path.join(resultsDir, "logs"),
        name="FastFlow-EfficientAD",
        version=1)
    
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

    # 3. Initialize the model
    # preProcessor = PreProcessor(transform = Compose([Resize((224, 224)), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    preProcessor = True
    
    visualizer = ImageVisualizer(# output_dir=prediction_path,
                                 fields=["image"],
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
    # val metrics (needed for early stopping)
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    val_metrics = [image_auroc, pixel_auroc]

    # test_metrics
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_f1score]
    evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
    # evaluator = Evaluator(test_metrics=[f1_score, auroc, aupr])
    # evaluator = False
    
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
        
    engine = Engine(
        max_epochs=max_epochs,
        default_root_dir='results',
        callbacks=[checkpointCallback, graphCallback, timerCallback],
        logger=logger,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=10
    )

    # 5. Train the model
    engine.fit(datamodule=datamodule, model=model)
    

    # 6. Validate on validation set. adjust thresholds
    runValidation = 'Yes' if engine._should_run_validation(engine.model, None) else 'No'
    
    print(f"Should we run validation: {runValidation}")
    
    if engine._should_run_validation(engine.model, None):
        engine.validate(
            model=model,
            datamodule=datamodule,
            # ckpt_path=checkpointPath
        )

    # 7. Test on test set
    res = engine.test(
        model=model,
        datamodule=datamodule,
        # ckpt_path=checkpointPath
    )
    
    # 8. Predict on test set
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        # ckpt_path=checkpointPath
    )
    
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
                
                # if predLabel and trueAnomaly:
                    
                #     niO+=1
                #     print(f"Predicted image {j} to be anomalous.")
                # else:
                #     iO+=1
                #     print(f"Predicted image {j} to be normal.")
                pred_score = prediction.pred_score  # Image-level anomaly score
                print(f"Anomaly score: {pred_score}")
    trueAnomalies = np.asarray(trueAnomalies)
    predLabels = np.asarray(predLabels)
    confusionMatrix = confusion_matrix(trueAnomalies, predLabels)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.subplots()
    
    CM_plot = ConfusionMatrixDisplay.from_predictions(trueAnomalies, predLabels, ax=ax)
    print(confusionMatrix)
    CM_plot.figure_.savefig(os.path.join(prediction_path, f"{modelName}_confusion_matrix.png"))
    
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
    
    logger.log_metrics(metrics={"image_TPR": tpr,
                               "image_TNR": tnr,
                               "image_FNR": fnr,
                               "image_FPR": fpr},
                       step=0)
        
    with open(os.path.join(prediction_path, f"{modelName}_results.txt"), 'w') as f:
        f.write(f"Model: {modelName}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Positive: {positive}\n")
        f.write(f"Negative: {negative}\n")
        f.write(f"TP: {tp}\n")
        f.write(f"TN: {tn}\n")
        f.write(f"FP: {fp}\n")
        f.write(f"FN: {fn}\n")
        f.write(f"TPR: {tpr}\n")
        f.write(f"TNR: {tnr}\n")
        f.write(f"FNR: {fnr}\n")
        f.write(f"FPR: {fpr}\n")
        f.write(f"F1 Score: {f1_score}\n")
            
    print("Finished")

    # engine.export(model)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train an anomaly detection model.")
    parser.add_argument("--dataset", type=str, default="kolektor", help="Which dataset to train on")
    parser.add_argument("--category", type=str, default="none", help="Which category to train on")
    parser.add_argument("--modelName", type=str, default="fastflow", help="Which method to train")
    
    parser.add_argument("--train_batch_size", type=int, default=1, help="Number of images per training batch")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Number of images per validation/test batch")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel processes for data loading")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs to train the model")

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
    main(dataset, category, modelName, args.train_batch_size, args.eval_batch_size, args.num_workers, args.max_epochs)
