import sys
import argparse
from anomalib.data import MVTecAD, BTech, Visa, Kolektor
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Dsr, ReverseDistillation, Fastflow, Patchcore, Stfpm
from anomalib.callbacks import ModelCheckpoint
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import F1Score, AUPR, AUROC, Evaluator
from anomalib.visualization import ImageVisualizer
from torchvision.transforms import Compose, Normalize, Resize
from anomalib.loggers import AnomalibTensorBoardLogger

from anomalib.data.datasets.image.mvtecad import CATEGORIES
import torch
from lightning.pytorch import Trainer
import os


DATASETS = ["mvtecad", "kolektor", "visa", "btech"]     # TODO. "isp-ad", "wfdd", (not in anomalib)
CATEGORIES = {
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
    checkpointDir = os.path.join("results", dataset, "checkpoints")
    prediction_path = os.path.join("results", dataset, category)
    
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
        dirpath=checkpointDir,
        filename=checkpointFile,
        monitor="train_loss",  # val_loss not found?
        verbose=True,
    )
    
    tensorboard_logger = AnomalibTensorBoardLogger(
        save_dir=os.path.join(resultsDir, "logs"),
        name="test_experiment",
        version=0)
    
    # 2. Create a dataset
    if dataset.lower() == "mvtecad":
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

    # 3. Initialize the model
    preProcessor = PreProcessor(transform = Compose([Resize((224, 224)), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    preProcessor = True
    
    visualizer = ImageVisualizer(output_dir=prediction_path)
                                #  fields=["image", "gt_mask"],
                                #  overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],
                                #  field_size=(256,256),
                                #  fields_config=DEFAULT_FIELDS_CONFIG,
                                #  overlay_fields_config=DEFAULT_OVERLAY_FIELDS_CONFIG,
                                #  text_config=DEFAULT_TEXT_CONFIG)
    visualizer = True
    postProcessor = PostProcessor(enable_normalization=True,
                                  enable_threshold_matching=True,
                                  enable_thresholding=True,
                                  image_sensitivity=0.01,
                                  pixel_sensitivity=0.01)
    postProcessor = True
    f1_score = F1Score(fields=["pred_label", "gt_label"])
    auroc = AUROC(fields=["pred_score", "gt_label"])
    aupr = AUPR(fields=["pred_score", "gt_label"])
    # evaluator = Evaluator(val_metrics=[f1_score, auroc, aupr], test_metrics=[f1_score, auroc, aupr])
    evaluator = Evaluator(test_metrics=[f1_score, auroc, aupr])
    evaluator = True
    
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
        callbacks=[checkpointCallback],
        logger=tensorboard_logger,
        accelerator="cpu",
        devices=1
    )

    # 5. Train the model
    engine.fit(datamodule=datamodule, model=model)
    
    runValidation = 'Yes' if engine._should_run_validation(engine.model, None) else 'No'
    
    print(f"Should we run validation: {runValidation}")
    
    # 6. Validate on validation set. adjust thresholds
    if engine._should_run_validation(engine.model, None):
        engine.validate(
            model=model,
            datamodule=datamodule,
            # ckpt_path=checkpointPath
        )
    
    # 7. Predict for visualisation of results
    visualizer = ImageVisualizer(output_dir=prediction_path)
    
    engine.model.visualizer = visualizer
    
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        # ckpt_path=checkpointPath
    )
    
    # 6. Access the results
    iO = 0
    niO = 0
    if predictions is not None:
        for i, batch in enumerate(predictions):
            for j, prediction in enumerate(batch):
                image_path = prediction.image_path
                anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
                pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
                if pred_label:
                    niO+=1
                    print(f"Predicted image {j} to be anomalous.")
                else:
                    iO+=1
                    print(f"Predicted image {j} to be normal.")
                pred_score = prediction.pred_score  # Image-level anomaly score
                print(f"Anomaly score: {pred_score}")
    print(f"Number of predicted anomalous samples: {niO}")
    print(f"Number of predicted normal samples: {iO}")
            
            # torchvision.utils.save_image(prediction.image, os.path.join(prediction_path, f"cable_image_{i}.png"))
            # torchvision.utils.save_image(prediction.anomaly_map, os.path.join(prediction_path, f"cable_anomaly_map_{i}.png"))
            
    print("Finished")

            
    
    
    # engine.export(model)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train an anomaly detection model.")
    parser.add_argument("--dataset", type=str, default="kolektor", help="Which MVTec category to train on")
    parser.add_argument("--category", type=str, default="none", help="Which MVTec category to train on")
    parser.add_argument("--modelName", type=str, default="EfficientAD-S", help="Which MVTec category to train on")
    
    parser.add_argument("--train_batch_size", type=int, default=1, help="Number of images per training batch")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Number of images per validation/test batch")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel processes for data loading")
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of epochs to train the model")

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
