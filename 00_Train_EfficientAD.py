import sys
import argparse
from anomalib.data import MVTecAD, BTech, Visa, Kolektor
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Dsr, ReverseDistillation, Fastflow, Patchcore, Stfpm
from anomalib.callbacks import ModelCheckpoint
from anomalib.data.datasets.image.mvtecad import CATEGORIES
import torch
from lightning.pytorch import Trainer
import os

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


# only for Cluster with NVIDIA L40S GPU
torch.set_float32_matmul_precision("high")          # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

def main(dataset, category, model_name, train_batch_size, eval_batch_size, num_workers, max_epochs):
    # 0. Add anomalib folder to path (if necessary)
    # sys.path.append("anomalib")
    
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
        )
    elif dataset.lower() == "kolektor":
        datasetPath = f"datasets/kolektor"
        print(f"searching for dataset at: {datasetPath}")
        datamodule = Kolektor(
            root=datasetPath,  # Path to download/store the dataset
            train_batch_size=train_batch_size,  # Number of images per training batch
            eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
            num_workers=num_workers,  # Number of parallel processes for data loading
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
        )

    # 3. Initialize the model
    if model_name.lower() == "efficientad-s":
        model = EfficientAd(model_size="small")
    elif model_name.lower() == "efficientad-m":
        model = EfficientAd(model_size="medium")
    elif model_name.lower() == "dsr":
        model = Dsr()
    elif model_name.lower() == "reversedistillation":
        model = ReverseDistillation()
    elif model_name.lower() == "reverse_distillation":
        model = ReverseDistillation()
    elif model_name.lower() == "rd":
        model = ReverseDistillation()
    elif model_name.lower() == "stfpm":
        model = Stfpm()
    elif model_name.lower() == "fastflow":
        model = Fastflow()
    elif model_name.lower() == "fast_flow":
        model = Fastflow()
    elif model_name.lower() == "patchcore":
        model = Patchcore()
    
    checkpointDir = "results/checkpoints"

    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)
        print(f"Directory created: {checkpointDir}")
    else:
        print(f"Directory already exists: {checkpointDir}")
    
    chechpointDatasetFolder = os.path.join(checkpointDir, dataset.lower())
    
    if not os.path.exists(chechpointDatasetFolder):
        os.makedirs(chechpointDatasetFolder)
        print(f"Directory created: {chechpointDatasetFolder}")
    else:
        print(f"Directory already exists: {chechpointDatasetFolder}")
        
    
    checkpointFile = f'{model_name}_{category}_best'
    print(f"Saving best result to: {chechpointDatasetFolder + '/' + checkpointFile}")

    # 4. Create the training engine
    checkpointCallback = ModelCheckpoint(
        dirpath=chechpointDatasetFolder,
        filename=checkpointFile,
        monitor="train_loss",  # val_loss not found?
        verbose=True,
        
    )
    engine = Engine(
        max_epochs=max_epochs,
        default_root_dir='results',
        callbacks=[checkpointCallback],
        accelerator="cpu",
        devices=1
    )

    # 5. Train the model
    engine.fit(datamodule=datamodule, model=model)
    
    # engine.export(model)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train an anomaly detection model.")
    parser.add_argument("--dataset", type=str, default="mvtecad", help="Which MVTec category to train on")
    parser.add_argument("--category", type=str, default="leather", help="Which MVTec category to train on")
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
