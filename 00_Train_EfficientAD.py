import sys
import argparse
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.callbacks import ModelCheckpoint
from anomalib.data.datasets.image.mvtecad import CATEGORIES
import torch
from lightning.pytorch import Trainer   

# only for Cluster with NVIDIA L40S GPU
torch.set_float32_matmul_precision("high")          # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

def main(train_batch_size, eval_batch_size, num_workers, max_epochs, category):
    # 0. Add anomalib folder to path (if necessary)
    # sys.path.append("anomalib")

    # 2. Create a dataset
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",  # Path to download/store the dataset
        category=category,  # MVTec category to use
        train_batch_size=train_batch_size,  # Number of images per training batch
        eval_batch_size=eval_batch_size,  # Number of images per validation/test batch
        num_workers=num_workers,  # Number of parallel processes for data loading
    )

    # 3. Initialize the model
    model = EfficientAd(model_size="small")

    # 4. Create the training engine
    checkpointCallback = ModelCheckpoint(
        dirpath="results/checkpoints",
        filename=f"EfficientAD_{category}_best.ckpt",
        monitor="train_loss",  # val_loss not found?
        verbose=True,
        
    )
    engine = Engine(
        max_epochs=max_epochs,
        default_root_dir='results',
        callbacks=[checkpointCallback],
        accelerator="cuda",
        devices=1
    )

    # 5. Train the model
    engine.fit(datamodule=datamodule, model=model)
    
    engine.export(model)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train an anomaly detection model.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Number of images per training batch")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Number of images per validation/test batch")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel processes for data loading")
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of epochs to train the model")
    parser.add_argument("--category", type=str, default="leather", help="Which MVTec category to train on")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.train_batch_size, args.eval_batch_size, args.num_workers, args.max_epochs, args.category)
