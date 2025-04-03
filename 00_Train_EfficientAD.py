# 0. add anomalib folder to path
# import sys
# sys.path.append("anomalib")

# 1. Import required modules
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.callbacks import ModelCheckpoint
from anomalib.data.datasets.image.mvtecad import CATEGORIES

# 2. Create a dataset
# MVTecAD is a popular dataset for anomaly detection
datamodule = MVTecAD(
    root="./datasets/MVTecAD",  # Path to download/store the dataset
    category="cable",  # MVTec category to use
    train_batch_size=1,  # Number of images per training batch
    eval_batch_size=32,  # Number of images per validation/test batch
    num_workers=8,  # Number of parallel processes for data loading
)

# 3. Initialize the model
# EfficientAd is a good default choice for beginners
model = EfficientAd()

# 4. Create the training engine
checkpointCallback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best",
    monitor="train_loss"      # val_loss not found?
)
engine = Engine(
    max_epochs=1,
    default_root_dir='results',
    callbacks=[checkpointCallback],
    # accelerator="gpu",
    # devices=1
)  # Train for 100 epochs

# 5. Train the model
from lightning.pytorch.accelerators import find_usable_cuda_devices
print(find_usable_cuda_devices(1))

engine.fit(datamodule=datamodule, model=model, accelerator="cuda", devices=find_usable_cuda_devices(1))