import os
import sys
import warnings
import yaml
import pathlib
from pathlib import Path
# Set BEFORE any fiftyone imports
os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://localhost"
os.environ["WANDB_API_KEY"] = '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'
# # sys.path.append("..")
# try:
#     base = Path(__file__).parent
# except NameError:
#     base = Path.cwd()  # fallback for notebooks/REPL

# sys.path.insert(0, str(base))
sys.path.append("src")

# Now safe to import
import wandb
import logging
from src.manager import AnomalyDetectionManager as ADM
from src.manager import DatasetSession as DS
from src.userConfigs import Product


wandb.login()


train       = True
evaluate    = True
datasetDir  = Path("datasets/")
configDir   = Path("configs/")
outputPath  = Path("results/")
productPath     = Path("Products/cable.yaml")
product: Product
logger = logging.getLogger("logger")
productConfigPath=Path(configDir/productPath)
# with productConfigPath.open("r", encoding="utf-8") as f:
#     productConfig = yaml.safe_load(f)
# print(productConfig["product"])

# tilingConfigPath = Path(configDir / "Tiling" / productConfig["tiling"]["config"])

manager, product = ADM.loadProduct(productConfigPath=productConfigPath, outputPath=outputPath, configDir=configDir)

# datasetSession = DS.loadDatasetFromDatabase(productConfig["dataset"]["name"])
datasetSession = DS.loadDatasetFromConfig(product.datasetConfig, overwrite=True, merge=False)
# datasetSession = DS.loadDatasetFromDisk(datasetDir/product.datasetConfig.name, product.datasetConfig.name)
datasetSession.select_category(product.name)
path = manager.adjustPaths(datasetName=datasetSession.datasetName, category=product.name)

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="openvino.runtime")
warnings.filterwarnings("ignore", category=FutureWarning)

# if train and manager.trainerPath is not None:
if train:
    # print(manager.isTilingSetup)
    manager.train(trainerConfig=product.trainerConfig, 
                  modelConfig=product.modelConfig,
                  datamoduleConfig=product.datamoduleConfig,
                  datasetSession=datasetSession,
                  tiling=manager.isTilingSetup,
                  tilingPipelineConfig=product.tilingPipelineConfig)
