import fiftyone as fo # base library and app
from fiftyone import ViewField as F # helper for defining views
import numpy as np
import os

from pathlib import Path
from PIL import Image
from torchvision.transforms.v2 import Resize

from anomalib.deploy import ExportType, TorchInferencer
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, Stfpm
from data.anomaly_datasets import FODataModule
from anomalib.models.components.base import AnomalibModule


from tiling.tiled_ensemble import TrainTiledEnsemble, EvalTiledEnsemble
from data.anomaly_datasets import importDataset, FODataModule
import argparse
import os
import logging
from rich import traceback
traceback.install()

# def setupModel(callbacks=None, logger=None, trainingConfig={}, ckptPath=None, tiling=False):
#     engineParams = {key: trainingConfig[key] for key in ENGINE_PARAMS if key in trainingConfig}
#     # datamoduleParams = {key: trainingConfig[key] for key in DATAMODULE_PARAMS if key in trainingConfig}

#     # datamodule = FODataModule(name="Train", samples=dataset, root=rootDir, **datamoduleParams)
#     # datamodule.setup()

#     # if tiling:
#     #     # logger.info("Image too large for one model. Image gets tiled and processed by multiple models.")
#     #     args = "configs/TiledEnsemble.yaml"
#     #     train_pipeline = TrainTiledEnsemble()
#     #     train_pipeline.setDatamodule(datamodule=datamodule)
#     #     # run training
#     #     return train_pipeline

#     engine = Engine(callbacks=list(callbacks.values()), logger=logger, **engineParams)

#     return engine

# def train_and_export_model(rootDir, dataset, model, transform=None, callbacks=None, logger=None, trainingConfig={}, ckptPath=None, tiling=False):
#     engineParams = {key: trainingConfig[key] for key in ENGINE_PARAMS if key in trainingConfig}
#     datamoduleParams = {key: trainingConfig[key] for key in DATAMODULE_PARAMS if key in trainingConfig}

#     datamodule = FODataModule(name="Train", samples=dataset, root=rootDir, **datamoduleParams)
#     datamodule.setup()

#     if tiling:
#         # logger.info("Image too large for one model. Image gets tiled and processed by multiple models.")
#         args = "configs/TiledEnsemble.yaml"
#         train_pipeline = TrainTiledEnsemble()
#         train_pipeline.setDatamodule(datamodule=datamodule)
#         # run training
#         train_pipeline.run(args)
#         return train_pipeline

#     engine = Engine(callbacks=list(callbacks.values()), logger=logger, **engineParams)
#     engine.fit(model=model, datamodule=datamodule)
#     engine.export(model=model,
#                   export_type=ExportType.TORCH,
#                   export_root=rootDir,
#                   model_file_name="model")
#     output_path = Path(engine.trainer.default_root_dir)

#     torch_model_path = os.path.join(rootDir, "weights", "torch", "model.pt")
#     metadata = os.path.join(rootDir, "metadata.json")

#     return engine, datamodule

def run_inference(sample_collection, engine: Engine, model:AnomalibModule, key:str):
    for sample in sample_collection.iter_samples(autosave=True, progress=True):
        output = engine.predict(data_path=sample.filepath, model=model)[0]
        
        conf = output.pred_score.item()
        anomaly = "anomaly" if output.pred_label else "normal"

        sample[f"pred_anomaly_score_{key}"] = conf
        sample[f"pred_anomaly_{key}"] = fo.Classification(label=anomaly)
        sample[f"pred_anomaly_map_{key}"] = fo.Heatmap(map=output.anomaly_map.data.numpy().squeeze()*255, range=[0,255])
        sample[f"pred_defect_mask_{key}"] = fo.Segmentation(mask=output.pred_mask.data.numpy().squeeze().astype(np.int16)*255)