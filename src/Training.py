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
from AnomalyDataset import FODataModule

ENGINE_PARAMS = ["max_epochs", "min_epochs", "log_every_n_steps"]
DATAMODULE_PARAMS = ["train_batch_size",
                   "train_augmentations",
                   "val_augmentations", 
                   "test_augmentations",
                   "augmentations",
                   "val_split_mode",
                   "val_split_ratio",
                   "test_split_mode",
                   "test_split_ratio",
                   "seed"]

def train_and_export_model(rootDir, dataset, model, transform=None, callbacks=None, logger=None, trainingConfig={}, ckptPath=None):
    engineParams = {key: trainingConfig[key] for key in ENGINE_PARAMS if key in trainingConfig}
    datamoduleParams = {key: trainingConfig[key] for key in DATAMODULE_PARAMS if key in trainingConfig}

    engine = Engine(callbacks=list(callbacks.values()), logger=logger, **engineParams)
    datamodule = FODataModule(name="Train", samples=dataset, root=rootDir, **datamoduleParams)
    datamodule.setup()
    engine.fit(model=model, datamodule=datamodule)

    # engine.export(
    #     model=model,
    #     export_type=ExportType.TORCH,
    # )
    engine.export(model=model,
                  export_type=ExportType.TORCH,
                  export_root=rootDir,
                  model_file_name="model")
    output_path = Path(engine.trainer.default_root_dir)

    torch_model_path = os.path.join(rootDir, "weights", "torch", "model.pt")
    metadata = os.path.join(rootDir, "metadata.json")

    inferencer = TorchInferencer(
        path=torch_model_path,
        device="cpu",
    )

    return engine, datamodule, inferencer

def run_inference(sample_collection, inferencer: TorchInferencer, key):
    for sample in sample_collection.iter_samples(autosave=True, progress=True):
        output = inferencer.predict(image=Image.open(sample.filepath))

        conf = output.pred_score.item()
        anomaly = "anomaly" if output.pred_label else "normal"

        sample[f"pred_anomaly_score_{key}"] = conf
        sample[f"pred_anomaly_{key}"] = fo.Classification(label=anomaly)
        sample[f"pred_anomaly_map_{key}"] = fo.Heatmap(map=output.anomaly_map.data.numpy().squeeze()*255, range=[0,255])
        sample[f"pred_defect_mask_{key}"] = fo.Segmentation(mask=output.pred_mask.data.numpy().squeeze().astype(np.int16)*255)