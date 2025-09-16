import fiftyone as fo # base library and app
from fiftyone import ViewField as F # helper for defining views

import numpy as np
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms.v2 import Resize

from anomalib import TaskType
from anomalib.models import Padim, Patchcore, Stfpm

from setup import define_metrics
from Training import train_and_export_model, run_inference
from AnomalyDataset import importTrainTestDataset

import torch
os.environ["TRUST_REMOTE_CODE"] = "1"

# def create_datamodule(rootDir, dataset, object_type, image_size=256, transform=None):
#     ## Build transform
#     if transform is None:
#         transform = Resize(image_size, antialias=True)

#     normal_data = dataset.match(F("category.label") == object_type).match(
#         F("split") == "train"
#     )
#     abnormal_data = (
#         dataset.match(F("category.label") == object_type)
#         .match(F("split") == "test")
#         .match(F("anomalyType.label") != "good")
#     )

#     normal_dir = os.path.join(rootDir, object_type, "normal")
#     abnormal_dir = os.path.join(rootDir, object_type, "abnormal")
#     mask_dir = os.path.join(rootDir, object_type, "mask")

#     # create directories if they do not exist
    
#     os.makedirs(abnormal_dir, exist_ok=True)
#     os.makedirs(mask_dir, exist_ok=True)

#     if not os.path.exists(str(normal_dir)):
#         os.makedirs(normal_dir, exist_ok=True)
#         normal_data.export(
#             export_dir=str(normal_dir),
#             dataset_type=fo.types.ImageDirectory,
#             export_media="symlink",
#         )

#     for sample in abnormal_data.iter_samples():
#         base_filename = sample.filename
#         dir_name = os.path.dirname(sample.filepath).split("/")[-1]
#         new_filename = f"{dir_name}_{base_filename}"
#         if not os.path.exists(os.path.join(abnormal_dir, new_filename)):
#             os.symlink(sample.filepath, os.path.join(abnormal_dir, new_filename))

#         if not os.path.exists(os.path.join(mask_dir, new_filename)) and sample.ground_truth is not None:
#             os.symlink(sample.ground_truth.mask_path, os.path.join(mask_dir, new_filename))

#     datamodule = Folder(
#         name=object_type,
#         root=rootDir,
#         normal_dir=os.path.relpath(normal_dir, start=rootDir),
#         abnormal_dir=os.path.relpath(abnormal_dir, start=rootDir),
#         mask_dir=os.path.relpath(mask_dir, start=rootDir),
#         train_batch_size=1,  # Number of images per training batch
#         eval_batch_size=1,  # Number of images per validation/test batch
#         num_workers=1,  # Number of parallel processes for data loading
#     )

#     datamodule.setup()
#     return datamodule

# def train_and_export_model(rootDir, dataset, object_type, model, transform=None):
#     engine = Engine(max_epochs=20)
#     datamodule = create_datamodule(rootDir, dataset, object_type, transform=transform)
#     engine.fit(model=model, datamodule=datamodule)

#     engine.export(
#         model=model,
#         export_type=ExportType.TORCH,
#     )
#     output_path = Path(engine.trainer.default_root_dir)

#     torch_model_path = output_path / "weights" / "torch" / "model.pt"
#     metadata = output_path / "weights" / "openvino" / "metadata.json"

#     inferencer = TorchInferencer(
#         path=torch_model_path,
#         device="cpu",
#     )

#     return engine, datamodule, inferencer

# def run_inference(sample_collection, inferencer: TorchInferencer, key, threshold=0.5):
#     for sample in sample_collection.iter_samples(autosave=True, progress=True):
#         output = inferencer.predict(image=Image.open(sample.filepath))

#         conf = output.pred_score.item()
#         anomaly = "normal" if conf < threshold else "anomaly"

#         map_path = "tmp/seg.png"

#         # pil_image = Image.fromarray(output.pred_mask.data.numpy().squeeze().astype(np.int16)*255).convert("L")
#         # pil_image.save(map_path)

#         sample[f"pred_anomaly_score_{key}"] = conf
#         sample[f"pred_anomaly_{key}"] = fo.Classification(label=anomaly)
#         sample[f"pred_anomaly_map_{key}"] = fo.Heatmap(map=output.anomaly_map.data.numpy().squeeze()*255, range=[0,255])
#         sample[f"pred_defect_mask_{key}"] = fo.Segmentation(mask=output.pred_mask.data.numpy().squeeze().astype(np.int16)*255)

# def run_engine_inference(dataloader, engine: Engine, model, key, threshold=0.5):
#     output = engine.predict(model=model, dataloaders=dataloader)
#     pass
#     return output
#     # conf = output.pred_score
#     # anomaly = "normal" if conf < threshold else "anomaly"

#     # sample[f"pred_anomaly_score_{key}"] = conf
#     # sample[f"pred_anomaly_{key}"] = fo.Classification(label=anomaly)
#     # sample[f"pred_anomaly_map_{key}"] = fo.Heatmap(map=output.anomaly_map)
#     # sample[f"pred_defect_mask_{key}"] = fo.Segmentation(mask=output.pred_mask)

# from anomalib.metrics import F1Score, AUPR, AUROC, F1AdaptiveThreshold

# def define_metrics():
#     # val metrics (needed for early stopping)
#     image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
#     pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
#     image_aupr = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
#     pixel_aupr = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
#     image_f1score = F1AdaptiveThreshold(fields=["pred_score", "gt_label"], prefix="image_")
#     val_metrics = [image_auroc, pixel_auroc, image_aupr, pixel_aupr, image_f1score]

#     # test_metrics
#     image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
#     image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
#     pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
#     pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
#     image_aupr = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
#     pixel_aupr = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
#     test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_f1score, image_aupr, pixel_aupr]
    
#     return val_metrics, test_metrics

if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")

    device = torch.device("cpu")

    dataset, info = importTrainTestDataset("datasets/MVTecAD_traintestdata/bottle", "TrainTestDataset", overwrite=True, split=("train", "test"))

    # 2. Explore the dataset using the App
    session = fo.launch_app(dataset)

    OBJECT = "bottle" ## object to train on
    ROOT_DIR = Path("tmp/datasets/MVTecAD_traintestdata/bottle") ## root directory to store data for anomalib
    TASK = TaskType.SEGMENTATION ## task type for the model
    IMAGE_SIZE = (256, 256) ## preprocess image size for uniformity
    from anomalib.post_processing import PostProcessor

    postProcessor = PostProcessor(enable_normalization=True,
                                  enable_threshold_matching=True,
                                  enable_thresholding=True,
                                  image_sensitivity=0.4,
                                  pixel_sensitivity=0.4)
    
    from anomalib.metrics import Evaluator

    val_metrics, test_metrics = define_metrics()
    
    evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)

    model = Padim(post_processor=postProcessor, evaluaator=evaluator)
    # model = Padim()

    engine, datamodule, inferencer = train_and_export_model(ROOT_DIR, dataset, model, transform=None)

    ## get the test split of the dataset
    test_split = dataset.match(F("category.label") == OBJECT).match(
        F("split") == "test"
    )

    # ## get the first sample from the test split
    test_image = Image.open(test_split.first().filepath)

    # output = inferencer.predict(image=test_image)
    output = inferencer.predict(test_image) 
    print(output)

    run_inference(test_split, inferencer, "padim")
    # output = run_engine_inference(datamodule.test_dataloader(), engine, model, "padim", threshold=0.8)

    session = fo.launch_app(view=test_split)

    for sample in test_split.iter_samples(autosave=True, progress=True):
        if sample["anomalyType"].label == "good":
            sample["defect_mask"] = fo.Segmentation(
                mask=np.zeros_like(sample["pred_defect_mask_padim"].mask)
            )

    old_labels = test_split.distinct("anomalyType.label")
    label_map = {label:"anomaly" for label in old_labels if label != "good"}
    label_map["good"] = "normal"
    mapped_view = test_split.map_labels("anomalyType", label_map)

    session.view = mapped_view.view()

    eval_classif_padim = mapped_view.evaluate_classifications(
        "pred_anomaly_padim",
        gt_field="anomalyType",
        eval_key="eval_classif_padim",
        method="binary",
        classes=["normal", "anomaly"],
    )

    eval_classif_padim.print_report()

    eval_seg_padim = mapped_view.evaluate_segmentations(
        "pred_defect_mask_padim",
        gt_field="ground_truth",
        eval_key="eval_seg_padim",
    )

    eval_seg_padim.print_report(classes=[0, 255])

    session.wait()