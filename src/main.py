import os
import sys
import yaml
import fiftyone as fo
import fiftyone.core.dataset as fod
import fiftyone.zoo as foz # zoo datasets and models
import cv2
import datetime
import logging

from contextlib import contextmanager
from fiftyone import ViewField as F # helper for defining views
from pathlib import Path
from copy import deepcopy
from functools import partial
from anomalib.metrics import Evaluator
from anomalib.deploy import ExportType
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import AUROC, AUPR, F1AdaptiveThreshold, F1Score
from src.data.anomaly_datasets import TestDataImporter, TrainTestDataImporter, importDataset, importPredictDataset, exportDataset
from setup import create_datamodule, create_model, setupTensorboardLoggingAndCallbacks, setupLogging, LoggerWriter, LoggerStdin, find_first_file
from configs.configs import load_config
from manager import clipEmbedding, resnetEmbedding
from settings import DATASETS, CATEGORIES, MODELS, DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG
from src.legacy.Training import run_inference, train_and_export_model, setupModel
from src.legacy.cameraProcessor import CameraProcessor
from anomalib.callbacks import LoadModelCallback

os.environ["TRUST_REMOTE_CODE"] = "1"

@contextmanager
def exclude_from_logger():
    original_stdout = sys.stdout
    sys.stdout = sys.__stdout__  # Restore original stdout
    try:
        yield
    finally:
        sys.stdout = original_stdout  # Restore logger stdout

def select_category(dataset):
    categories = dataset.distinct("category.label")
    if len(categories) == 0:
        print("No category found in the dataset. Exiting...")
        exit(1)
    if len(categories) == 1:
        print(f"Only one category found. Selecting category '{categories[0]}'")
        return categories[0]
    while True:
        category = input(
            f"Please provide the category to train on: \n"
            f"Possible categories: {categories}:\n"
        ).strip()
        if category == "":
            category = categories[0]
            break
        elif category not in categories:
            print(f"Category {category} not found! Available categories are: {categories}")
        else:
            break
    print(f"Category set to: {category}")

    return category

def parseSplitKeywords(user_input, availableSplits):
    possibleSplits = {"train", "val", "test", "pred"}
    unavailableSplits = set()
    for pos in possibleSplits:
        if pos not in availableSplits:
            unavailableSplits.add(pos)
    split_keywords = {
        'train': 'train',
        'training': 'train',
        'test': 'test',
        'testing': 'test',
        'validation': 'val',
        'val': 'val',
        'predict': 'pred',
        'prediction': 'pred',
        'pred': 'pred'
    }
    all_keywords = {'all', 'everything', 'all splits'}
    user_input_lower = user_input.lower()
    detected_splits = set()
    if any(keyword in user_input_lower for keyword in all_keywords):
        return list(availableSplits), list(unavailableSplits), True
    unavailableSplits = set()
    for keyword in split_keywords:
        if keyword in user_input_lower:
            normalized_split = split_keywords[keyword]
            if normalized_split in availableSplits:
                detected_splits.add(normalized_split)
            else:
                unavailableSplits.add(normalized_split)
    return list(detected_splits), list(unavailableSplits), False

def createSplitView(dataset, user_input):
    availableSplits = dataset.distinct("split")
    detected_splits, unavailableSplits, allKeyword = parseSplitKeywords(user_input, availableSplits)
    if unavailableSplits:
        print(f"Warning: The following splits are not available in the dataset: {', '.join(unavailableSplits)}")
    if not detected_splits:
        print("No valid split keywords detected in the input.")
        return None
    if allKeyword:
        return dataset.view()
    view = dataset.match(
        fo.ViewField("split").is_in(detected_splits)
    )
    return view

def main():
    dataset = None
    datamodule = None
    model = None
    inferencer = None
    engine = None
    session = None
    logFileName = "general.log"
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    logDir = os.path.join("logs", now)
    outputPath = "results"
    weightPath = None

    # Create the general logger
    generalLogger = logging.getLogger('general')
    generalLogger.setLevel(logging.INFO)
    logFormater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create a file handler for the general logger
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    fileHandler = logging.FileHandler(os.path.join(logDir, logFileName))
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(logFormater)
    generalLogger.addHandler(fileHandler)
    # Create a StreamHandler to duplicate console output to the logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    generalLogger.addHandler(console_handler)
    sys.stdout = LoggerWriter(generalLogger, logging.INFO)
    sys.stdin = LoggerStdin(generalLogger, logging.INFO)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    print("Welcome to the ML Model CLI!")

    firstTime = True
    loadModelCallback = None
    callbacks = dict()
    
    session = {"model": None,
               "modelConfig": None,
               "modelName": None,
               "dataset": None,
               "datasetPath": None,}

    while True:
        if firstTime:
            print("\nMenu:")
            print("1. Generate a new model")
            print("2. Load a model")
            print("3. Generate an embedding")
            print("4. Train the model")
            print("5. Retrain the model with new training parameters")
            print("6. Add data to the dataset")
            print("7. Replace current dataset with a new dataset")
            print("8. Add pred dataset")
            print("9. Inference on current dataset")
            print("10. Save dataset to disk")
            print("11. Enter continuous folder mode")
            print("12. Enter camera mode")
            print("q. Exit")
        else:
            with exclude_from_logger():
                print("\nMenu:")
                print("1. Generate a new model")
                print("2. Load a model")
                print("3. Generate an embedding")
                print("4. Train the model")
                print("5. Retrain the model with new training parameters")
                print("6. Add data to the dataset")
                print("7. Replace current dataset with a new dataset")
                print("8. Add pred dataset")
                print("9. Inference on current dataset")
                print("10. Save dataset to disk")
                print("11. Enter continuous folder mode")
                print("12. Enter camera mode")
                print("q. Exit")
        firstTime = False

        menu_choice = input("Enter your choice [1-11]: ").strip()

        if menu_choice == "1":
            configDir = "configs"
            while True:
                userPath = input("Please provide the path to your YAML config file: ").strip()
                if not userPath.lower().endswith('.yaml'):
                    userPath = userPath + '.yaml'
                configPath = os.path.join(configDir, userPath)
                if userPath == "q":
                    break
                elif userPath == "":
                    configPath = os.path.join(configDir, "padim.yaml")
                    break
                if not os.path.exists(configPath):
                    print("Error: Config file not found.")
                else:
                    break
            if userPath == "q":
                continue
            else:
                modelConfig, fullCopyPath = load_config(configPath, copyPath=logDir)
                modelName = modelConfig.pop("model").lower()

                print(f"Model {modelName} loaded: {modelConfig}")
                print(f"Model config copied to {fullCopyPath}.")
                model = create_model(modelName, modelConfig)
                print(f"Model {model} created.")

        elif menu_choice == "2":
            folderPath = input("Please provide the path to your logs folder (e.g. logs/20251028-0946): ").strip()
            modelPath = find_first_file(folderPath, "model.pt")
            for model in MODELS:
                configPath = find_first_file(folderPath, f"{model}.yaml")
            modelConfig, fullCopyPath = load_config(configPath, copyPath=logDir)
            modelName = modelConfig.pop("model").lower()
            model = create_model(modelName, modelConfig)

            if not os.path.exists(modelPath):
                print("Error: Model file not found.")
                continue
            else:
                print(f"Found model weights at {modelPath}")
                loadModelCallback = LoadModelCallback(weights_path=modelPath) 
                if loadModelCallback is not None:
                    callbacks["loadModel"] = loadModelCallback
                print("Model loaded.")

        elif menu_choice == "3":
            print("Creating embedding...")
            if dataset is None:
                print("No dataset loaded. Please load or create a dataset first.")
                continue
            with exclude_from_logger():
                clipEmbedding(dataset)
            print("Finished embedding computation.")
            print("Please reload the FiftyOne app to see the new visualizations.")
            print("You find the visualizations by clicking the '+' next to Samples and choosing Embeddings.")

        elif menu_choice == "4":
            if model is None:
                print("No model loaded. Please generate or load a model first.")
                continue
            if dataset is None:
                print("No dataset loaded. Please load or create a dataset first.")
                continue
            print("Training the model...")
            while True:
                config_name = input("Please provide the name of your training config file in the configs folder (or press Enter for default, 'q' to quit): ").strip()
                if config_name.lower() == "q":
                    break
                elif config_name == "":
                    trainingConfigPath = f"configs/{modelName}_Training.yaml"
                    break
                else:
                    trainingConfigPath = os.path.join("configs", config_name)
                    if not os.path.exists(trainingConfigPath):
                        print("Error: Config file not found in configs folder.")
                    else:
                        break
            with open(trainingConfigPath, 'r') as f:
                trainingConfig = yaml.safe_load(f)
            category = select_category(dataset)

            runName = f"{modelName}-{datasetName}-{category}"
            versionName = "version_0"
            if not os.path.exists(os.path.join(logDir, runName, versionName)):
                os.makedirs(os.path.join(logDir, runName, versionName))
            print("Logging to log directory:", os.path.join(logDir, runName))
            tblogger, callbacks, ckptPath = setupTensorboardLoggingAndCallbacks(logDir=logDir, runName=runName, versionName=versionName, version=0)

            logger, modelDir = setupLogging(logDir=logDir, runName=runName, versionName=versionName)

            print("Training model...")
    
            engine, datamodule = train_and_export_model(rootDir=modelDir,
                                                        dataset=dataset,
                                                        model=model,
                                                        trainingConfig=trainingConfig,
                                                        logger=tblogger,
                                                        callbacks=callbacks)
            # model.evaluator = tmpEvaluator
            print("Running inference on dataset...")
            with exclude_from_logger():
                run_inference(dataset, engine, modelName)
            session = fo.launch_app(dataset)
            weightPath = os.path.join(modelDir, "weights", "torch", "model.pt")
            print(f"Exported model weights to {weightPath}")


        elif menu_choice == "5":
            print("Retraining the model with new parameters...")
            raise NotImplementedError

        elif menu_choice == "6" or menu_choice == "7" or menu_choice == "8":
            while True:
                print("Please provide the name of the training data folder in datasets. E.g.: MVTecAD (or type 'q' to quit): ")
                datasetName = input().strip()
                if datasetName.lower() == 'q':
                    break
                elif datasetName == "":
                    datasetName = "traintest"
                dataDir = os.path.join("datasets", datasetName)
                if not os.path.isdir(dataDir):
                    print("Error: Directory not found.")
                else:
                    print(f"Training data directory set to: {dataDir}")
                    break
            if menu_choice == "6":
                if dataset is None:
                    # If no dataset is loaded, treat as "Replace current dataset"
                    newDatasetName = input("Please provide a name for the new dataset (blank input keeps the current name): ").strip()
                    if newDatasetName != "":
                        datasetName = newDatasetName
                    print(f"Dataset name set to: {datasetName}")
                    print("Importing dataset...")
                    dataset, _ = importDataset(
                        path=dataDir,
                        name=datasetName,
                        overwrite=False,
                        split=["train", "test"],
                    )
                else:
                    newDatasetName = input("Please provide a name for the merged dataset (blank input adds '_merged' to current name): ").strip()
                    if newDatasetName != "":
                        datasetName = newDatasetName
                    else:
                        datasetName += '_merged'
                    print(f"Dataset name set to: {datasetName}")
                    print("Importing dataset...")
                    newDataset, _ = importDataset(
                        path=dataDir,
                        name=datasetName,
                        overwrite=False,
                        split=["train", "test"],
                    )
                    dataset.merge_samples(newDataset)
            elif menu_choice == "7":
                newDatasetName = input("Please provide a name for the new dataset (blank input keeps the current name): ").strip()
                if newDatasetName != "":
                    datasetName = newDatasetName
                print(f"Dataset name set to: {datasetName}")
                print("Importing dataset...")
                dataset, _ = importDataset(
                    path=dataDir,
                    name=datasetName,
                    overwrite=False,
                    split=["train", "test"],
                )
            elif menu_choice == "8":
                newDatasetName = input("Please provide a name for the pred dataset (blank input adds '_pred' to current name): ").strip()
                if newDatasetName != "":
                    datasetName = newDatasetName
                else:
                    datasetName += '_pred'
                print(f"Dataset name set to: {datasetName}")
                print("Importing dataset...")
                newDataset, _ = importDataset(
                    path=dataDir,
                    name=datasetName,
                    overwrite=False,
                    split="pred",
                )
                if dataset is None:
                    dataset = newDataset
                else:
                    dataset.merge_samples(newDataset)
            session = fo.launch_app(dataset)

        elif menu_choice == "9":
            if engine is None:
                print("Train the model first.")
                continue
            print("Running inference on dataset...")
            splits = input("Select the splits to run the inference on (training, validation, test, prediction, all)\n")
            split_view = createSplitView(dataset, splits)
            run_inference(split_view, engine, modelName)
            session = fo.launch_app(split_view)

        elif menu_choice == "10":
            print("Exporting...")
            name = input("Dataset name:\n")
            exportDataset(dataset=dataset, path=os.path.join("datasets", name))

        elif menu_choice == "11":
            print("Entering continuous folder mode. In this mode the program continuously observes a folder of choice. For each image a prediction is made and the result saved.")
            print("After the processing of an image it is moved to another folder of choice such that it is not processed again.")
            obsDir = input("Which folder should be observed: ")

        elif menu_choice == "12":
            print("Entering continuous camera mode. In this mode a connected camera is used for a continuous camera feed. After it gets the signal to take a picture it process it with the AD model.")
            folder = input("Please enter a name for the folder in datasets where the images should be saved.\n")
            if folder == "":
                folder = "cameraPrediction"
            fault = input("Please enter a name for the fault type which should be saved. E.g. 'scratch' or 'bent'\n")
            if fault == "":
                fault = "unknown"
            saveDir = os.path.join("datasets",folder,fault,"prediction")
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            else:
                print(f"Warning: Directory {saveDir} already exists. New images will override the existing files.")
            print("The signal can be given manually via keyboard or automatically.")
            manual = input("Manual (m) or automatic (a)\n")
            if "m" in manual:
                manual = True
            elif "a" in manual:
                manual = False
            elif "q" in manual:
                continue
            else:
                continue
            manualString = "manual" if manual else "automatic"
            print(f"You chose {manualString} mode")
            if manual:
                print("Creating Camera Processor")
                cameraProcessor = CameraProcessor()
                saveDir = os.path.join("datasets","cameraPrediction","face","prediction")
                cameraProcessor.set_crop_region(256, 256)
                cameraProcessor.set_save_directory(saveDir)
                cameraProcessor.register_capture_callback(cameraProcessor.save_image)
                cameraProcessor.start()
                cameraProcessor.display_frames()
                print(f"Finished capturing to {saveDir}")
            else:
                raise NotImplementedError

        elif menu_choice == "q":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
