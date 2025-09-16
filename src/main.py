import os
import yaml

import fiftyone as fo
import fiftyone.core.dataset as fod
import fiftyone.zoo as foz # zoo datasets and models
from fiftyone import ViewField as F # helper for defining views
from pathlib import Path
from copy import deepcopy


from anomalib.metrics import Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import AUROC, AUPR, F1AdaptiveThreshold, F1Score


from AnomalyDataset import TestDataImporter, TrainTestDataImporter, importDataset, importPredictDataset
from setup import create_datamodule, create_model, setupTensorboardLoggingAndCallbacks, setupLogging, define_metrics
from Configs import load_config
from Visualisation import clipEmbedding, resnetEmbedding
from settings import DATASETS, CATEGORIES, MODELS, DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG
from Training import run_inference, train_and_export_model

os.environ["TRUST_REMOTE_CODE"] = "1"

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
    # List of common split-related keywords
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

    # Keywords indicating "all splits"
    all_keywords = {'all', 'everything', 'all splits'}

    # Normalize input and find matches
    user_input_lower = user_input.lower()
    detected_splits = set()

    # Check if the user wants all splits
    if any(keyword in user_input_lower for keyword in all_keywords):
        return list(availableSplits), list(unavailableSplits), True

    unavailableSplits = set()
    # Check for specific splits
    for keyword in split_keywords:
        if keyword in user_input_lower:
            normalized_split = split_keywords[keyword]
            if normalized_split in availableSplits:
                detected_splits.add(normalized_split)
            else:
                unavailableSplits.append(normalized_split)

    return list(detected_splits), list(unavailableSplits), False

def createSplitView(dataset, user_input):
    availableSplits = dataset.distinct("split")
    detected_splits, unavailableSplits, allKeyword = parseSplitKeywords(user_input, availableSplits)

    if unavailableSplits:
        print(f"Warning: The following splits are not available in the dataset: {', '.join(unavailableSplits)}")

    if not detected_splits:
        print("No valid split keywords detected in the input.")
        return None

    # If "all" is detected, return the entire dataset
    if allKeyword:
        return dataset.view()

    # Otherwise, create a view for the detected splits
    view = dataset.match(
        fo.ViewField("split").is_in(detected_splits)
    )

    return view


# os.environ["FIFTYONE_DATABASE_URI"]="mongodb://127.0.0.1:6969/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.5.7"

def main():
    print("Welcome to the ML Model CLI!")

    # Step 1: Ask user if they want to train a new model or use an existing one
    choice = input("Do you want to train a new model (1) or use an existing model (2)? [1/2]: ").strip()

    while True:
        if choice == "1":
            # Ask for YAML config file
            configDir = "configs"
            while True:
                userPath = input("Please provide the path to your YAML config file: ").strip()
                configPath = os.path.join(configDir, userPath)
                if userPath == "q":
                    break
                elif userPath == "":
                    configPath = os.path.join(configDir, "patchcore.yaml")
                    break
                if not os.path.exists(configPath):
                    print("Error: Config file not found.")
                else:
                    break
            if userPath == "q":
                continue
            else:
                # modelName = config_path.split("/")[-1].split(".")[0]
                modelConfig = load_config(configPath)
                modelName = modelConfig.pop("model").lower()
                print(f"Model {modelName} loaded: {modelConfig}")
                
                model = create_model(modelName, modelConfig)

                print(f"Model {model} created. ")
                break
        elif choice == "2":
            # Ask for model.pt file
            model_path = input("Please provide the path to your model.pt file: ").strip()
            if not os.path.exists(model_path):
                print("Error: Model file not found.")
                return
            model = "existing_model"  # Placeholder for your model loading logic
        else:
            print("Invalid choice. Exiting.")
            return
    
    while True:
        print("Please provide the name of the training data folder in datasets. E.g.: MVTecAD (or type 'q' to quit): ")
        datasetName = input().strip()
        if datasetName.lower() == 'q':
            print("Exiting.")
            return
        elif datasetName == "":
            datasetName = "MVTecAD/train"

        dataDir = os.path.join("datasets", datasetName)

        if not os.path.isdir(dataDir):
            print("Error: Directory not found.")
        else:
            print(f"Training data directory set to: {dataDir}")
            break

    newDatasetName = input("Please provide a name for the dataset (e.g., 'MVTecAD'): ").strip()
    if newDatasetName != "":
        datasetName = newDatasetName
    print(f"Dataset name set to: {datasetName}")

    print("Importing dataset...")
    dataset, _ = importDataset(
        path=dataDir,
        name=datasetName,
        overwrite=False,
        split=["train", 'test'],
    )

    session = fo.launch_app(dataset)
    exploreData = input("Press 'e' to explore the dataset in the FiftyOne app, or any other key to continue: ").strip()
    if exploreData.lower() == 'e':    # 3. Explore data
        clipEmbedding(dataset)
        print("Finished embedding computation.")
        print("Please reload the FiftyOne app to see the new visualizations.")
        print("You find the visualizations by clicking the '+' next to Samples and choosing Embeddings.")
        # resnetEmbedding(dataset)

    # Step 2: Menu system
    while True:
        print("\nMenu:")
        print("1. Train the model")
        print("2. Retrain the model with new training parameters")
        print("3. Add data to the dataset.")
        print("4. Replace current dataset with a new dataset")
        print("5. Inference on current dataset")
        print("6. Enter continuous mode")
        print("q. Exit")

        menu_choice = input("Enter your choice [1-5]: ").strip()

        if menu_choice == "1":
            print("Training the model...")
            # Add your training logic here

            while True:
                config_name = input("Please provide the name of your training config file in the configs folder (or press Enter for default, 'q' to quit): ").strip()
                if config_name.lower() == "q":
                    print("Exiting.")
                    return
                elif config_name == "":
                    trainingConfigPath = "configs/PaDiM_Training.yaml"
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

            resultsPath = "results_training"
            if not os.path.exists(resultsPath):
                os.makedirs(resultsPath)
            logDir = "logs"
            runName = f"{modelName}-{datasetName}-{category}"
            versionName = "version_0"
            # versionName = f"version_{version}"
            if not os.path.exists(os.path.join(logDir, runName, versionName)):
                os.makedirs(os.path.join(logDir, runName, versionName))
            
            print("Logging to log directory:", os.path.join(logDir, runName))

            tblogger, callbacks, ckptPath = setupTensorboardLoggingAndCallbacks(logDir=logDir, runName=runName, versionName=versionName, version=0)
            logger = setupLogging(logDir=logDir, runName=runName, versionName=versionName)

            print("Training model...")
            rootDir = Path("tmp/datasets/data") ## root directory to store data for anomalib

            # Check if evaluator fits to training data.
            #   If data does not contain val/test data we can't use an evaluator which relies on gt data
            tmpEvaluator = deepcopy(model.evaluator)
            
            if "val" not in dataset.tags:
                valMetricsSaved = model.evaluator.val_metrics
                valMetrics = None
            else:
                valMetrics = model.evaluator.val_metrics
                valMetricsSaved = model.evaluator.val_metrics
            if "test" not in dataset.tags:
                testMetricsSaved = model.evaluator.test_metrics
                testMetrics = None
            else:
                testMetrics = model.evaluator.val_metrics
                testMetricsSaved = model.evaluator.test_metrics            
            model.evaluator = Evaluator(val_metrics=valMetrics, test_metrics=testMetrics)

            engine, datamodule, inferencer = train_and_export_model(rootDir, dataset, model, trainingConfig=trainingConfig)
            
            # Add the evaluators back to the model incase a val/test dataset is added.
            model.evaluator = tmpEvaluator
            
            ## get the test split of the dataset
            # train_split = dataset.match(F("category.label") == category).match(
            #     F("split") == "train"
            # )

            print("Running inference on dataset...")
            run_inference(dataset, inferencer, modelName)

            session = fo.launch_app(dataset)


        elif menu_choice == "2":
            # Retrain the model with new training parameters
            raise NotImplementedError
            print("Retraining the model with new parameters...")
            # Add your logic here
        elif menu_choice == "3" or menu_choice == "4":
            # Train the model with further training data
            while True:
                print("Please provide the name of the training data folder in datasets. E.g.: MVTecAD (or type 'q' to quit): ")
                datasetName = input().strip()
                if datasetName.lower() == 'q':
                    print("Exiting.")
                    return
                elif datasetName == "":
                    datasetName = "train"

                dataDir = os.path.join("datasets", datasetName)

                if not os.path.isdir(dataDir):
                    print("Error: Directory not found.")
                else:
                    print(f"Training data directory set to: {dataDir}")
                    break
            
            if menu_choice == "3":
                newDatasetName = input("Please provide a name for the merged dataset (blank input adds '_merged' to current name): ").strip()
                if newDatasetName != "":
                    datasetName = newDatasetName
                else:
                    datasetName += '_merged'
                print(f"Dataset name set to: {datasetName}")

            if menu_choice == "4":
                newDatasetName = input("Please provide a name for the new dataset (blank input keeps the current name): ").strip()
                if newDatasetName != "":
                    datasetName = newDatasetName
                print(f"Dataset name set to: {datasetName}")

            print("Importing dataset...")
            newDataset, _ = importDataset(
                path=dataDir,
                name=datasetName,
                overwrite=False,
                split=["train", 'test'],
            )

            if menu_choice == "3":
                dataset.merge_samples(newDataset)

            session = fo.launch_app(dataset)

        elif menu_choice == "5":
            print("Running inference on dataset...")
            splits = input("Select the splits to run the inference on (training, validation, test, prediction, all)\n")
            split_view = createSplitView(dataset, splits)
            run_inference(split_view, inferencer, modelName)
            session = fo.launch_app(split_view)
        elif menu_choice == "6":
            print("Entering continuous mode. In this mode the program continuously observes a folder of choice. For each image a prediction is made and the result saved.")
            print("After the processing of an image it is moved to another folder of choice such that it is not processed again.")
            input

        elif menu_choice == "q":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
