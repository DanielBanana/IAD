import os
import yaml

import fiftyone as fo
import fiftyone.core.dataset as fod
import fiftyone.zoo as foz # zoo datasets and models

from anomalib.metrics import Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import AUROC, AUPR, F1AdaptiveThreshold, F1Score

from torchvision import transforms
from AnomalyDataset import AnomalyImageTreeImporter, importAnomalyDataset
from setup import create_datamodule, create_model, setupTensorboardLoggingAndCallbacks, setupLogging, define_metrics
from Configs import load_config
from Visualisation import clipEmbedding, resnetEmbedding


def getDefaultEvaluator():
    val_metrics, test_metrics = define_metrics()
    evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
    return evaluator

def main():
    print("Welcome to the ML Model CLI!")

    # Step 1: Ask user if they want to train a new model or use an existing one
    choice = input("Do you want to train a new model (1) or use an existing model (2)? [1/2]: ").strip()

    while True:
        if choice == "1":
            # Ask for YAML config file
            while True:
                config_path = input("Please provide the path to your YAML config file: ").strip()
                if config_path == "q":
                    break
                elif config_path == "":
                    config_path = "configs/PaDiM.yaml"
                    break
                elif not os.path.exists(config_path):
                    print("Error: Config file not found.")
                else:
                    break
            if config_path == "q":
                continue
            else:
                config = load_config(config_path)
                print(f"Config loaded: {config}")
                model = "new_model"  # Placeholder for your model training logic
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
        print("Please provide the name of the training data in folder datasets. E.g.: MVTecAD (or type 'q' to quit): ")
        datasetName = input().strip()
        if datasetName.lower() == 'q':
            print("Exiting.")
            return
        elif datasetName == "":
            datasetName = "MVTecAD"

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
    importer = AnomalyImageTreeImporter(
        dataset_dir=dataDir,
        compute_metadata=True,
        classes=None,
        unlabeled="_unlabeled",
        shuffle=False,
        seed=None,
        max_samples=None,
    )

    dataset = fod.Dataset(name=datasetName, overwrite=False)
    dataset, info = importAnomalyDataset(dataset, importer)

    exploreData = input("Press 'e' to explore the dataset in the FiftyOne app, or any other key to continue: ").strip()
    if exploreData.lower() == 'e':    # 3. Explore data
        session = fo.launch_app(dataset)

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
        print("3. Train the model with further training data")
        print("4. Validate the model")
        print("q. Exit")

        menu_choice = input("Enter your choice [1-5]: ").strip()

        if menu_choice == "1":
            # Train the model
            print("Training the model...")
            # Add your training logic here
        elif menu_choice == "2":
            # Retrain the model with new training parameters
            print("Retraining the model with new parameters...")
            # Add your logic here
        elif menu_choice == "3":
            # Train the model with further training data
            dataDir = input("Please provide the directory with further training data: ").strip()
            if not os.path.isdir(dataDir):
                print("Error: Directory not found.")
                continue
            print(f"Training with data from {dataDir}...")
            # Add your logic here
        elif menu_choice == "4":
            # Validate the model
            val_dir = input("Please provide the directory with validation data: ").strip()
            if not os.path.isdir(val_dir):
                print("Error: Directory not found.")
                continue
            print(f"Validating model with data from {val_dir}...")
            # Add your logic here
        elif menu_choice == "q":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
