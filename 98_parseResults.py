import os
import pandas as pd
import argparse

def parse_results(results_path):
    # Initialize a list to store the results
    results = []

    # Walk through the directory
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if file.endswith('_results.txt'):
                file_path = os.path.join(root, file)
                # Extract modelName from the filename
                model_name = file.replace('_results.txt', '')
                with open(file_path, 'r') as f:
                    # Read the file and parse the metrics
                    content = f.readlines()
                    metrics = {'Model': model_name}
                    for line in content:
                        if ':' in line:
                            key, value = line.strip().split(':')
                            metrics[key.strip()] = value.strip()

                    # Append the results to the list
                    results.append(metrics)

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Parse results from anomaly detection model runs.")
    parser.add_argument("--dataset", type=str, default="mvtecad", help="Which dataset's results to parse")

    # Parse the arguments
    args = parser.parse_args()

    # Define the results path
    results_path = os.path.join("results", args.dataset)

    # Parse the results
    results_df = parse_results(results_path)

    # Print the results DataFrame
    print(results_df)

    # Optionally, save the DataFrame to a CSV file
    results_df.to_csv(os.path.join(results_path, 'compiled_results.csv'), index=False)
