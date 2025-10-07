import os
import pandas as pd
from datetime import datetime

def collect_results(root_dir):
    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame()

    # Traverse the directory tree
    for dirpath, _, filenames in os.walk(root_dir):
        if 'results.csv' in filenames:
            # Construct the full path to the results.csv file
            csv_path = os.path.join(dirpath, 'results.csv')

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_path)

            # Extract the folder name and parse it as a datetime object
            folder_name = os.path.basename(dirpath)
            try:
                folder_datetime = datetime.strptime(folder_name, "%Y-%m-%d-%H_%M_%S")
                df['folder_datetime'] = folder_datetime
            except ValueError:
                # Handle unexpected folder name formats
                df['folder_datetime'] = None

            # Append the DataFrame to the results DataFrame
            results_df = pd.concat([results_df, df], ignore_index=True)

    return results_df

if __name__ == "__main__":
    # Specify the root directory to search for results.csv files
    root_directory = "runs/benchmark"

    # Collect the results
    results_dataframe = collect_results(root_directory)

    # Display the collected results
    print(results_dataframe)

    # Optionally, save the collected results to a new CSV file
    results_dataframe.to_csv("collected_results.csv", index=False)
