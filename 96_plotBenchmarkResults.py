import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
df = pd.read_csv('collected_results.csv')

# Group by dataset, model, and calculate the mean of the performance metrics
grouped_df = df.groupby(['dataset', 'model']).agg({
    'image_AUROC': 'mean',
    'image_F1Score': 'mean',
    'pixel_AUROC': 'mean',
    'pixel_F1Score': 'mean'
}).reset_index()

# Group by dataset, model, and calculate the mean of the timing metrics
grouped_time_df = df.groupby(['dataset', 'model']).agg({
    'job_duration': 'mean',
    'fit_duration': 'mean',
    'test_duration': 'mean'
}).reset_index()


# Melt the DataFrame for performance metrics
melted_performance_df = pd.melt(grouped_df, id_vars=['dataset', 'model'], value_vars=['image_AUROC', 'image_F1Score', 'pixel_AUROC', 'pixel_F1Score'],
                                var_name='metric', value_name='value')

# Set the style of the plots
sns.set(style="whitegrid")

# Get unique datasets
datasets = melted_performance_df['dataset'].unique()

# Melt the DataFrame for time metrics
melted_time_df = pd.melt(grouped_time_df, id_vars=['dataset', 'model'], value_vars=['job_duration', 'fit_duration', 'test_duration'],
                         var_name='metric', value_name='value')

# Create a performance metric plot for each dataset
for dataset in datasets:
    plt.figure(figsize=(12, 8))
    dataset_performance_df = melted_performance_df[melted_performance_df['dataset'] == dataset]
    sns.barplot(x='metric', y='value', hue='model', data=dataset_performance_df)
    plt.title(f'Average Performance Metrics for Dataset: {dataset}')
    plt.xlabel('Metric')
    plt.ylabel('Average Value')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'performance_metrics_{dataset}.png', bbox_inches='tight')
    # Create timing metric plots for each dataset
    plt.figure(figsize=(12, 8))
    dataset_time_df = melted_time_df[melted_time_df['dataset'] == dataset]
    sns.barplot(x='metric', y='value', hue='model', data=dataset_time_df)
    plt.title(f'Average Timing Metrics for Dataset: {dataset}')
    plt.xlabel('Metric')
    plt.ylabel('Average Duration (seconds)')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'timing_metrics_{dataset}.png', bbox_inches='tight')