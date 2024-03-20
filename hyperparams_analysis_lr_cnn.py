import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and aggregate data from multiple CSV files
def load_and_aggregate_data(files_path, relevant_columns):
    all_dfs = []  # List to hold all DataFrame objects
    for file in glob.glob(files_path):
        df = pd.read_csv(file)
        df = df[relevant_columns]  # Extract relevant columns
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

# File path patterns for LR and CNN data
lr_files_path = "./results/gridsearch_detail_results_lr_*.csv"
cnn_files_path = "./results/optuna_results_*.csv"

# Load and aggregate data
lr_data = load_and_aggregate_data(lr_files_path, ["param_C", "param_penalty", "mean_test_MCC"])
cnn_data = load_and_aggregate_data(cnn_files_path, ["params_kernel_sizes", "params_optimizer", "value"])

# Visualization setup
sns.set_context("talk")  # This makes labels and titles more readable
sns.set_style("whitegrid")  # Adds a grid for better readability, but keeps the background white for cleanliness
fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # Adjusted figsize for better spacing and readability

# LR Heatmap
lr_heatmap = lr_data.pivot_table(index="param_penalty", columns="param_C", values="mean_test_MCC", aggfunc="mean")
sns.heatmap(lr_heatmap, ax=axs[0], annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
axs[0].set_title("Logistic Regression: C vs Penalty", fontsize=20)
axs[0].set_xlabel("C Value", fontsize=18)
axs[0].set_ylabel("Penalty Type", fontsize=18)
axs[0].tick_params(axis='both', which='major', labelsize=16)

# CNN Heatmap
cnn_heatmap = cnn_data.pivot_table(index="params_kernel_sizes", columns="params_optimizer", values="value", aggfunc="mean")
sns.heatmap(cnn_heatmap, ax=axs[1], annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
axs[1].set_title("CNN: Kernel Sizes vs Optimizer", fontsize=20)
axs[1].set_xlabel("Optimizer", fontsize=18)
axs[1].set_ylabel("Kernel Sizes", fontsize=18)
axs[1].tick_params(axis='both', which='major', labelsize=16)

plt.savefig("./results/hyperparams_analysis_lr_cnn.png", dpi=300, bbox_inches="tight")

# Save LR and CNN heatmap data to a CSV file
lr_heatmap.to_csv("./results/hyperparams_analysis_lr.csv")
cnn_heatmap.to_csv("./results/hyperparams_analysis_cnn.csv")