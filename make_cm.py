import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the style and font scale for better readability
sns.set(style="white", font_scale=1.1)

folder = "./model_performance_results_old_new"
results_df = pd.read_csv(f"{folder}/model_performance_results_old_new.csv")

def plot_confusion_matrix(df, task, dataset_type, ax):
    task_df = df[(df["Task"] == task) & (df["Dataset_Type"] == dataset_type)]

    TP = task_df["TP"].values[0]
    TN = task_df["TN"].values[0]
    FP = task_df["FP"].values[0]
    FN = task_df["FN"].values[0]

    confusion_mtx = np.array([[TP, FN], [FP, TN]])

    # Define class labels for each task
    if task == "IC-MP":
        labels = ["IC", "MP"]
    elif task == "IT-MP":
        labels = ["IT", "MP"]
    else:  # IC-IT
        labels = ["IC", "IT"]

    # Use a color map that offers good contrast
    cmap = sns.light_palette("blue", as_cmap=True)

    sns.heatmap(
        confusion_mtx,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        xticklabels=[f"Predicted {labels[0]}", f"Predicted {labels[1]}"],
        yticklabels=[f"Actual {labels[0]}", f"Actual {labels[1]}"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold", "color": "black"}
    )

    # Use descriptive names for datasets
    dataset_name = "Novel Dataset" if dataset_type == "new" else "Taju et al. Dataset"
    ax.set_title(f"{task} - {dataset_name}", fontsize=16, weight="bold")

# Adjust figure size for smaller boxes and overall layout
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12), dpi=100)

tasks = ["IC-MP", "IT-MP", "IC-IT"]
dataset_types = ["old", "new"]

for i, task in enumerate(tasks):
    for j, dataset_type in enumerate(dataset_types):
        plot_confusion_matrix(results_df, task, dataset_type, axes[i, j])

plt.tight_layout()

# Save the plot with improvements
plt.savefig(f"{folder}/confusion_matrices_refined.png")
