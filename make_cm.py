import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # Ensure numpy is imported

folder = "./model_performance_results_old_new"
results_df = pd.read_csv(f"{folder}/model_performance_results_old_new.csv")

def plot_confusion_matrix(df, task, dataset_type, ax):
    task_df = df[(df['Task'] == task) & (df['Dataset_Type'] == dataset_type)]
    
    TP = task_df['TP'].values[0]
    TN = task_df['TN'].values[0]
    FP = task_df['FP'].values[0]
    FN = task_df['FN'].values[0]
    
    # Use np.nan for the bottom-right cell
    confusion_mtx = np.array([[TP, FN, TP + FN], [FP, TN, FP + TN], [TP + FP, FN + TN, np.nan]])

    # Plot confusion matrix with 'annot_kws' to handle np.nan
    sns.heatmap(confusion_mtx, annot=True, fmt='.0f', cmap='Blues', cbar=False,
                xticklabels=['Predicted IC', 'Predicted MP', 'Total'],
                yticklabels=['Actual IC', 'Actual MP', 'Total'], ax=ax, annot_kws={"ha": 'center', "va": 'center'})
    
    ax.set_title(f'{task} - {dataset_type} Dataset')

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))
fig.suptitle('Confusion Matrices for Baseline and New Datasets')

tasks = ['IC-MP', 'IT-MP', 'IC-IT']
dataset_types = ['old', 'new']

for i, task in enumerate(tasks):
    for j, dataset_type in enumerate(dataset_types):
        plot_confusion_matrix(results_df, task, dataset_type, axes[i, j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
