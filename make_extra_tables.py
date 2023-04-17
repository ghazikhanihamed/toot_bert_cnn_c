import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from settings.settings import (
    RESULTS_PATH,
    FROZEN,
    FINETUNED,
    REPRESENTATIONS,
    TASKS,
    TASKS_SHORT,
    PLM_PARAM_SIZE,
    LATEX_PATH,
    PLM_ORDER,
    PLM_ORDER_FINETUNED
)

DATASETS = ["balanced", "imbalanced"]

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = f'{p.get_height():.2f}'
            ax.text(_x, _y - 0.1, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def show_delta_on_bars(axs, delta_values):
    def _show_on_single_plot(ax):
        half_bar_count = len(ax.patches) // 2
        for i in range(half_bar_count):
            p1 = ax.patches[i]
            p2 = ax.patches[i + half_bar_count]
            _x = (p1.get_x() + p1.get_width() / 2 + p2.get_x() + p2.get_width() / 2) / 2
            _y = max(p1.get_height(), p2.get_height()) + 0.02
            value = f'{delta_values[i]:.2f}'
            ax.text(_x, _y + 0.03, f'Δ = {value}', ha="center", fontsize=11, color='red')

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def extract_mean(value):
    return float(value.split("±")[0])

def extract_std(value):
    return float(value.split("±")[1])
    
representation_types = [FROZEN, FINETUNED]
precision_types = ["half", "full"]
representers = REPRESENTATIONS
tasks = TASKS
datasets = ["balanced", "imbalanced"]

# We read the csv file of the full results
df = pd.read_csv(os.path.join(RESULTS_PATH, "mean_balanced_imbalanced_results.csv"))
# Filter out the task "ionchannels_iontransporters"
# df = df[df['Task'] != 'ionchannels_iontransporters']

balanced_df = df[df['Dataset'] == 'balanced']
imbalanced_df = df[df['Dataset'] == 'imbalanced']
df['MCC_mean'] = df['MCC'].apply(extract_mean)
df['MCC_std'] = df['MCC'].apply(extract_std)
df["plm_size"] = df["Representer"].apply(lambda x: PLM_PARAM_SIZE[x])
df['Task'] = df['Task'].apply(lambda x: TASKS_SHORT[x])

# Create the violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='plm_size', y='MCC_mean', hue='Representation', data=df, inner='quartile', palette='Set2', order=PLM_ORDER)

# Set plot title and labels
# plt.title('Violin plot of MCC scores for different Tasks, Representers, and Classifiers')
plt.xlabel('PLM')
plt.ylabel('MCC Score')

# Adjust legend
plt.legend(title='Task', loc='lower center')

# Show the plot
plt.show()