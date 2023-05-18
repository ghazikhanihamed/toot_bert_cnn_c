import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
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
import re

DATASETS = ["balanced", "imbalanced"]

# Define a function to compute the p-value


def compute_p_value(mean1, std1, n1, mean2, std2, n2):
    se1 = std1 / np.sqrt(n1)
    se2 = std2 / np.sqrt(n2)
    sed = np.sqrt(se1**2.0 + se2**2.0)
    t_stat = (mean1 - mean2) / sed
    df = n1 + n2 - 2
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return p


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
            _x = (p1.get_x() + p1.get_width() / 2 +
                  p2.get_x() + p2.get_width() / 2) / 2
            _y = max(p1.get_height(), p2.get_height()) + 0.02
            value = f'{delta_values[i]:.2f}'
            ax.text(_x, _y + 0.03, f'Δ = {value}',
                    ha="center", fontsize=11, color='red')

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def extract_mean(value):
    return float(value.split("±")[0])


def extract_std(value):
    return float(value.split("±")[1])

# Define a function to extract the size from the 'PLM' column
def extract_size(text):
    match = re.search(r'\((\d+)([MB])\)', text)
    if match:
        size = int(match.group(1))
        suffix = match.group(2)
        
        if suffix == 'M':
            size *= 1_000_000  # Convert to millions
        elif suffix == 'B':
            size *= 1_000_000_000  # Convert to billions
        
        return size
    return 0

pd.set_option('display.float_format', '{:.2e}'.format)

representation_types = [FROZEN, FINETUNED]
precision_types = ["half", "full"]
representers = REPRESENTATIONS
tasks = TASKS
datasets = ["balanced", "imbalanced"]

# We read the csv file of the full results
df = pd.read_csv(os.path.join(
    RESULTS_PATH, "mean_balanced_imbalanced_results_trad_cnn.csv"))
# Filter out the task "ionchannels_iontransporters"
# df = df[df['Task'] != 'ionchannels_iontransporters']


df['MCC_mean'] = df['MCC'].apply(extract_mean)
df['MCC_std'] = df['MCC'].apply(extract_std)
df['Accuracy_mean'] = df['Accuracy'].apply(extract_mean)
df['Accuracy_std'] = df['Accuracy'].apply(extract_std)
df['Sensitivity_mean'] = df['Sensitivity'].apply(extract_mean)
df['Sensitivity_std'] = df['Sensitivity'].apply(extract_std)
df['Specificity_mean'] = df['Specificity'].apply(extract_mean)
df['Specificity_std'] = df['Specificity'].apply(extract_std)
df["plm_size"] = df["Representer"].apply(lambda x: PLM_PARAM_SIZE[x])
df['task_short'] = df['Task'].apply(lambda x: TASKS_SHORT[x])

# --------------------------------------------- Representer

# We will compute the p-value for each task between the different representers
pvalue_dict = {}
for task in df['task_short'].unique():
    protbert = df[(df['task_short'] == task) & (df['Representer'] == 'ProtBERT')]
    protbert_bfd = df[(df['task_short'] == task) & (df['Representer'] == 'ProtBERT-BFD')]
    prott5 = df[(df['task_short'] == task) & (df['Representer'] == 'ProtT5')]
    esm1b = df[(df['task_short'] == task) & (df['Representer'] == 'ESM-1b')]
    esm2 = df[(df['task_short'] == task) & (df['Representer'] == 'ESM-2')]
    esm2_15b = df[(df['task_short'] == task) & (df['Representer'] == 'ESM-2_15B')]

    # Ensure the rows are in the same order
    protbert = protbert.sort_index()
    protbert_bfd = protbert_bfd.sort_index()
    prott5 = prott5.sort_index()
    esm1b = esm1b.sort_index()
    esm2 = esm2.sort_index()
    esm2_15b = esm2_15b.sort_index()


    # Compute the p-value
    _, p_val = stats.f_oneway(protbert['MCC_mean'], protbert_bfd['MCC_mean'], prott5['MCC_mean'], esm1b['MCC_mean'], esm2['MCC_mean'], esm2_15b['MCC_mean'])
    pvalue_dict[task] = p_val


# Group by Task and Dataset, and calculate the mean MCC
mean_mcc = df.groupby(['task_short', 'plm_size'])[
    'MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['task_short', 'plm_size'])['MCC_std'].mean().reset_index()

mean_accuracy = df.groupby(['task_short', 'plm_size'])[
    'Accuracy_mean'].mean().reset_index()
std_accuracy = df.groupby(['task_short', 'plm_size'])[
    'Accuracy_std'].mean().reset_index()

mean_sensitivity = df.groupby(['task_short', 'plm_size'])[
    'Sensitivity_mean'].mean().reset_index()
std_sensitivity = df.groupby(['task_short', 'plm_size'])[
    'Sensitivity_std'].mean().reset_index()

mean_specificity = df.groupby(['task_short', 'plm_size'])[
    'Specificity_mean'].mean().reset_index()
std_specificity = df.groupby(['task_short', 'plm_size'])[
    'Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['task_short', 'plm_size'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['task_short', 'plm_size'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity,
                       on=['task_short', 'plm_size'])
specificity = pd.merge(mean_specificity, std_specificity,
                       on=['task_short', 'plm_size'])

# Merge all metric dataframes
df_metrics = pd.concat([mcc, accuracy, sensitivity, specificity], axis=1)

# Remove duplicate columns
df_metrics = df_metrics.loc[:, ~df_metrics.columns.duplicated()]

# Create a new column for each metric as mean±std
df_metrics['MCC'] = df_metrics['MCC_mean'].map(
    '{:.2f}'.format) + '±' + df_metrics['MCC_std'].map('{:.2f}'.format)
df_metrics['Accuracy'] = df_metrics['Accuracy_mean'].map(
    '{:.2f}'.format) + '±' + df_metrics['Accuracy_std'].map('{:.2f}'.format)
df_metrics['Sensitivity'] = df_metrics['Sensitivity_mean'].map(
    '{:.2f}'.format) + '±' + df_metrics['Sensitivity_std'].map('{:.2f}'.format)
df_metrics['Specificity'] = df_metrics['Specificity_mean'].map(
    '{:.2f}'.format) + '±' + df_metrics['Specificity_std'].map('{:.2f}'.format)

# Select final columns for the table
df_metrics = df_metrics[['task_short', 'plm_size',
                         'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]
# Rename columns: Task, PLM, MCC, Accuracy, Sensitivity, Specificity
df_metrics = df_metrics.rename(columns={'task_short': 'Task', 'plm_size': 'PLM'})

# We change the order of the tasks as IC-MP, IT-MP, and IC-IT
df_metrics['Task'] = df_metrics['Task'].astype(
    'category').cat.reorder_categories(['IC-MP', 'IT-MP', 'IC-IT'], ordered=True)
df_metrics = df_metrics.sort_values('Task')

# Apply the function to create a new 'Size' column
df_metrics['Size'] = df_metrics['PLM'].apply(extract_size)

# Sort the dataframe by 'Size' column while maintaining the order of the 'Task' column
sorted_df = df_metrics.sort_values(by=['Task', 'Size'])

# Remove the 'Size' column if not needed anymore
df_metrics = sorted_df.drop('Size', axis=1)

# Add the p-values to the table
df_metrics['P-value'] = df_metrics['Task'].map(pvalue_dict).map('{:.2e}'.format)

# Generate LaTeX table
latex_table = df_metrics.to_latex(
    index=False, float_format="%.2f", escape=False, column_format='lcccccc')

# save the table to a file
with open(os.path.join(LATEX_PATH, "mean_task_plm_results_cnn.tex"), "w") as f:
    f.write(latex_table)

# --------------------------------------------- Classifier

# Compute the p-value for each task between the classifiers
pvalue_dict = {}
for task in df['task_short'].unique():
    svm = df[(df['task_short'] == task) & (df['Classifier'] == 'SVM')]
    rf = df[(df['task_short'] == task) & (df['Classifier'] == 'RF')]
    lr = df[(df['task_short'] == task) & (df['Classifier'] == 'LR')]
    ffnn = df[(df['task_short'] == task) & (df['Classifier'] == 'FFNN')]
    cnn = df[(df['task_short'] == task) & (df['Classifier'] == 'CNN')]
    knn = df[(df['task_short'] == task) & (df['Classifier'] == 'kNN')]

    # Ensure the rows are in the same order
    svm = svm.sort_index()
    rf = rf.sort_index()
    lr = lr.sort_index()
    ffnn = ffnn.sort_index()
    cnn = cnn.sort_index()
    knn = knn.sort_index()

    # Compute the p-value for each task between the classifiers
    _, p_val = stats.f_oneway(svm['MCC_mean'], rf['MCC_mean'], lr['MCC_mean'], ffnn['MCC_mean'], cnn['MCC_mean'], knn['MCC_mean'])

    # Add the p-value to the dictionary
    pvalue_dict[task] = p_val


# Group by Task and Classifier, and calculate the mean MCC
mean_mcc = df.groupby(['task_short', 'Classifier'])[
    'MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['task_short', 'Classifier'])['MCC_std'].mean().reset_index()

mean_accuracy = df.groupby(['task_short', 'Classifier'])[
    'Accuracy_mean'].mean().reset_index()
std_accuracy = df.groupby(['task_short', 'Classifier'])[
    'Accuracy_std'].mean().reset_index()

mean_sensitivity = df.groupby(['task_short', 'Classifier'])[
    'Sensitivity_mean'].mean().reset_index()
std_sensitivity = df.groupby(['task_short', 'Classifier'])[
    'Sensitivity_std'].mean().reset_index()

mean_specificity = df.groupby(['task_short', 'Classifier'])[
    'Specificity_mean'].mean().reset_index()
std_specificity = df.groupby(['task_short', 'Classifier'])[
    'Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['task_short', 'Classifier'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['task_short', 'Classifier'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity,
                       on=['task_short', 'Classifier'])
specificity = pd.merge(mean_specificity, std_specificity,
                       on=['task_short', 'Classifier'])

# Merge all metric dataframes
df_metrics = pd.concat([mcc, accuracy, sensitivity, specificity], axis=1)

# Remove duplicate columns
df_metrics = df_metrics.loc[:, ~df_metrics.columns.duplicated()]

# Create a new column for each metric as mean±std
df_metrics['MCC'] = df_metrics['MCC_mean'].map(
    '{:.2f}'.format) + '±' + df_metrics['MCC_std'].map('{:.2f}'.format)
df_metrics['Accuracy'] = df_metrics['Accuracy_mean'].map(
    '{:.2f}'.format) + '±' + df_metrics['Accuracy_std'].map('{:.2f}'.format)
df_metrics['Sensitivity'] = df_metrics['Sensitivity_mean'].map(
    '{:.2f}'.format) + '±' + df_metrics['Sensitivity_std'].map('{:.2f}'.format)
df_metrics['Specificity'] = df_metrics['Specificity_mean'].map(
    '{:.2f}'.format) + '±' + df_metrics['Specificity_std'].map('{:.2f}'.format)

# Select final columns for the table
df_metrics = df_metrics[['task_short', 'Classifier',
                         'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]
df_metrics = df_metrics.rename(columns={'task_short': 'Task'})

# We change the order of the tasks as IC-MP, IT-MP, and IC-IT
df_metrics['Task'] = df_metrics['Task'].astype(
    'category').cat.reorder_categories(['IC-MP', 'IT-MP', 'IC-IT'], ordered=True)
df_metrics = df_metrics.sort_values('Task')

# Define the custom sorting order for the 'Classifier' column
custom_order = ['LR', 'kNN', 'RF', 'SVM', 'FFNN', 'CNN']

# Convert the 'Classifier' column to categorical data with the custom order
df_metrics['Classifier'] = pd.Categorical(df_metrics['Classifier'], categories=custom_order, ordered=True)

# Sort the dataframe by 'Task' and 'Classifier'
df_metrics = df_metrics.sort_values(by=['Task', 'Classifier'])

# Add the p-values to the table
df_metrics['P-value'] = df_metrics['Task'].map(pvalue_dict).map('{:.2e}'.format)

# Generate LaTeX table
latex_table = df_metrics.to_latex(
    index=False, float_format="%.2f", escape=False, column_format='lccccccc')

# save the table to a file
with open(os.path.join(LATEX_PATH, "mean_task_classifier_results_cnn.tex"), "w") as f:
    f.write(latex_table)