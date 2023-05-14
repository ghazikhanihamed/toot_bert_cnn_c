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
            _x = (p1.get_x() + p1.get_width() / 2 + p2.get_x() + p2.get_width() / 2) / 2
            _y = max(p1.get_height(), p2.get_height()) + 0.02
            if np.isnan(_y):
                continue
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
df = pd.read_csv(os.path.join(RESULTS_PATH, "mean_balanced_imbalanced_results_trad_cnn.csv"))

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

# --------------------------------------------- TASKS

# Group by Task and Dataset, and calculate the mean MCC
mean_mcc = df.groupby(['task_short', 'Representation'])['MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['task_short', 'Representation'])['MCC_std'].mean().reset_index()

mean_accuracy = df.groupby(['task_short', 'Representation'])['Accuracy_mean'].mean().reset_index()
std_accuracy = df.groupby(['task_short', 'Representation'])['Accuracy_std'].mean().reset_index()

mean_sensitivity = df.groupby(['task_short', 'Representation'])['Sensitivity_mean'].mean().reset_index()
std_sensitivity = df.groupby(['task_short', 'Representation'])['Sensitivity_std'].mean().reset_index()

mean_specificity = df.groupby(['task_short', 'Representation'])['Specificity_mean'].mean().reset_index()
std_specificity = df.groupby(['task_short', 'Representation'])['Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['task_short', 'Representation'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['task_short', 'Representation'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity, on=['task_short', 'Representation'])
specificity = pd.merge(mean_specificity, std_specificity, on=['task_short', 'Representation'])

# Merge all metric dataframes
df_metrics = pd.concat([mcc, accuracy, sensitivity, specificity], axis=1)

# Remove duplicate columns
df_metrics = df_metrics.loc[:,~df_metrics.columns.duplicated()]

# Create a new column for each metric as mean±std
df_metrics['MCC'] = df_metrics['MCC_mean'].map('{:.2f}'.format) + '±' + df_metrics['MCC_std'].map('{:.2f}'.format)
df_metrics['Accuracy'] = df_metrics['Accuracy_mean'].map('{:.2f}'.format) + '±' + df_metrics['Accuracy_std'].map('{:.2f}'.format)
df_metrics['Sensitivity'] = df_metrics['Sensitivity_mean'].map('{:.2f}'.format) + '±' + df_metrics['Sensitivity_std'].map('{:.2f}'.format)
df_metrics['Specificity'] = df_metrics['Specificity_mean'].map('{:.2f}'.format) + '±' + df_metrics['Specificity_std'].map('{:.2f}'.format)

# Select final columns for the table
df_metrics = df_metrics[['task_short', 'Representation', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]
df_metrics = df_metrics.rename(columns={'task_short': 'Task'})

# Generate LaTeX table
latex_table = df_metrics.to_latex(index=False, float_format="%.2f", escape=False, column_format='lccccc')

# save the table to a file
with open(os.path.join(LATEX_PATH, "mean_frozen_finetuned_results_task_cnn.tex"), "w") as f:
    f.write(latex_table)

delta_mcc = mean_mcc[mean_mcc['Representation'] == 'finetuned']['MCC_mean'].values - mean_mcc[mean_mcc['Representation'] == 'frozen']['MCC_mean'].values

# Create the bar plot
plt.figure(figsize=(10, 4))
barplot = sns.barplot(
    x='task_short',
    y='MCC_mean',
    hue='Representation',
    data=mean_mcc,
    ci=None,
    palette="colorblind"
)

# X and Y axis labels
plt.xlabel("Task")
plt.ylabel("MCC")

# Add legend
plt.legend(title="Representation")

# Add error bars
for i, p in enumerate(barplot.patches):
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    err = std_mcc.loc[i, 'MCC_std']
    plt.errorbar(x, y, yerr=err, capsize=3, elinewidth=1.5, color='black', ls='none')

# Add MCC values on top of each bar
show_values_on_bars(barplot)

show_delta_on_bars(barplot, delta_mcc)
# Display the plot
# plt.show()
plt.savefig(os.path.join(LATEX_PATH, "mean_frozen_finetuned_results_task_cnn.png"), bbox_inches='tight', dpi=300)
plt.close()

# ---------------------------------------------

# Group by Classifier and Representation, and calculate the mean MCC
mean_mcc = df.groupby(['Classifier', 'Representation'])['MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['Classifier', 'Representation'])['MCC_std'].mean().reset_index()

mean_accuracy = df.groupby(['Classifier', 'Representation'])['Accuracy_mean'].mean().reset_index()
std_accuracy = df.groupby(['Classifier', 'Representation'])['Accuracy_std'].mean().reset_index()

mean_sensitivity = df.groupby(['Classifier', 'Representation'])['Sensitivity_mean'].mean().reset_index()
std_sensitivity = df.groupby(['Classifier', 'Representation'])['Sensitivity_std'].mean().reset_index()

mean_specificity = df.groupby(['Classifier', 'Representation'])['Specificity_mean'].mean().reset_index()
std_specificity = df.groupby(['Classifier', 'Representation'])['Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['Classifier', 'Representation'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['Classifier', 'Representation'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity, on=['Classifier', 'Representation'])
specificity = pd.merge(mean_specificity, std_specificity, on=['Classifier', 'Representation'])

# Merge all metric dataframes
df_metrics = pd.concat([mcc, accuracy, sensitivity, specificity], axis=1)

# Remove duplicate columns
df_metrics = df_metrics.loc[:,~df_metrics.columns.duplicated()]

# Create a new column for each metric as mean±std
df_metrics['MCC'] = df_metrics['MCC_mean'].map('{:.2f}'.format) + '±' + df_metrics['MCC_std'].map('{:.2f}'.format)
df_metrics['Accuracy'] = df_metrics['Accuracy_mean'].map('{:.2f}'.format) + '±' + df_metrics['Accuracy_std'].map('{:.2f}'.format)
df_metrics['Sensitivity'] = df_metrics['Sensitivity_mean'].map('{:.2f}'.format) + '±' + df_metrics['Sensitivity_std'].map('{:.2f}'.format)
df_metrics['Specificity'] = df_metrics['Specificity_mean'].map('{:.2f}'.format) + '±' + df_metrics['Specificity_std'].map('{:.2f}'.format)

# Select final columns for the table
df_metrics = df_metrics[['Classifier', 'Representation', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]

# Generate LaTeX table
latex_table = df_metrics.to_latex(index=False, float_format="%.2f", escape=False, column_format='lccccc')

# save the table to a file
with open(os.path.join(LATEX_PATH, "mean_frozen_finetuned_results_classifier_cnn.tex"), "w") as f:
    f.write(latex_table)


delta_mcc = mean_mcc[mean_mcc['Representation'] == 'finetuned']['MCC_mean'].values - mean_mcc[mean_mcc['Representation'] == 'frozen']['MCC_mean'].values


# Create the bar plot
plt.figure(figsize=(10, 4))
barplot = sns.barplot(
    x='Classifier',
    y='MCC_mean',
    hue='Representation',
    data=mean_mcc,
    ci=None,
    palette="colorblind"
)

# X and Y axis labels
plt.xlabel("Classifier")
plt.ylabel("MCC")

# Add legend
plt.legend(title="Representation")

# Add error bars
for i, p in enumerate(barplot.patches):
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    err = std_mcc.loc[i, 'MCC_std']
    plt.errorbar(x, y, yerr=err, capsize=3, elinewidth=1.5, color='black', ls='none')

# Add Sensitivity values on top of each bar
show_values_on_bars(barplot)

show_delta_on_bars(barplot, delta_mcc)

# Display the plot
# plt.show()
plt.savefig(os.path.join(LATEX_PATH, "mean_frozen_finetuned_results_classifier_cnn.png"), bbox_inches='tight', dpi=300)
plt.close()


# Group by Representer and Dataset, and calculate the mean MCC
mean_mcc = df.groupby(['plm_size', 'Representation'])['MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['plm_size', 'Representation'])['MCC_std'].mean().reset_index()

mean_accuracy = df.groupby(['plm_size', 'Representation'])['Accuracy_mean'].mean().reset_index()
std_accuracy = df.groupby(['plm_size', 'Representation'])['Accuracy_std'].mean().reset_index()

mean_sensitivity = df.groupby(['plm_size', 'Representation'])['Sensitivity_mean'].mean().reset_index()
std_sensitivity = df.groupby(['plm_size', 'Representation'])['Sensitivity_std'].mean().reset_index()

mean_specificity = df.groupby(['plm_size', 'Representation'])['Specificity_mean'].mean().reset_index()
std_specificity = df.groupby(['plm_size', 'Representation'])['Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['plm_size', 'Representation'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['plm_size', 'Representation'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity, on=['plm_size', 'Representation'])
specificity = pd.merge(mean_specificity, std_specificity, on=['plm_size', 'Representation'])

# Merge all metric dataframes
df_metrics = pd.concat([mcc, accuracy, sensitivity, specificity], axis=1)

# Remove duplicate columns
df_metrics = df_metrics.loc[:,~df_metrics.columns.duplicated()]

# Create a new column for each metric as mean±std
df_metrics['MCC'] = df_metrics['MCC_mean'].map('{:.2f}'.format) + '±' + df_metrics['MCC_std'].map('{:.2f}'.format)
df_metrics['Accuracy'] = df_metrics['Accuracy_mean'].map('{:.2f}'.format) + '±' + df_metrics['Accuracy_std'].map('{:.2f}'.format)
df_metrics['Sensitivity'] = df_metrics['Sensitivity_mean'].map('{:.2f}'.format) + '±' + df_metrics['Sensitivity_std'].map('{:.2f}'.format)
df_metrics['Specificity'] = df_metrics['Specificity_mean'].map('{:.2f}'.format) + '±' + df_metrics['Specificity_std'].map('{:.2f}'.format)

# Select final columns for the table
df_metrics = df_metrics[['plm_size', 'Representation', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]
df_metrics = df_metrics.sort_values(by=['plm_size'], ascending=True)

# Generate LaTeX table
latex_table = df_metrics.to_latex(index=False, float_format="%.2f", escape=False, column_format='lccccc')

# save the table to a file
with open(os.path.join(LATEX_PATH, "mean_frozen_finetuned_results_plm_cnn.tex"), "w") as f:
    f.write(latex_table)

delta_mcc = mean_mcc[mean_mcc['Representation'] == 'finetuned'][['plm_size', 'MCC_mean']].reset_index(drop=True)
frozen_mcc_values = mean_mcc[mean_mcc['Representation'] == 'frozen'][['plm_size', 'MCC_mean']].reset_index(drop=True)
delta_mcc['MCC_mean'] = [finetuned - frozen_mcc_values.loc[frozen_mcc_values['plm_size'] == plm_size, 'MCC_mean'].values[0] 
                         for plm_size, finetuned in zip(delta_mcc['plm_size'], delta_mcc['MCC_mean'])]

# We sort the values by plm_size the same as in PLM_ORDER_FINETUNED list
delta_mcc['plm_size'] = pd.Categorical(delta_mcc['plm_size'], categories=PLM_ORDER_FINETUNED, ordered=True)
# Sort delta_mcc based on the plm_size column with the specified order
delta_mcc_sorted = delta_mcc.sort_values('plm_size')
# Reset the index of the sorted DataFrame
delta_mcc_sorted.reset_index(drop=True, inplace=True)


# Create the bar plot
plt.figure(figsize=(10, 4))
ax = barplot = sns.barplot(
    x='plm_size',
    y='MCC_mean',
    hue='Representation',
    data=mean_mcc,
    ci=None,
    palette="colorblind",
    order=PLM_ORDER
)

# X and Y axis labels
plt.xlabel("PLM (parameter size)")
plt.ylabel("MCC")

plt.xticks(fontsize=9)

# Add legend
plt.legend(title="Representation", loc='lower left')

# Add MCC values on top of each bar
show_values_on_bars(barplot)

# Add error bars
bars = ax.containers
i = 0
for bar in bars[0].get_children() + bars[1].get_children():
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    # if y is nan
    if np.isnan(y):
        continue
    error = std_mcc.iloc[i]["MCC_std"]
    ax.errorbar(x, y, yerr=error, fmt="none", capsize=5, c="black", elinewidth=1)
    i += 1

show_delta_on_bars(barplot, delta_mcc_sorted['MCC_mean'].values)

# Display the plot
# plt.show()
plt.savefig(os.path.join(LATEX_PATH, "mean_frozen_finetuned_results_plm_cnn.png"), bbox_inches='tight', dpi=300)
plt.close()

