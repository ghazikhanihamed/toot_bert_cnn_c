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
    PLM_ORDER_FINETUNED,
    PLM_ORDER_SHORT,
    CLASSIFIER_ORDER,
    PLM_ORDER_FINETUNED_SHORT
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
            if i > len(delta_values) - 1:
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
    
pd.set_option('display.float_format', '{:.2e}'.format)

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

pvalue_dict = {}
# For each task, calculate the differences
for task in df['task_short'].unique():
    # Get balanced and imbalanced rows for this task
    frozen_rows = df[(df['task_short'] == task) &
                       (df['Representation'] == 'frozen')]
    finetuned_rows = df[(df['task_short'] == task) &
                         (df['Representation'] == 'finetuned')]
    
    # We filter the rows of frozen_rows to only keep the rows that have a finetuned counterpart
    frozen_rows = frozen_rows[frozen_rows['Representer'].isin(finetuned_rows['Representer'])]

    # Ensure the rows are in the same order
    frozen_rows = frozen_rows.sort_index()
    finetuned_rows = finetuned_rows.sort_index()

    
    # Conduct paired t-test
    _, p_value = stats.ttest_rel(
        frozen_rows['MCC_mean'], finetuned_rows['MCC_mean'])
    pvalue_dict[task] = p_value

# Group by Task and Dataset, and calculate the mean MCC
mean_mcc = df.groupby(['task_short', 'Representation'])['MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['task_short', 'Representation'])['MCC_std'].mean().reset_index()

# -------------------------- Sorting: mean_mcc for plotting: Task, Representation
# We change the order of mean_mcc as IC-MP, IT-MP, and IC-IT
mean_mcc['task_short'] = mean_mcc['task_short'].astype(
    'category').cat.reorder_categories(['IC-MP', 'IT-MP', 'IC-IT'], ordered=True)
mean_mcc = mean_mcc.sort_values('task_short')
mean_mcc['Representation'] = pd.Categorical(
    mean_mcc['Representation'], categories=representation_types, ordered=True)
mean_mcc = mean_mcc.sort_values(by=['task_short', 'Representation'])
# --------------------------

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

# Sorting df_metrics for latex table: Task, Representation ----------------------------------
df_metrics['Task'] = df_metrics['Task'].astype(
    'category').cat.reorder_categories(['IC-MP', 'IT-MP', 'IC-IT'], ordered=True)
df_metrics = df_metrics.sort_values('Task')
df_metrics['Representation'] = pd.Categorical(
    df_metrics['Representation'], categories=representation_types, ordered=True)
# Sort the dataframe by 'Task' and 'Representation'
df_metrics = df_metrics.sort_values(by=['Task', 'Representation'])
# ---------------------------------------------------------

# Add p-values
df_metrics['P-value'] = df_metrics['Task'].apply(lambda x: pvalue_dict[x]).map('{:.2e}'.format)

# Generate LaTeX table
latex_table = df_metrics.to_latex(index=False, float_format="%.2f", escape=False, column_format='lcccccc')

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

pvalue_dict = {}
# For each task, calculate the differences
for classifier in df['Classifier'].unique():
    # Get balanced and imbalanced rows for this task
    frozen_rows = df[(df['Classifier'] == classifier) &
                       (df['Representation'] == 'frozen')]
    finetuned_rows = df[(df['Classifier'] == classifier) &
                         (df['Representation'] == 'finetuned')]
    
        # We filter the rows of frozen_rows to only keep the rows that have a finetuned counterpart
    frozen_rows = frozen_rows[frozen_rows['Representer'].isin(finetuned_rows['Representer'])]

    # Ensure the rows are in the same order
    frozen_rows = frozen_rows.sort_index()
    finetuned_rows = finetuned_rows.sort_index()

    # Conduct paired t-test
    _, p_value = stats.ttest_rel(
        frozen_rows['MCC_mean'], finetuned_rows['MCC_mean'])
    pvalue_dict[classifier] = p_value

# Group by Classifier and Representation, and calculate the mean MCC
mean_mcc = df.groupby(['Classifier', 'Representation'])['MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['Classifier', 'Representation'])['MCC_std'].mean().reset_index()

# Sorting mean_mcc for plotting: Classifier, Representation ----------------------------------
mean_mcc['Classifier'] = mean_mcc['Classifier'].astype(
    'category').cat.reorder_categories(CLASSIFIER_ORDER, ordered=True)
mean_mcc = mean_mcc.sort_values('Classifier')
mean_mcc['Representation'] = pd.Categorical(
    mean_mcc['Representation'], categories=representation_types, ordered=True)
# Sort the dataframe by 'Classifier' and 'Representation'
mean_mcc = mean_mcc.sort_values(by=['Classifier', 'Representation'])
# ---------------------------------------------------------

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

# Sorting df_metrics for latex table: Classifier, Representation ----------------------------------
df_metrics['Classifier'] = df_metrics['Classifier'].astype(
    'category').cat.reorder_categories(CLASSIFIER_ORDER, ordered=True)
df_metrics = df_metrics.sort_values('Classifier')
df_metrics['Representation'] = pd.Categorical(df_metrics['Representation'], categories=representation_types, ordered=True)
# Sort the dataframe by 'Classifier' and 'Representation'
df_metrics = df_metrics.sort_values(by=['Classifier', 'Representation'])
# ---------------------------------------------------------

# Add p-values
df_metrics['P-value'] = df_metrics['Classifier'].map(pvalue_dict).map('{:.2e}'.format)

# Generate LaTeX table
latex_table = df_metrics.to_latex(index=False, float_format="%.2f", escape=False, column_format='lcccccc')

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

# PLMs ------------------------------------------------------------------------------------------

pvalue_dict = {}
# For each task, calculate the differences
for Representer in df['Representer'].unique():
    # Get balanced and imbalanced rows for this task
    frozen_rows = df[(df['Representer'] == Representer) &
                       (df['Representation'] == 'frozen')]
    finetuned_rows = df[(df['Representer'] == Representer) &
                         (df['Representation'] == 'finetuned')]
    
    # We filter the rows of frozen_rows to only keep the rows that have a finetuned counterpart
    frozen_rows = frozen_rows[frozen_rows['Representer'].isin(finetuned_rows['Representer'])]

    # Ensure the rows are in the same order
    frozen_rows = frozen_rows.sort_index()
    finetuned_rows = finetuned_rows.sort_index()

    # Conduct paired t-test
    _, p_value = stats.ttest_rel(
        frozen_rows['MCC_mean'], finetuned_rows['MCC_mean'])
    pvalue_dict[Representer] = p_value

# Group by Representer and Dataset, and calculate the mean MCC
mean_mcc = df.groupby(['Representer', 'Representation'])['MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['Representer', 'Representation'])['MCC_std'].mean().reset_index()

# Sorting mean_mcc for plotting: Representer, Representation ----------------------------------
mean_mcc['Representer'] = mean_mcc['Representer'].astype(
    'category').cat.reorder_categories(PLM_ORDER_SHORT, ordered=True)
mean_mcc = mean_mcc.sort_values('Representer')
mean_mcc['Representation'] = pd.Categorical(mean_mcc['Representation'], categories=representation_types, ordered=True)
# Sort the dataframe by 'Representer' and 'Representation'
mean_mcc = mean_mcc.sort_values(by=['Representer', 'Representation'])
# ---------------------------------------------------------

mean_accuracy = df.groupby(['Representer', 'Representation'])['Accuracy_mean'].mean().reset_index()
std_accuracy = df.groupby(['Representer', 'Representation'])['Accuracy_std'].mean().reset_index()

mean_sensitivity = df.groupby(['Representer', 'Representation'])['Sensitivity_mean'].mean().reset_index()
std_sensitivity = df.groupby(['Representer', 'Representation'])['Sensitivity_std'].mean().reset_index()

mean_specificity = df.groupby(['Representer', 'Representation'])['Specificity_mean'].mean().reset_index()
std_specificity = df.groupby(['Representer', 'Representation'])['Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['Representer', 'Representation'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['Representer', 'Representation'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity, on=['Representer', 'Representation'])
specificity = pd.merge(mean_specificity, std_specificity, on=['Representer', 'Representation'])

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
df_metrics = df_metrics[['Representer', 'Representation', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]
df_metrics = df_metrics.sort_values(by=['Representer'], ascending=True)

# Rename column Representer to PLM
df_metrics = df_metrics.rename(columns={'Representer': 'PLM'})

# Sorting df_metrics for latex table: PLM, Representation ----------------------------------
df_metrics['PLM'] = df_metrics['PLM'].astype(
    'category').cat.reorder_categories(PLM_ORDER_SHORT, ordered=True)
df_metrics = df_metrics.sort_values('PLM')
df_metrics['Representation'] = pd.Categorical(df_metrics['Representation'], categories=representation_types, ordered=True)
# Sort the dataframe by 'PLM' and 'Representation'
df_metrics = df_metrics.sort_values(by=['PLM', 'Representation'])
# ---------------------------------------------------------

# Add p-values
df_metrics['P-value'] = df_metrics['PLM'].map(pvalue_dict).map('{:.2e}'.format)

# Generate LaTeX table
latex_table = df_metrics.to_latex(index=False, float_format="%.2f", escape=False, column_format='lcccccc')

# save the table to a file
with open(os.path.join(LATEX_PATH, "mean_frozen_finetuned_results_plm_cnn.tex"), "w") as f:
    f.write(latex_table)

delta_mcc = mean_mcc[mean_mcc['Representation'] == 'finetuned'][['Representer', 'MCC_mean']].reset_index(drop=True)
frozen_mcc_values = mean_mcc[mean_mcc['Representation'] == 'frozen'][['Representer', 'MCC_mean']].reset_index(drop=True)
delta_mcc['MCC_mean'] = [finetuned - frozen_mcc_values.loc[frozen_mcc_values['Representer'] == Representer, 'MCC_mean'].values[0] 
                         for Representer, finetuned in zip(delta_mcc['Representer'], delta_mcc['MCC_mean'])]

# We sort the values by Representer the same as in PLM_ORDER_FINETUNED list
delta_mcc['Representer'] = pd.Categorical(delta_mcc['Representer'], categories=PLM_ORDER_FINETUNED_SHORT, ordered=True)
# Sort delta_mcc based on the Representer column with the specified order
delta_mcc_sorted = delta_mcc.sort_values('Representer')
# Reset the index of the sorted DataFrame
delta_mcc_sorted.reset_index(drop=True, inplace=True)


# Create the bar plot
plt.figure(figsize=(10, 4))
ax = barplot = sns.barplot(
    x='Representer',
    y='MCC_mean',
    hue='Representation',
    data=mean_mcc,
    ci=None,
    palette="colorblind",
    order=PLM_ORDER_SHORT
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

