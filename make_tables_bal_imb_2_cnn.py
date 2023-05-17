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
df = df[df['Task'] != 'ionchannels_iontransporters']


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
df['MCC_diff'] = np.nan

# --------------------------------------------- TASKS

pvalue_dict = {}
# For each task, calculate the differences
for task in df['task_short'].unique():
    # Get balanced and imbalanced rows for this task
    balanced_rows = df[(df['task_short'] == task) &
                       (df['Dataset'] == 'balanced')]
    imbalanced_rows = df[(df['task_short'] == task) &
                         (df['Dataset'] == 'imbalanced')]

    # Ensure the rows are in the same order
    balanced_rows = balanced_rows.sort_index()
    imbalanced_rows = imbalanced_rows.sort_index()

    # Conduct paired t-test
    _, p_value = stats.ttest_rel(
        balanced_rows['MCC_mean'], imbalanced_rows['MCC_mean'])
    pvalue_dict[task] = p_value


balanced_df = df[df['Dataset'] == 'balanced']
imbalanced_df = df[df['Dataset'] == 'imbalanced']


# Group by Task and Dataset, and calculate the mean MCC
mean_mcc = df.groupby(['task_short', 'Dataset'])[
    'MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['task_short', 'Dataset'])['MCC_std'].mean().reset_index()

mean_accuracy = df.groupby(['task_short', 'Dataset'])[
    'Accuracy_mean'].mean().reset_index()
std_accuracy = df.groupby(['task_short', 'Dataset'])[
    'Accuracy_std'].mean().reset_index()

mean_sensitivity = df.groupby(['task_short', 'Dataset'])[
    'Sensitivity_mean'].mean().reset_index()
std_sensitivity = df.groupby(['task_short', 'Dataset'])[
    'Sensitivity_std'].mean().reset_index()

mean_specificity = df.groupby(['task_short', 'Dataset'])[
    'Specificity_mean'].mean().reset_index()
std_specificity = df.groupby(['task_short', 'Dataset'])[
    'Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['task_short', 'Dataset'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['task_short', 'Dataset'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity,
                       on=['task_short', 'Dataset'])
specificity = pd.merge(mean_specificity, std_specificity,
                       on=['task_short', 'Dataset'])

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
df_metrics = df_metrics[['task_short', 'Dataset',
                         'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]
df_metrics = df_metrics.rename(columns={'task_short': 'Task'})

# We add the p-values to the table
df_metrics['P-value'] = df_metrics['Task'].apply(lambda x: pvalue_dict[x]).map('{:.2e}'.format)


# Generate LaTeX table
latex_table = df_metrics.to_latex(
    index=False, float_format="%.2f", escape=False, column_format='lcccccc')

# save the table to a file
with open(os.path.join(LATEX_PATH, "mean_balanced_imbalanced_results_task_cnn.tex"), "w") as f:
    f.write(latex_table)

delta_mcc = mean_mcc[mean_mcc['Dataset'] == 'imbalanced']['MCC_mean'].values - \
    mean_mcc[mean_mcc['Dataset'] == 'balanced']['MCC_mean'].values

# Create the bar plot
plt.figure(figsize=(10, 4))
barplot = sns.barplot(
    x='task_short',
    y='MCC_mean',
    hue='Dataset',
    data=mean_mcc,
    ci=None,
    palette="colorblind"
)

# X and Y axis labels
plt.xlabel("Task")
plt.ylabel("MCC")

# Add legend
plt.legend(title="Dataset")

# Add error bars
for i, p in enumerate(barplot.patches):
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    err = std_mcc.loc[i, 'MCC_std']
    plt.errorbar(x, y, yerr=err, capsize=3,
                 elinewidth=1.5, color='black', ls='none')

# Add MCC values on top of each bar
show_values_on_bars(barplot)

show_delta_on_bars(barplot, delta_mcc)
# Display the plot
# plt.show()
plt.savefig(os.path.join(
    LATEX_PATH, "mean_balanced_imbalanced_results_task_cnn.png"), bbox_inches='tight', dpi=300)
plt.close()

# ---------------------------------------------

pvalue_dict = {}
# For each task, calculate the differences
for classifier in df['Classifier'].unique():
    # Get balanced and imbalanced rows for this task
    balanced_rows = df[(df['Classifier'] == classifier)
                       & (df['Dataset'] == 'balanced')]
    imbalanced_rows = df[(df['Classifier'] == classifier)
                         & (df['Dataset'] == 'imbalanced')]

    # Ensure the rows are in the same order
    balanced_rows = balanced_rows.sort_index()
    imbalanced_rows = imbalanced_rows.sort_index()

    # Conduct paired t-test
    _, p_value = stats.ttest_rel(
        balanced_rows['MCC_mean'], imbalanced_rows['MCC_mean'])
    pvalue_dict[classifier] = p_value


# Group by Classifier and Dataset, and calculate the mean MCC
mean_mcc = df.groupby(['Classifier', 'Dataset'])[
    'MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['Classifier', 'Dataset'])['MCC_std'].mean().reset_index()

mean_accuracy = df.groupby(['Classifier', 'Dataset'])[
    'Accuracy_mean'].mean().reset_index()
std_accuracy = df.groupby(['Classifier', 'Dataset'])[
    'Accuracy_std'].mean().reset_index()

mean_sensitivity = df.groupby(['Classifier', 'Dataset'])[
    'Sensitivity_mean'].mean().reset_index()
std_sensitivity = df.groupby(['Classifier', 'Dataset'])[
    'Sensitivity_std'].mean().reset_index()

mean_specificity = df.groupby(['Classifier', 'Dataset'])[
    'Specificity_mean'].mean().reset_index()
std_specificity = df.groupby(['Classifier', 'Dataset'])[
    'Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['Classifier', 'Dataset'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['Classifier', 'Dataset'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity,
                       on=['Classifier', 'Dataset'])
specificity = pd.merge(mean_specificity, std_specificity,
                       on=['Classifier', 'Dataset'])

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
df_metrics = df_metrics[['Classifier', 'Dataset',
                         'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]

# We add a column for the p-value
df_metrics['P-value'] = df_metrics['Classifier'].map(pvalue_dict).map('{:.2e}'.format)

# Generate LaTeX table
latex_table = df_metrics.to_latex(
    index=False, float_format="%.2f", escape=False, column_format='lcccccc')

# save the table to a file
with open(os.path.join(LATEX_PATH, "mean_balanced_imbalanced_results_classifier_cnn.tex"), "w") as f:
    f.write(latex_table)


delta_mcc = mean_mcc[mean_mcc['Dataset'] == 'imbalanced']['MCC_mean'].values - \
    mean_mcc[mean_mcc['Dataset'] == 'balanced']['MCC_mean'].values


# Create the bar plot
plt.figure(figsize=(10, 4))
barplot = sns.barplot(
    x='Classifier',
    y='MCC_mean',
    hue='Dataset',
    data=mean_mcc,
    ci=None,
    palette="colorblind"
)

# X and Y axis labels
plt.xlabel("Classifier")
plt.ylabel("MCC")

# Add legend
plt.legend(title="Dataset")

# Add error bars
for i, p in enumerate(barplot.patches):
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    err = std_mcc.loc[i, 'MCC_std']
    plt.errorbar(x, y, yerr=err, capsize=3,
                 elinewidth=1.5, color='black', ls='none')

# Add Sensitivity values on top of each bar
show_values_on_bars(barplot)

show_delta_on_bars(barplot, delta_mcc)

# Display the plot
# plt.show()
plt.savefig(os.path.join(
    LATEX_PATH, "mean_balanced_imbalanced_results_classifier_cnn.png"), bbox_inches='tight', dpi=300)
plt.close()

#  PLM SIZE

pvalue_dict = {}

for plm_size in df['plm_size'].unique():
    # Get the rows for the current plm_size
    balanced_rows = df[(df['plm_size'] == plm_size) &
                       (df['Dataset'] == 'balanced')]
    imbalanced_rows = df[(df['plm_size'] == plm_size) &
                         (df['Dataset'] == 'imbalanced')]

    # Ensure the rows are in the same order
    balanced_rows = balanced_rows.sort_index()
    imbalanced_rows = imbalanced_rows.sort_index()

    # Conduct paired t-test
    _, p_value = stats.ttest_rel(
        balanced_rows['MCC_mean'], imbalanced_rows['MCC_mean'])
    pvalue_dict[plm_size] = p_value

# Group by Representer and Dataset, and calculate the mean MCC
mean_mcc = df.groupby(['plm_size', 'Dataset'])['MCC_mean'].mean().reset_index()
std_mcc = df.groupby(['plm_size', 'Dataset'])['MCC_std'].mean().reset_index()

mean_accuracy = df.groupby(['plm_size', 'Dataset'])[
    'Accuracy_mean'].mean().reset_index()
std_accuracy = df.groupby(['plm_size', 'Dataset'])[
    'Accuracy_std'].mean().reset_index()

mean_sensitivity = df.groupby(['plm_size', 'Dataset'])[
    'Sensitivity_mean'].mean().reset_index()
std_sensitivity = df.groupby(['plm_size', 'Dataset'])[
    'Sensitivity_std'].mean().reset_index()

mean_specificity = df.groupby(['plm_size', 'Dataset'])[
    'Specificity_mean'].mean().reset_index()
std_specificity = df.groupby(['plm_size', 'Dataset'])[
    'Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['plm_size', 'Dataset'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['plm_size', 'Dataset'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity,
                       on=['plm_size', 'Dataset'])
specificity = pd.merge(mean_specificity, std_specificity,
                       on=['plm_size', 'Dataset'])

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
df_metrics = df_metrics[['plm_size', 'Dataset',
                         'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]
df_metrics = df_metrics.sort_values(by=['plm_size'], ascending=True)

# Change column name plm_size to PLM
df_metrics = df_metrics.rename(columns={'plm_size': 'PLM'})

# Add p-values
df_metrics['P-value'] = df_metrics['PLM'].map(pvalue_dict).map('{:.2e}'.format)

# Generate LaTeX table
latex_table = df_metrics.to_latex(
    index=False, float_format="%.2f", escape=False, column_format='lcccccc')

# save the table to a file
with open(os.path.join(LATEX_PATH, "mean_balanced_imbalanced_results_plm_cnn.tex"), "w") as f:
    f.write(latex_table)


# delta_mcc = mean_mcc[mean_mcc['Dataset'] == 'imbalanced']['MCC_mean'] - mean_mcc[mean_mcc['Dataset'] == 'balanced']['MCC_mean']
delta_mcc = mean_mcc[mean_mcc['Dataset'] == 'balanced'][[
    'plm_size', 'MCC_mean']].reset_index(drop=True)
frozen_mcc_values = mean_mcc[mean_mcc['Dataset'] == 'imbalanced'][[
    'plm_size', 'MCC_mean']].reset_index(drop=True)
delta_mcc['MCC_mean'] = [finetuned - frozen_mcc_values.loc[frozen_mcc_values['plm_size'] == plm_size, 'MCC_mean'].values[0]
                         for plm_size, finetuned in zip(delta_mcc['plm_size'], delta_mcc['MCC_mean'])]

# We sort the values by plm_size the same as in PLM_ORDER_FINETUNED list
delta_mcc['plm_size'] = pd.Categorical(
    delta_mcc['plm_size'], categories=PLM_ORDER, ordered=True)
# Sort delta_mcc based on the plm_size column with the specified order
delta_mcc_sorted = delta_mcc.sort_values('plm_size')
# Reset the index of the sorted DataFrame
delta_mcc_sorted.reset_index(drop=True, inplace=True)

# Create the bar plot
plt.figure(figsize=(10, 4))
barplot = sns.barplot(
    x='plm_size',
    y='MCC_mean',
    hue='Dataset',
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
plt.legend(title="Dataset", loc='lower left')

# Add error bars
for i, p in enumerate(barplot.patches):
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    err = std_mcc.loc[i, 'MCC_std']
    plt.errorbar(x, y, yerr=err, capsize=3,
                 elinewidth=1.5, color='black', ls='none')

# Add Sensitivity values on top of each bar
show_values_on_bars(barplot)

show_delta_on_bars(barplot, delta_mcc_sorted['MCC_mean'].values)

# Display the plot
# plt.show()
plt.savefig(os.path.join(
    LATEX_PATH, "mean_balanced_imbalanced_results_plm_cnn.png"), bbox_inches='tight', dpi=300)
plt.close()


# Fine-tuned vs frozen --------------------------------------------------------


# Filter DataFrame to only include frozen representation
finetuned_df = df[df['Representation'] == 'finetuned']

pvalue_dict = {}

for plm_size in finetuned_df['plm_size'].unique():
    balanced_rows = finetuned_df[(finetuned_df['plm_size'] == plm_size) & (
        finetuned_df['Dataset'] == 'balanced')]
    imbalanced_rows = finetuned_df[(finetuned_df['plm_size'] == plm_size) & (
        finetuned_df['Dataset'] == 'imbalanced')]

    # Ensure the rows are in the same order
    balanced_rows = balanced_rows.sort_index()
    imbalanced_rows = imbalanced_rows.sort_index()

    # Conduct paired t-test
    _, p_value = stats.ttest_rel(
        balanced_rows['MCC_mean'], imbalanced_rows['MCC_mean'])
    pvalue_dict[plm_size] = p_value


# Group by Representer and Dataset, and calculate the mean MCC
mean_mcc = finetuned_df.groupby(['plm_size', 'Dataset'])[
    'MCC_mean'].mean().reset_index()
std_mcc = finetuned_df.groupby(['plm_size', 'Dataset'])[
    'MCC_std'].mean().reset_index()

mean_accuracy = finetuned_df.groupby(['plm_size', 'Dataset'])[
    'Accuracy_mean'].mean().reset_index()
std_accuracy = finetuned_df.groupby(['plm_size', 'Dataset'])[
    'Accuracy_std'].mean().reset_index()

mean_sensitivity = finetuned_df.groupby(['plm_size', 'Dataset'])[
    'Sensitivity_mean'].mean().reset_index()
std_sensitivity = finetuned_df.groupby(['plm_size', 'Dataset'])[
    'Sensitivity_std'].mean().reset_index()

mean_specificity = finetuned_df.groupby(['plm_size', 'Dataset'])[
    'Specificity_mean'].mean().reset_index()
std_specificity = finetuned_df.groupby(['plm_size', 'Dataset'])[
    'Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['plm_size', 'Dataset'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['plm_size', 'Dataset'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity,
                       on=['plm_size', 'Dataset'])
specificity = pd.merge(mean_specificity, std_specificity,
                       on=['plm_size', 'Dataset'])

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
df_metrics = df_metrics[['plm_size', 'Dataset',
                         'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]
# Rename columns
df_metrics.columns = ['PLM', 'Dataset', 'MCC',
                      'Accuracy', 'Sensitivity', 'Specificity']

# Sort the dataframe by PLM size
df_metrics = df_metrics.sort_values('PLM')

# add p-values
df_metrics['P-value'] = df_metrics['PLM'].map(pvalue_dict).map('{:.2e}'.format)

# Save the dataframe as a latex table
latex_table = df_metrics.to_latex(
    index=False, escape=False, column_format='lcccccc')

# Save the latex table to a file
with open(os.path.join(LATEX_PATH, "mean_balanced_imbalanced_results_plm_finetuned_cnn.tex"), 'w') as f:
    f.write(latex_table)

mean_mcc['plm_size'] = mean_mcc['plm_size'].astype('category')
mean_mcc['Dataset'] = mean_mcc['Dataset'].astype('category')

diff_mcc = mean_mcc.pivot_table(
    values='MCC_mean', index='plm_size', columns='Dataset').reset_index()
diff_mcc['MCC_diff'] = diff_mcc['imbalanced'] - diff_mcc['balanced']

average_mcc_diff = diff_mcc['MCC_diff'].mean()

print(f"Average MCC difference: {average_mcc_diff}")

# delta_mcc = mean_mcc[mean_mcc['Dataset'] == 'imbalanced']['MCC_mean'] - mean_mcc[mean_mcc['Dataset'] == 'balanced']['MCC_mean']
delta_mcc = mean_mcc[mean_mcc['Dataset'] == 'imbalanced'][[
    'plm_size', 'MCC_mean']].reset_index(drop=True)
frozen_mcc_values = mean_mcc[mean_mcc['Dataset'] == 'balanced'][[
    'plm_size', 'MCC_mean']].reset_index(drop=True)
delta_mcc['MCC_mean'] = [finetuned - frozen_mcc_values.loc[frozen_mcc_values['plm_size'] == plm_size, 'MCC_mean'].values[0]
                         for plm_size, finetuned in zip(delta_mcc['plm_size'], delta_mcc['MCC_mean'])]

# We sort the values by plm_size the same as in PLM_ORDER_FINETUNED list
delta_mcc['plm_size'] = pd.Categorical(
    delta_mcc['plm_size'], categories=PLM_ORDER_FINETUNED, ordered=True)
# Sort delta_mcc based on the plm_size column with the specified order
delta_mcc_sorted = delta_mcc.sort_values('plm_size')
# Reset the index of the sorted DataFrame
delta_mcc_sorted.reset_index(drop=True, inplace=True)

# Create the bar plot
plt.figure(figsize=(10, 4))
barplot = sns.barplot(
    x='plm_size',
    y='MCC_mean',
    hue='Dataset',
    data=mean_mcc,
    ci=None,
    palette="colorblind",
    order=PLM_ORDER_FINETUNED
)

# X and Y axis labels
plt.xlabel("PLM (parameter size)")
plt.ylabel("MCC")

plt.xticks(fontsize=9)

# Add legend
plt.legend(title="Dataset", loc='lower left')

# Add error bars
for i, p in enumerate(barplot.patches):
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    err = std_mcc.loc[i, 'MCC_std']
    plt.errorbar(x, y, yerr=err, capsize=3,
                 elinewidth=1.5, color='black', ls='none')

# Add MCC values on top of each bar
show_values_on_bars(barplot)

show_delta_on_bars(barplot, delta_mcc_sorted['MCC_mean'].values)

# Display the plot
# plt.show()
plt.savefig(os.path.join(
    LATEX_PATH, "mean_balanced_imbalanced_results_plm_finetuned_cnn.png"), bbox_inches='tight', dpi=300)
plt.close()

# Filter DataFrame to only include frozen representation
frozen_df = df[df['Representation'] == 'frozen']

pvalue_dict = {}
for plm_size in frozen_df['plm_size'].unique():
    balanced_rows = frozen_df[(frozen_df['plm_size'] == plm_size) & (
        frozen_df['Dataset'] == 'balanced')]
    imbalanced_rows = frozen_df[(frozen_df['plm_size'] == plm_size) & (
        frozen_df['Dataset'] == 'imbalanced')]
    
    # Ensure the rows are in the same order
    balanced_rows = balanced_rows.sort_index()
    imbalanced_rows = imbalanced_rows.sort_index()

    # Conduct paired t-test
    _, p_value = stats.ttest_rel(
        balanced_rows['MCC_mean'], imbalanced_rows['MCC_mean'])
    pvalue_dict[plm_size] = p_value

# Group by Representer and Dataset, and calculate the mean MCC
mean_mcc = frozen_df.groupby(['plm_size', 'Dataset'])[
    'MCC_mean'].mean().reset_index()
std_mcc = frozen_df.groupby(['plm_size', 'Dataset'])[
    'MCC_std'].mean().reset_index()

mean_accuracy = frozen_df.groupby(['plm_size', 'Dataset'])[
    'Accuracy_mean'].mean().reset_index()
std_accuracy = frozen_df.groupby(['plm_size', 'Dataset'])[
    'Accuracy_std'].mean().reset_index()

mean_sensitivity = frozen_df.groupby(['plm_size', 'Dataset'])[
    'Sensitivity_mean'].mean().reset_index()
std_sensitivity = frozen_df.groupby(['plm_size', 'Dataset'])[
    'Sensitivity_std'].mean().reset_index()

mean_specificity = frozen_df.groupby(['plm_size', 'Dataset'])[
    'Specificity_mean'].mean().reset_index()
std_specificity = frozen_df.groupby(['plm_size', 'Dataset'])[
    'Specificity_std'].mean().reset_index()

# Merge the mean and std dataframes for each metric
mcc = pd.merge(mean_mcc, std_mcc, on=['plm_size', 'Dataset'])
accuracy = pd.merge(mean_accuracy, std_accuracy, on=['plm_size', 'Dataset'])
sensitivity = pd.merge(mean_sensitivity, std_sensitivity,
                       on=['plm_size', 'Dataset'])
specificity = pd.merge(mean_specificity, std_specificity,
                       on=['plm_size', 'Dataset'])

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
df_metrics = df_metrics[['plm_size', 'Dataset',
                         'MCC', 'Accuracy', 'Sensitivity', 'Specificity']]
# Rename columns
df_metrics.columns = ['PLM', 'Dataset', 'MCC',
                      'Accuracy', 'Sensitivity', 'Specificity']

# Sort the dataframe by PLM size
df_metrics = df_metrics.sort_values('PLM')

# Add p-values to the table
df_metrics['P-value'] = df_metrics['PLM'].map(pvalue_dict).map('{:.2e}'.format)

# Save the dataframe as a latex table
latex_table = df_metrics.to_latex(
    index=False, escape=False, column_format='lcccccc')

# Save the latex table to a file
with open(os.path.join(LATEX_PATH, "mean_balanced_imbalanced_results_plm_frozen_cnn.tex"), 'w') as f:
    f.write(latex_table)

mean_mcc['plm_size'] = mean_mcc['plm_size'].astype('category')
mean_mcc['Dataset'] = mean_mcc['Dataset'].astype('category')

diff_mcc = mean_mcc.pivot_table(
    values='MCC_mean', index='plm_size', columns='Dataset').reset_index()
diff_mcc['MCC_diff'] = diff_mcc['balanced'] - diff_mcc['imbalanced']

average_mcc_diff = diff_mcc['MCC_diff'].mean()

print(f"Average MCC difference: {average_mcc_diff}")

# delta_mcc = mean_mcc[mean_mcc['Dataset'] == 'imbalanced']['MCC_mean'] - mean_mcc[mean_mcc['Dataset'] == 'balanced']['MCC_mean']
delta_mcc = mean_mcc[mean_mcc['Dataset'] == 'balanced'][[
    'plm_size', 'MCC_mean']].reset_index(drop=True)
frozen_mcc_values = mean_mcc[mean_mcc['Dataset'] == 'imbalanced'][[
    'plm_size', 'MCC_mean']].reset_index(drop=True)
delta_mcc['MCC_mean'] = [finetuned - frozen_mcc_values.loc[frozen_mcc_values['plm_size'] == plm_size, 'MCC_mean'].values[0]
                         for plm_size, finetuned in zip(delta_mcc['plm_size'], delta_mcc['MCC_mean'])]

# We sort the values by plm_size the same as in PLM_ORDER_FINETUNED list
delta_mcc['plm_size'] = pd.Categorical(
    delta_mcc['plm_size'], categories=PLM_ORDER, ordered=True)
# Sort delta_mcc based on the plm_size column with the specified order
delta_mcc_sorted = delta_mcc.sort_values('plm_size')
# Reset the index of the sorted DataFrame
delta_mcc_sorted.reset_index(drop=True, inplace=True)

# Create the bar plot
plt.figure(figsize=(10, 4))
barplot = sns.barplot(
    x='plm_size',
    y='MCC_mean',
    hue='Dataset',
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
plt.legend(title="Dataset", loc='lower left')

# Add error bars
for i, p in enumerate(barplot.patches):
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    err = std_mcc.loc[i, 'MCC_std']
    plt.errorbar(x, y, yerr=err, capsize=3,
                 elinewidth=1.5, color='black', ls='none')

# Add MCC values on top of each bar
show_values_on_bars(barplot)

show_delta_on_bars(barplot, delta_mcc_sorted['MCC_mean'].values)

# y limit to 1
plt.ylim(0, 1)

# Display the plot
# plt.show()
plt.savefig(os.path.join(
    LATEX_PATH, "mean_balanced_imbalanced_results_plm_frozen_cnn.png"), bbox_inches='tight', dpi=300)
plt.close()
