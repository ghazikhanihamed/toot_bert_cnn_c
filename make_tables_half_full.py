import pandas as pd
import os
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# We read the csv file of the full results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH,
                 "mean_balanced_imbalanced_results.csv"))

# The dataframe has the following columns: "Task", "Dataset", "Representation", "Representer", "Precision", "Classifier", "MCC"


# We make a new dataframe out of the df dataframe, focusing on the Precision column.
# The new dataframe has the following columns: "Task", "Dataset", where we group by the representer under each category of "Task", "Precision".
# Then we take the best MCC value for each category of "Task", "Precision".

representation_types = [settings.FROZEN, settings.FINETUNED]
precision_types = ["half", "full"]
representers = settings.REPRESENTATIONS
# We use TASKS_SHORT dictionary to shorten the names of the tasks with the keys the long names and the values the short names
tasks = settings.TASKS
datasets = ["balanced", "imbalanced"]

ds_best_mcc = []
for task in tasks:
    for precision in precision_types:
        df_temp = df[(df["Task"] == task) & (df["Precision"] == precision)]
        if not df_temp.empty:
            for representer in representers:
                df_temp2 = df_temp[df_temp["Representer"]
                                   == representer["name"]]
                if not df_temp2.empty:
                    # We take the first three rows of the df_temp2 sorted by MCC value split by "±" and take the first element
                    three_best_mcc = df_temp2["MCC"].str.split(
                        "±").str[0].astype(float).nlargest(3).index.tolist()
                    df_three_best_mcc = df_temp2.loc[three_best_mcc]

                    # We take the best MCC value for each category of "Task", "Dataset" and "Representer"
                    best_mcc_id = df_temp2["MCC"].str.split(
                        "±").str[0].astype(float).idxmax()

                    best_mcc = df_temp2.loc[best_mcc_id, "MCC"]
                    ds_best_mcc.append(
                        [representer["name"], settings.TASKS_SHORT[task], precision, best_mcc])

# We create the dataframe with columns "PLM", "IC-MP Balanced", "IC-MP Imbalanced", "IT-MP Balanced, "IT-MP Imbalanced"
df_table = pd.DataFrame(ds_best_mcc, columns=[
                        "PLM", "Task", "Precision", "MCC"])
# Create a new column 'Task-Precision' by combining 'Task' and 'Precision' columns
df_table['Task-Precision'] = df_table['Task'] + ' ' + df_table['Precision']

# Pivot the DataFrame to the desired structure
df_table = df_table.pivot(index='PLM', columns='Task-Precision', values='MCC')

# We replace the NaN values with a dash
df_table.fillna('-', inplace=True)

# Reset the index
df_table.reset_index(inplace=True)

# Rename the index
df_table.index.name = None

# Convert the 'PLM' column to a categorical data type with the specified custom order
df_table['PLM'] = pd.Categorical(df_table['PLM'], categories=[
                                 representer["name"] for representer in representers], ordered=True)

# Sort the DataFrame based on the custom order of the 'PLM' column
df_table.sort_values(by='PLM', inplace=True)

# Reset the index
df_table.reset_index(drop=True, inplace=True)

# We create a latex table from the dataframe with the header boldface
latex_table = df_table.to_latex(
    index=False, escape=True, column_format='l' + 'c' * (len(df_table.columns) - 1))

# Replace the column names with their boldface versions
for col in df_table.columns:
    latex_table = latex_table.replace(col, '\\textbf{' + col + '}')

# Save the modified LaTeX table to a file
with open(os.path.join(settings.LATEX_PATH, "mean_half_full_results.tex"), 'w') as f:
    f.write(latex_table)

# Melt the DataFrame to 'long' format for easier plotting with Seaborn
df_table_melted = df_table.melt(
    id_vars='PLM', var_name='Task-Precision', value_name='MCC')

df_table_melted[['Mean', 'Error']] = df_table_melted['MCC'].str.split('±', expand=True)
df_table_melted['Mean'] = pd.to_numeric(df_table_melted['Mean'], errors='coerce')
df_table_melted['Error'] = pd.to_numeric(df_table_melted['Error'], errors='coerce')

# Set the style and context
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")

plt.figure(figsize=(12, 6))

# Add new column 'Precision' and fill it with 'half' or 'full'
df_table_melted['Precision'] = df_table_melted['Task-Precision'].apply(lambda x: 'full' if 'full' in x else 'half')

# Group by PLM and Precision, and take the mean and average the error
df_grouped = df_table_melted.groupby(['PLM', 'Precision']).agg({'Mean': 'mean', 'Error': 'mean'}).reset_index()

# Create line plot with markers and error bars
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

plms = df_grouped['PLM'].unique()
precisions = df_grouped['Precision'].unique()

markers = ['o', 'v', 's', 'D', 'P', 'X']

x = np.arange(len(plms))
width = 0.35

for i, precision in enumerate(precisions):
    precision_data = df_grouped[df_grouped["Precision"] == precision]
    plt.bar(x + (i - 0.5) * width, precision_data["Mean"], width, yerr=precision_data["Error"], capsize=3, label=f'{precision} precision')

plt.ylabel('MCC')
plt.xticks(x, plms)
plt.legend()

plt.tight_layout()
# Save the plot
plt.savefig(os.path.join(settings.LATEX_PATH,
                         "mean_half_full_results_bar.png"), dpi=300, bbox_inches='tight')



# Group by PLM and Precision, and take the mean and average the error
df_grouped = df_table_melted.groupby(['PLM', 'Task-Precision']).agg({'Mean': 'mean', 'Error': 'mean'}).reset_index()

# Create grouped bar plot with error bars
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

precisions = ['full', 'half']
plms = df_grouped['PLM'].unique()

x = np.arange(len(plms))
width = 0.35

for i, precision in enumerate(precisions):
    y = df_grouped[df_grouped["Task-Precision"].str.contains(precision)].groupby('PLM')['Mean'].mean().values
    error = df_grouped[df_grouped["Task-Precision"].str.contains(precision)].groupby('PLM')['Error'].mean().values
    offset = -width / 2 if i % 2 == 0 else width / 2
    plt.bar(x + offset, y, width, label=f'{precision} precision', yerr=error, capsize=3)
    plt.errorbar(x + offset, y, yerr=error, fmt="none", capsize=3, elinewidth=1.5, ecolor=sns.color_palette("colorblind")[i % 2])

plt.xticks(x, plms, rotation=45)
plt.ylabel('MCC')
plt.legend()

plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig(os.path.join(settings.LATEX_PATH,
                         "mean_half_full_results_bar.png"), dpi=300, bbox_inches='tight')

plt.close()


df_table_melted['MCC_numeric'] = df_table_melted['MCC'].apply(
    lambda x: float(x.split('±')[0]) if x != '-' else np.nan)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")


# Create the bar plot using Seaborn
plt.figure(figsize=(10, 4))
ax = sns.barplot(x='PLM', y='MCC_numeric',
                 hue='Task-Precision', data=df_table_melted)

# Customize the plot

plt.xlabel('Protein Language Models')
plt.ylabel('MCC')
plt.legend(loc='lower right', fontsize=8)

# Set the y-axis limits
plt.ylim(0.4)

# Loop through each bar and shift the position if the x_label is ProtT5 or ESM-2_15B
shift_width = 0.15
for i, bar in enumerate(ax.containers):
    for p in bar.patches:
        # , ax.get_xticks()[2.26], ax.get_xticks()[5.26]]:  # Indices for ProtT5 and ESM-2_15B
        if p.get_x() in [ax.get_xticks()[5], ax.get_xticks()[2]]:
            p.set_x(p.get_x() - shift_width if i == 3 or i == 5 else p.get_x())
        elif p.get_x() in [2.2666666666666666, 5.266666666666667]:
            p.set_x(p.get_x() - shift_width*1.9 if i ==
                    3 or i == 5 else p.get_x())

# Loop through each bar in the plot
for p in ax.patches:
    # Get the bar's x and y coordinates and its height
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    height = p.get_height()
    # Write the value of the bar on top of it
    ax.text(x, y, f'{height:.2f}', ha='center', va='bottom', fontsize=5)

# plt.show()
# Save the plot
plt.savefig(os.path.join(settings.LATEX_PATH,
                         "mean_half_full_results.png"), dpi=300, bbox_inches='tight')

plt.close()

# Pivot the dataframe to create a matrix of MCC values
df_pivot = df_table_melted.pivot('PLM', 'Task-Precision', 'MCC_numeric')

# Create the heatmap using Seaborn
plt.figure(figsize=(12, 4))
ax1 = sns.heatmap(df_pivot, annot=True, cmap="coolwarm",
                  xticklabels=True, cbar=True)

ax1.set_xlabel('')
ax1.set_ylabel('')

# plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig(os.path.join(settings.LATEX_PATH,
                         "mean_half_full_results_heatmap.png"), dpi=300, bbox_inches='tight')
