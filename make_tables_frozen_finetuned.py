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


# We make a new dataframe out of the df dataframe, focusing on the Representation column.
# The new dataframe has the following columns: "Task", "Representation", where we group by the representer under each category of "Task", "Representation".
# Then we take the best MCC value for each category of "Task", "Representation".

representation_types = [settings.FROZEN, settings.FINETUNED]
precision_types = ["half", "full"]
representers = settings.REPRESENTATIONS
# We use TASKS_SHORT dictionary to shorten the names of the tasks with the keys the long names and the values the short names
tasks = settings.TASKS
datasets = ["balanced", "imbalanced"]

ds_best_mcc = []
for task in tasks:
    for representation in representation_types:
        df_temp = df[(df["Task"] == task) & (
            df["Representation"] == representation)]
        if not df_temp.empty:
            for representer in representers:
                df_temp2 = df_temp[df_temp["Representer"]
                                   == representer["name"]]
                if not df_temp2.empty:
                    # We take the first three rows of the df_temp2 sorted by MCC value split by "±" and take the first element
                    three_best_mcc = df_temp2["MCC"].str.split(
                        "±").str[0].astype(float).nlargest(3).index.tolist()
                    df_three_best_mcc = df_temp2.loc[three_best_mcc]

                    # We take the best MCC value for each category of "Task", "Representation" and "Representer"
                    best_mcc_id = df_temp2["MCC"].str.split(
                        "±").str[0].astype(float).idxmax()

                    best_mcc = df_temp2.loc[best_mcc_id, "MCC"]
                    ds_best_mcc.append(
                        [representer["name"], settings.TASKS_SHORT[task], representation, best_mcc])

# We create the dataframe
df_table = pd.DataFrame(ds_best_mcc, columns=[
                        "PLM", "Task", "Representation", "MCC"])
# Create a new column 'Task-Representation' by combining 'Task' and 'Representation' columns
df_table['Task-Representation'] = df_table['Task'] + \
    ' ' + df_table['Representation']

# Pivot the DataFrame to the desired structure
df_table = df_table.pivot(
    index='PLM', columns='Task-Representation', values='MCC')

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
latex_table = df.to_latex(
    index=False, escape=True, column_format='l' + 'c' * (len(df_table.columns) - 1))

# Replace the column names with their boldface versions
for col in df_table.columns:
    latex_table = latex_table.replace(col, '\\textbf{' + col + '}')

# Save the modified LaTeX table to a file
with open(os.path.join(settings.LATEX_PATH, "mean_balanced_imbalanced_results.tex"), 'w') as f:
    f.write(latex_table)

# Melt the DataFrame to 'long' format for easier plotting with Seaborn
df_table_melted = df_table.melt(
    id_vars='PLM', var_name='Task-Representation', value_name='MCC')

df_table_melted['MCC_numeric'] = df_table_melted['MCC'].apply(
    lambda x: float(x.split('±')[0]) if x != '-' else np.nan)

df_table_melted['Type'] = df_table_melted['Task-Representation'].apply(lambda x: 'Frozen' if 'frozen' in x else 'Finetuned')

# Group by PLM and Type, then compute the mean of MCC_numeric
df_mean = df_table_melted.groupby(['PLM', 'Type']).mean().reset_index()

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")

# Plot the bar plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df_mean, x='PLM', y='MCC_numeric', hue='Type')

plt.xlabel('Protein Language Models')
plt.ylabel('Average MCC')
plt.legend(loc='lower right', fontsize=9)

# Set the y-axis limits
plt.ylim(0.4)

# Loop through each bar and shift the position if the x_label is ProtT5 or ESM-2_15B
shift_width = 0.17
for i, bar in enumerate(ax.containers):
    for p in bar.patches:
        if p.get_x() in [ax.get_xticks()[5], ax.get_xticks()[2]]:  # Indices for ProtT5 and ESM-2_15B
            p.set_x(p.get_x() - shift_width if i == 0 else p.get_x() - shift_width)

# Loop through each bar in the plot
for p in ax.patches:
    # Get the bar's x and y coordinates and its height
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    height = p.get_height()
    # Write the value of the bar on top of it
    ax.text(x, y, f'{height:.2f}', ha='center', va='bottom', fontsize=12)

# plt.show()
# Save the plot
plt.savefig(os.path.join(settings.LATEX_PATH,
                         "mean_frozen_finetuned_results_bar.png"), dpi=300, bbox_inches='tight')

plt.close()


# Create the bar plot using Seaborn
plt.figure(figsize=(10, 4))
ax = sns.barplot(x='PLM', y='MCC_numeric',
                 hue='Task-Representation', data=df_table_melted)

# Customize the plot

plt.xlabel('Protein Language Models')
plt.ylabel('MCC')
plt.legend(loc='lower right', fontsize=9)

# Set the y-axis limits
plt.ylim(0.4)

# Loop through each bar and shift the position if the x_label is ProtT5 or ESM-2_15B
shift_width = 0.15
for i, bar in enumerate(ax.containers):
    for p in bar.patches:
        if p.get_x() in [ax.get_xticks()[5], ax.get_xticks()[2]]:#, ax.get_xticks()[2.26], ax.get_xticks()[5.26]]:  # Indices for ProtT5 and ESM-2_15B
            p.set_x(p.get_x() - shift_width if i == 3 or i == 5 else p.get_x())
        elif p.get_x() in [2.2666666666666666, 5.266666666666667]:
            p.set_x(p.get_x() - shift_width*1.9 if i == 3 or i == 5 else p.get_x())

# Loop through each bar in the plot
for p in ax.patches:
    # Get the bar's x and y coordinates and its height
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    height = p.get_height()
    # Write the value of the bar on top of it
    ax.text(x, y, f'{height:.2f}', ha='center', va='bottom', fontsize=7)

# plt.show()
# Save the plot
plt.savefig(os.path.join(settings.LATEX_PATH,
                         "mean_frozen_finetuned_results.png"), dpi=300, bbox_inches='tight')

plt.close()

# Pivot the dataframe to create a matrix of MCC values
df_pivot = df_table_melted.pivot('PLM', 'Task-Representation', 'MCC_numeric')

# Create the heatmap using Seaborn
plt.figure(figsize=(12, 4))
ax1 = sns.heatmap(df_pivot, annot=True, cmap="coolwarm", xticklabels=True, cbar=True)

ax1.set_xlabel('')
ax1.set_ylabel('')

# plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig(os.path.join(settings.LATEX_PATH,
                            "mean_frozen_finetuned_results_heatmap.png"), dpi=300, bbox_inches='tight')

plt.close()