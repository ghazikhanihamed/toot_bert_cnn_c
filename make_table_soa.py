import pandas as pd
import os
from settings import settings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd

data = [
    ['DeepIon', 'IC-MP', 'PSSM', 'CNN', 86.53, 0.37],
    ['DeepIon', 'IT-MP', 'PSSM', 'CNN', 83.78, 0.37],
    ['DeepIon', 'IC-IT', '-', '-', None, None],
    ['MFPS_CNN', 'IC-MP', 'PSSM', 'CNN', 94.60, 0.62],
    ['MFPS_CNN', 'IT-MP', 'PSSM', 'CNN', 93.30, 0.59],
    ['MFPS_CNN', 'IC-IT', '-', '-', None, None],
    ['TooT-BERT-C', 'IC-MP', 'ProtBERT-BFD', 'LR', 98.24, 0.85],
    ['TooT-BERT-C', 'IT-MP', 'ProtBERT-BFD', 'LR', 95.43, 0.64],
    ['TooT-BERT-C', 'IC-IT', 'ProtBERT-BFD', 'LR', 85.38, 0.71],
    ['Proposed method', 'IC-MP', 'ESM-1b', 'LR', 98.24, 0.85],
    ['Proposed method', 'IT-MP', 'ESM-1b', 'LR', 95.98, 0.69],
    ['Proposed method', 'IC-IT', 'ESM-2', 'CNN', 93.85, 0.87]
]

columns = ['Project', 'Task', 'Encoder', 'Classifier', 'Accuracy', 'MCC']
df = pd.DataFrame(data, columns=columns)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")
ax = plt.figure(figsize=(12, 6))

# Create a custom color palette with a highlighted color for the Proposed project
palette = sns.color_palette("deep")


# Function to adjust the alpha value of the bars
def custom_barplot(data, x, y, hue, palette, highlight_index):
    unique_hue = data[hue].unique()
    unique_x = data[x].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, hue_val in enumerate(unique_hue):
        for j, x_val in enumerate(unique_x):
            subset = data[(data[hue] == hue_val) & (data[x] == x_val)]
            color = palette[i]

            # If the project is the "Proposed" project, increase the alpha value
            alpha = 1.0 if j == highlight_index else 0.8

            ax.bar(
                x=j + i / (len(unique_hue) + 1),
                height=subset[y].values[0],
                width=1 / (len(unique_hue) + 1),
                color=color,
                alpha=alpha,
            )

    return ax

ax = custom_barplot(df, "Project", "MCC", "Task", palette, 3)

# Add horizontal lines for the baseline DeepIon and MFPS_CNN projects
deepion_mcc = df[df["Project"] == "DeepIon"]["MCC"].mean()
plt.axhline(y=deepion_mcc, linestyle='--', color='black')

mfps_cnn_mcc = df[df["Project"] == "MFPS_CNN"]["MCC"].mean()
plt.axhline(y=mfps_cnn_mcc, linestyle='--', color='gray')

# Add horizontal lines for the baseline DeepIon and MFPS_CNN projects
plt.axhline(y=deepion_mcc, linestyle='--', color='black')
plt.axhline(y=mfps_cnn_mcc, linestyle='--', color='gray')

# Add vertical line to separate MFPS_CNN and TooT-BERT-C
plt.axvline(x=1.6, linestyle=':', color='black')

# We make the xticks in the middle of the bars
plt.xticks(np.arange(0.1, len(df["Project"].unique()) + 0.1, 1), df["Project"].unique())
plt.ylabel("MCC")

# Create a custom legend
legend_elements = [plt.Line2D([0], [0], color=palette[i], lw=4, label=label) for i, label in enumerate(df["Task"].unique())]
plt.legend(handles=legend_elements, loc="upper left", title="Task")

# Annotate the bars with the MCC values
# Loop through each bar in the plot
for p in ax.patches:
    # Get the bar's x and y coordinates and its height
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    height = p.get_height()
    # Write the value of the bar on top of it
    ax.text(x, y, f'{height:.2f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(settings.LATEX_PATH, "independent_soa_cnn.png"), bbox_inches='tight', dpi=300)
plt.close()

# We load the results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH + "results_best_test.csv"))

# We take the best results from each model by MCC
df_best = df.sort_values(by=["MCC"], ascending=False).groupby("Task").head(1)

# We make a new dataframe with the best results with columns: Project, Task, Encoder, Classifier, Accuracy, MCC
new_df = []
for row in df_best.itertuples():
    new_df.append(["Proposed", row.Task, row.Representer, row.Classifier, row.Accuracy, row.MCC])

df_best_results = pd.DataFrame(new_df, columns=["Project", "Task", "Encoder", "Classifier", "Accuracy", "MCC"])

df_best_results["Task"] = df_best_results["Task"].map(settings.TASKS_SHORT)

# We change the order of the rows by the order of the tasks
order = ["IC-MP", "IT-MP", "IC-IT"]
df_best_results = df_best_results.set_index("Task").reindex(order).reset_index()

# We round the values of the MCC column
df_best_results["MCC"] = df_best_results["MCC"].apply(lambda x: np.round(x, 2))

# We multiply the Accuracy values by 100 and round them to 2 decimal places and if there is one zero after the decimal point, we put two zeros
df_best_results["Accuracy"] = df_best_results["Accuracy"].apply(lambda x: np.round(x * 100, 2))

# We make a latex table out of the df_table_short dataframe
# We create a latex table from the dataframe with the header boldface
latex_table = df_best_results.to_latex(
    index=False, escape=True, column_format='l' + 'c' * (len(df_best_results.columns) - 1))

# Replace the column names with their boldface versions
for col in df_best_results.columns:
    latex_table = latex_table.replace(col, '\\textbf{' + col + '}')

# Save the modified LaTeX table to a file
with open(os.path.join(settings.LATEX_PATH, "test_soa_results.tex"), 'w') as f:
    f.write(latex_table)

