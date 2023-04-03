import pandas as pd
import os
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt

# We read the csv file of the full results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH,
                 "mean_balanced_imbalanced_results.csv"))

# The dataframe has the following columns: "Task", "Dataset", "Representation", "Representer", "Precision", "Classifier",
# We will find the best MCC for each task

tasks = settings.TASKS

ds_best_mcc = []
for task in tasks:
    df_temp = df[df["Task"] == task]
    if not df_temp.empty:
        # We take the first three rows of the df_temp2 sorted by MCC value split by "±" and take the first element
        three_best_mcc = df_temp["MCC"].str.split(
            "±").str[0].astype(float).nlargest(3).index.tolist()
        df_three_best_mcc = df_temp.loc[three_best_mcc]

        # We take the best MCC value for each category of "Task", "Dataset" and "Representer"
        best_mcc_id = df_temp["MCC"].str.split(
            "±").str[0].astype(float).idxmax()

        df_best_mcc = df_temp.loc[best_mcc_id]
        ds_best_mcc.append(df_three_best_mcc)
        # ds_best_mcc.append(df_best_mcc)

df_table = pd.concat(ds_best_mcc)

# We create a dataframe from df_table with short names of the tasks, datasets and representations, representer, precision,  classifier, and MCC
df_table_short = df_table.copy()
df_table_short["Task"] = df_table_short["Task"].map(settings.TASKS_SHORT)
df_table_short["Dataset"] = df_table_short["Dataset"]
df_table_short["Representation"] = df_table_short["Representation"]
df_table_short["Representer"] = df_table_short["Representer"]
df_table_short["Precision"] = df_table_short["Precision"]
df_table_short["Classifier"] = df_table_short["Classifier"]
df_table_short["MCC"] = df_table_short["MCC"]

# We don't need the sensitivity, specificity, and accuracy columns
df_table_short = df_table_short.drop(
    columns=["Sensitivity", "Specificity", "Accuracy"])


# We make a latex table out of the df_table_short dataframe
# We create a latex table from the dataframe with the header boldface
latex_table = df_table_short.to_latex(
    index=False, escape=True, column_format='l' + 'c' * (len(df_table_short.columns) - 1))

# Replace the column names with their boldface versions
for col in df_table_short.columns:
    latex_table = latex_table.replace(col, '\\textbf{' + col + '}')

# Save the modified LaTeX table to a file
with open(os.path.join(settings.LATEX_PATH, "best_results.tex"), 'w') as f:
    f.write(latex_table)
