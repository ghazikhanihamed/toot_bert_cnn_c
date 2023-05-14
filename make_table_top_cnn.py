import pandas as pd
import os
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt

# We read the csv file of the full results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH,
                 "mean_balanced_imbalanced_results_trad_cnn.csv"))

# The dataframe has the following columns: "Task", "Dataset", "Representation", "Representer", "Precision", "Classifier",
# We will find the best MCC for each task

tasks = settings.TASKS

ds_best_mcc = []
for task in tasks:
    df_temp = df[df["Task"] == task]
    if not df_temp.empty:
        # for each classifier, we find the best MCC
        for classifier in df_temp["Classifier"].unique():
            df_temp_classifier = df_temp[df_temp["Classifier"] == classifier]
            
            # Best top 1 MCC from df_temp_classifier    
            best_mcc = df_temp_classifier["MCC"].max()

            # We find the row with the best MCC and select only the first row
            df_temp_best_mcc = df_temp_classifier[df_temp_classifier["MCC"] == best_mcc].iloc[0]

            # we make the df_temp_best_mcc a dataframe with one row
            df_temp_best_mcc = pd.DataFrame(df_temp_best_mcc).transpose()

            # We append the row with the best MCC to the list
            ds_best_mcc.append(df_temp_best_mcc)

# We create a dataframe from ds_best_mcc
df_table = pd.concat(ds_best_mcc)

# We create a dataframe from df_table with short names of the tasks, datasets and representations, representer, precision,  classifier, and MCC
df_table_short = df_table.copy()
df_table_short["Task"] = df_table_short["Task"].map(settings.TASKS_SHORT)

# df_table_short["mean_mcc"] = df_table_short["MCC"].str.split("±").str[0].astype(float)

# # Order df_table_short first by Tast and then by "mean_mcc"
# df_table_short = df_table_short.sort_values(["Task", "mean_mcc"], ascending=[False, False])

# we change the order of the columns
df_table_short = df_table_short[["Task", "Dataset", "Representation",
                                    "Representer", "Classifier", "MCC", "Accuracy", "Sensitivity", "Specificity"]]

# We change the names of the columns of Accuracy, Sensitivity, Specificity to Acc, Sen, Spc
df_table_short = df_table_short.rename(columns={"Accuracy": "Acc", "Sensitivity": "Sen", "Specificity": "Spc"})

# We replace the 'na' values with a dash of the Dataset column
df_table_short = df_table_short.replace({'Dataset': {'na': '-'}})

# We make a latex table out of the df_table_short dataframe
# We create a latex table from the dataframe with the header boldface
latex_table = df_table_short.to_latex(
    index=False, escape=True, column_format='l' + 'c' * (len(df_table_short.columns) - 1))

# Replace the column names with their boldface versions
for col in df_table_short.columns:
    latex_table = latex_table.replace(col, '\\textbf{' + col + '}')

# Save the modified LaTeX table to a file
with open(os.path.join(settings.LATEX_PATH, "best_results_cnn.tex"), 'w') as f:
    f.write(latex_table)
