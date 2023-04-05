import pandas as pd
import os
from settings import settings
import numpy as np


# We load the results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH + "results_toot_bert_c_test.csv"))

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

