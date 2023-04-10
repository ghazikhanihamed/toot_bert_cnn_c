import os
import pandas as pd
from settings import settings

# We read the csv file of the full results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH,
                 "mean_balanced_imbalanced_results_trad_cnn.csv"))

# For each task, we make different tables with MCC
tasks = settings.TASKS

for task in tasks:
    if task == settings.IONCHANNELS_MEMBRANEPROTEINS or task == settings.IONTRANSPORTERS_MEMBRANEPROTEINS:
        continue
    df_temp = df[df["Task"] == task]
    # Order by MCC
    df_temp = df_temp.sort_values(by=["MCC"], ascending=False)

    # create a list of all possible combinations
    representers = df_temp['Representer'].unique()
    representations = df_temp['Representation'].unique()
    precisions = df_temp['Precision'].unique()
    classifiers = df_temp['Classifier'].unique()

    # create an empty dataframe to store the data
    new_df = pd.DataFrame()

    # loop over all possible combinations and add the MCC value from the initial dataframe to the new dataframe
    for representer in representers:
        for representation in representations:
            for precision in precisions:
                row = {
                    'Representer': representer,
                    'Representation': representation,
                    'Precision': precision
                }
                for classifier in classifiers:
                    match = df_temp.loc[(df_temp['Representer']==representer) & (df_temp['Representation']==representation) & (df_temp['Precision']==precision) & (df_temp['Classifier']==classifier)]
                    if len(match) > 0:
                        row[classifier] = match['MCC'].values[0]
                    else:
                        row[classifier] = float('nan')
                new_df = new_df.append(row, ignore_index=True)

    # re-order the columns
    new_df = new_df[['Representer', 'Representation', 'Precision', 'SVM', 'RF', 'kNN', 'LR', 'FFNN']]

    # We replace the NaN values with a dash
    new_df = new_df.fillna('-')

    # We create a latex table from the dataframe with the header boldface
    latex_table = new_df.to_latex(
        index=False, escape=True, column_format='l' + 'c' * (len(new_df.columns) - 1))

    # Replace the column names with their boldface versions
    for col in new_df.columns:
        latex_table = latex_table.replace(col, '\\textbf{' + col + '}')

    # Save the modified LaTeX table to a file
    with open(os.path.join(settings.LATEX_PATH, "table_detail_" + task + "_trad_cnn.tex"), "w") as f:
        f.write(latex_table)

