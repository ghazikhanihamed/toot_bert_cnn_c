import pandas as pd
import os
from settings import settings

# We read the csv file of the full results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH, "mean_balanced_imbalanced_results.csv"))

# The dataframe has the following columns: "Task", "Dataset", "Representation", "Representer", "Precision", "Classifier", "MCC"


# We make a new dataframe out of the df dataframe, focusing on the Dataset column.
# The new dataframe has the following columns: "Task", "Dataset", where we group by the representer under each category of "Task", "Dataset".
# Then we take the best MCC value for each category of "Task", "Dataset".

representation_types = [settings.FROZEN, settings.FINETUNED]
precision_types = ["half", "full"]
representers = settings.REPRESENTATIONS
# tasks without ionchannels_iontransporters
tasks = settings.TASKS[:-1]
datasets = ["balanced", "imbalanced"]

ds_best_mcc = []
for task in tasks:
    for dataset in datasets:
        df_temp = df[(df["Task"] == task) & (df["Dataset"] == dataset)]
        if not df_temp.empty:
            for representer in representers:
                df_temp2 = df_temp[df_temp["Representer"] == representer["name"]]
                if not df_temp2.empty:
                    best_mcc = df_temp2["MCC"].str.split("±").str[0].astype(float).max()
                    ds_best_mcc.append([task, dataset, representer["name"], best_mcc])

# We create the dataframe
df_ds_best_mcc = pd.DataFrame(ds_best_mcc, columns=["Task", "Dataset", "Representer", "MCC"])

a=1