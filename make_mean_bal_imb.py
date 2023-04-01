import pandas as pd
import os
from settings import settings

# We read the csv file of the full results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH, "full_results.csv"))

# The dataframe has the following columns: "Task", "Dataset", "Dataset_number", "Representation", "Representer", "Precision", "Classifier", "MCC"
# We want to create a dataframe with the following columns: "Task", "Dataset", "Representation", "Representer", "Precision", "Classifier", "MCC", where
# we filter the results by "Dataset" if they are balanced. Then for each category of "Task", "Representation", "Representer", "Precision", "Classifier",
# we take the mean of the mean and std which are separated by ±.

# We filter the dataframe by "Dataset" if they are balanced
df_balanced = df[df["Dataset"] == "balanced"]

# We filter the rest of the dataframe by "Dataset" if they are not balanced
df_imbalanced = df[df["Dataset"] != "balanced"]
# We remove the "Dataset_number" column
df_imbalanced = df_imbalanced.drop(columns=["Dataset_number"])

# For each category of "Task", "Representation", "Representer", "Precision", "Classifier", we take the mean of the mean and std which are separated by ±.
representation_types = [settings.FROZEN, settings.FINETUNED]
precision_types = ["half", "full"]
representers = settings.REPRESENTATIONS
tasks = settings.TASKS

classifiers = ["SVM", "RF", "kNN", "LR", "FFNN"]

mean_balanced_results = []
for representer in representers:
    for precision in precision_types:
        for classifier in classifiers:
            for representation_type in representation_types:
                for task in tasks:
                    df_temp = df_balanced[(df_balanced["Representer"] == representer["name"]) & (df_balanced["Precision"] == precision) & (df_balanced["Classifier"] == classifier) & (df_balanced["Representation"] == representation_type) & (df_balanced["Task"] == task)]
                    if not df_temp.empty:
                        # We split the mean and std of the MCC column by ±. Then we take the mean of the mean and std.
                        mean = df_temp["MCC"].str.split("±").str[0].astype(float).mean()
                        std = df_temp["MCC"].str.split("±").str[1].astype(float).mean()
                        mean_balanced_results.append([task, "balanced", representation_type, representer["name"], precision, classifier, f"{mean:.2f}±{std:.2f}"])

# We create the dataframe
df_mean_balanced_results = pd.DataFrame(mean_balanced_results, columns=["Task", "Dataset", "Representation", "Representer", "Precision", "Classifier", "MCC"])

# We merge the df_mean_balanced_results dataframe with the df_imbalanced dataframe
df_mean_results = pd.concat([df_mean_balanced_results, df_imbalanced])

# We reset the index
df_mean_results = df_mean_results.reset_index(drop=True)

# We save the dataframe
df_mean_results.to_csv(os.path.join(settings.RESULTS_PATH, "mean_balanced_imbalanced_results.csv"), index=False)