import pandas as pd
import numpy as np
import os
from settings import settings
import json

# # We filter the results that start with "gridsearch_results" and end with ".csv" from settings.RESULTS_PATH
# results_garbage = [f for f in os.listdir(settings.RESULTS_PATH) if f.startswith(
#     "CNN_CV_results") and not (f.endswith("full.csv") or f.endswith("half.csv"))]

# # We remove the garbage results
# for result in results_garbage:
#     os.remove(os.path.join(settings.RESULTS_PATH, result))


results = [f for f in os.listdir(settings.RESULTS_PATH) if f.startswith(
    "CNN_CV_results") and f.endswith(".csv")]


representers = settings.REPRESENTATIONS
representation_types = [settings.FROZEN, settings.FINETUNED]
precision_types = ["half", "full"]
dataset_types = ["imbalanced", "balanced"]

representer_models = [f"{typ}_{rep['name']}" for rep in representers for typ in representation_types if rep['name'] not in [settings.PROTT5['name'],
                                                                                                                            settings.ESM2_15B['name']]] + [f"{settings.FROZEN}_{settings.PROTT5['name']}", f"{settings.FROZEN}_{settings.ESM2_15B['name']}"]


# We will make a pandas dataframe with all the results. The columns will be: task (ionchannels_membraneproteins, iontransporters_membraneproteins, ionchannels_iontransporters),
# precision (half, full), dataset (imbalanced, balanced), representer (prott5, esm2_15b, ...), representation (frozen, finetuned), classifier (CNN).
# All the information is stored in the name of the file.
# The name of the file is: CNN_CV_results_ + dataset_name + _ + dataset_type + _ + dataset_number + _ + representation_type + _ + representer_model + _ + precision_type
# For each fold, the mean of the metrics is stored in the csv file, so there are 5 rows for each fold and 4 columns for each metric (Sensitivity, Specificity, Accuracy, MCC).

df_results = []
for result in results:
    task = result.split("_")[3]+"_"+result.split("_")[4]
    dataset = result.split("_")[5]
    dataset_number = result.split("_")[6]
    representation = result.split("_")[7]
    if len(result.split("_")) == 10:
        precision = result.split("_")[9].split(".")[0]
        representer = result.split("_")[8]
    else:
        precision = result.split("_")[10].split(".")[0]
        representer = result.split("_")[8]+"_"+result.split("_")[9]

    # We read the csv file and consider the first row as the header
    df = pd.read_csv(os.path.join(settings.RESULTS_PATH, result), header=0)

    # We compute the mean and standard deviation of the metrics for each metric
    sensitivity_mean = np.mean(df["Sensitivity"])
    sensitivity_std = np.std(df["Sensitivity"])
    sensitivity = '{:.2f}'.format(round(sensitivity_mean, 2) * 100) + \
        u"\u00B1" + '{:.2f}'.format(round(sensitivity_std, 2) * 100)
    specificity_mean = np.mean(df["Specificity"])
    specificity_std = np.std(df["Specificity"])
    specificity = '{:.2f}'.format(round(specificity_mean, 2) * 100) + \
        u"\u00B1" + '{:.2f}'.format(round(specificity_std, 2) * 100)
    accuracy_mean = np.mean(df["Accuracy"])
    accuracy_std = np.std(df["Accuracy"])
    accuracy = '{:.2f}'.format(round(accuracy_mean, 2) * 100) + \
        u"\u00B1" + '{:.2f}'.format(round(accuracy_std, 2) * 100)
    mcc_mean = np.mean(df["MCC"])
    mcc_std = np.std(df["MCC"])
    mcc = '{:.2f}'.format(round(mcc_mean, 2)) + u"\u00B1" + \
        '{:.2f}'.format(round(mcc_std, 2))
    
    df_results.append([task, dataset, dataset_number, representation, representer, precision, "CNN", sensitivity, specificity, accuracy, mcc])

    # We add the results to the a new dataframe


# We create a dataframe with the results
df_cnn = pd.DataFrame(df_results, columns=[
    "Task", "Dataset", "Dataset_number", "Representation", "Representer", "Precision", "Classifier", "Sensitivity", "Specificity", "Accuracy", "MCC"])

# We save the dataframe in a csv file
df_cnn.to_csv(os.path.join(settings.RESULTS_PATH, "full_cnn_results.csv"), index=False)

# We read the csv file of the full results
df_trad = pd.read_csv(os.path.join(settings.RESULTS_PATH, "full_results.csv"))

# We add the results of the CNN to the dataframe of the traditional classifiers
df_trad = df_trad.append(df_cnn, ignore_index=True)

# We save the dataframe in a csv file
df_trad.to_csv(os.path.join(settings.RESULTS_PATH, "full_trad_cnn_results.csv"), index=False)
