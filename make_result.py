import pandas as pd
import numpy as np
import os
from settings import settings
import json

# We filter the results that start with "gridsearch_results" and end with ".csv" from settings.RESULTS_PATH
results = [f for f in os.listdir(settings.RESULTS_PATH) if f.startswith(
    "gridsearch_results") and f.endswith(".csv")]


representers = settings.REPRESENTATIONS
representation_types = [settings.FROZEN, settings.FINETUNED]
precision_types = ["half", "full"]
dataset_types = ["imbalanced", "balanced"]

representer_models = [f"{typ}_{rep['name']}" for rep in representers for typ in representation_types if rep['name'] not in [settings.PROTT5['name'],
                                                                                                                            settings.ESM2_15B['name']]] + [f"{settings.FROZEN}_{settings.PROTT5['name']}", f"{settings.FROZEN}_{settings.ESM2_15B['name']}"]


# We will make a pandas dataframe with all the results. The columns will be: task (ionchannels_membraneproteins, iontransporters_membraneproteins, ionchannels_iontransporters),
# precision (half, full), dataset (imbalanced, balanced), representer (prott5, esm2_15b, ...), representation (frozen, finetuned), classifier (svm, rf, ...).
# All the information is stored in the name of the file, except the classifier which is the columns of the csv files.
# The name of the file is: "gridsearch_results_ + dataset_name + _ + dataset_type + _ + dataset_number + _ + representation_type + _ + representer_model + _ + precision_type"
# We will take the values of row "MCC" for the results.
# The row "MCC" have the this structure: MCC,"{'Train': 'x.xx±x.xx', 'Val': 'x.xx±x.xx'}", repeated for each classifier.
# We only need the the "Val" part, so we will take the second element of the list and then the second element of the dictionary.
# If in the name it is not mentioned full, then it is half precision.

df_results = []
for result in results:
    task = result.split("_")[2]+"_"+result.split("_")[3]
    dataset = result.split("_")[4]
    dataset_number = result.split("_")[5]
    representation = result.split("_")[6]
    if len(result.split("_")) == 9:
        if "full" in result:
            precision = "full"
            representer = result.split("_")[7].split(".")[0]
        else:
            precision = "half"
            representer = result.split(
                "_")[7]+"_"+result.split("_")[8].split(".")[0]
    else:
        precision = "half"
        representer = result.split("_")[7].split(".")[0]

    df = pd.read_csv(os.path.join(settings.RESULTS_PATH, result), names=[
        "Metric", "SVM", "RF", "kNN", "LR", "FFNN"])
    df = df.drop(df.index[0])
    # We transpose the dataframe, so rows are the columns and columns are the rows
    df = df.transpose()
    # We rename the columns with the first row
    df.columns = df.iloc[0]
    # We drop the first row
    df = df.drop(df.index[0])
    # We reset the index
    df = df.reset_index()
    # We change the name of the first column
    df = df.rename(columns={"index": "Model"})

    # We take the model and MCC column for MCC we take the value of the key "Val" with two separate columns
    df_model_mcc = df[["Model", "MCC"]]
    df_model_mcc["MCC"] = df_model_mcc["MCC"].apply(lambda x: eval(x)["Val"])

    # We transpose the dataframe, so rows are the columns and columns are the rows
    df_model_mcc = df_model_mcc.transpose()

    # We rename the columns with the first row
    df_model_mcc.columns = df_model_mcc.iloc[0]
    # We drop the first row
    df_model_mcc = df_model_mcc.drop(df_model_mcc.index[0])
    # We reset the index
    df_model_mcc = df_model_mcc.reset_index()
    # We remove Metric column
    df_model_mcc = df_model_mcc.drop(columns=["Metric"])

    # For each classifier (columns of the df_model_mcc dataframe) we take the value of the MCC
    for classifier in df_model_mcc.columns:
        df_results.append(
            [task, dataset, dataset_number, representation, representer, precision, classifier, df_model_mcc[classifier][0]])


# We create a dataframe with the results
df = pd.DataFrame(df_results, columns=[
    "Task", "Dataset", "Dataset_number", "Representation", "Representer", "Precision", "Classifier", "MCC"])

# We save the dataframe in a csv file
df.to_csv(os.path.join(settings.RESULTS_PATH, "full_results.csv"), index=False)
