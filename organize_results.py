import pandas as pd
import numpy as np
import os
from settings import settings
import json

# We filter the results that start with "gridsearch_results" and end with ".csv" from settings.RESULTS_PATH
results = [f for f in os.listdir(settings.RESULTS_PATH) if f.startswith(
    "gridsearch_results") and f.endswith(".csv")]

# Then from results we filter those that contain "ionchannels_membraneproteins"
results_ionchannels_membraneproteins = [
    f for f in results if "ionchannels_membraneproteins" in f]

# We do the same wiht "iontransporters_membraneproteins"
results_iontransporters_membraneproteins = [
    f for f in results if "iontransporters_membraneproteins" in f]

# And with "ionchannels_iontransporters"
results_ionchannels_iontransporters = [
    f for f in results if "ionchannels_iontransporters" in f]


representers = settings.REPRESENTATIONS
representation_types = [settings.FROZEN, settings.FINETUNED]
precision_types = ["half", "full"]
dataset_types = ["imbalanced", "balanced"]

representer_models = [f"{typ}_{rep['name']}" for rep in representers for typ in representation_types if rep['name'] not in [settings.PROTT5['name'],
                                                                                                             settings.ESM2_15B['name']]] + [f"{settings.FROZEN}_{settings.PROTT5['name']}", f"{settings.FROZEN}_{settings.ESM2_15B['name']}"]
# --------------------- ionchannels_membraneproteins imbalanced ---------------------
# -----------------------------------------------------------------------------------

# Table for ionchannels_membraneproteins imbalanced
results_ionchannels_membraneproteins_imbalanced = [
    f for f in results_ionchannels_membraneproteins if "imbalanced" in f]
df_table_ionchannels_membraneproteins_imbalanced = pd.DataFrame()
for rep_typ in representer_models:
    # Load csv file that ends with rep_typ.csv
    file = [f for f in results_ionchannels_membraneproteins_imbalanced if f.endswith(
        f"{rep_typ}.csv")][0]
    df = pd.read_csv(os.path.join(settings.RESULTS_PATH, file), names=[
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

    # We make the latex table as follows: 1st column is rep, 2nd column is typ, and the rest of columns are the df_model_mcc
    representation = rep_typ.split("_")[1] if len(rep_typ.split(
        "_")) == 2 else rep_typ.split("_")[1] + "_" + rep_typ.split("_")[2]
    type = rep_typ.split("_")[0]
    df_table_ionchannels_membraneproteins_imbalanced = pd.concat(
        [df_table_ionchannels_membraneproteins_imbalanced, pd.concat([pd.DataFrame([[representation, type]], columns=["Representation", "Type"]), df_model_mcc], axis=1)], axis=0)
# We reset the index
df_table_ionchannels_membraneproteins_imbalanced = df_table_ionchannels_membraneproteins_imbalanced.reset_index(drop=True)
# We make the latex table
df_table_ionchannels_membraneproteins_imbalanced.to_latex(
    os.path.join(settings.LATEX_PATH, "table_ionchannels_membraneproteins_imbalanced.tex"), index=False, escape=False)


# --------------------- ionchannels_membraneproteins balanced ---------------------
# -----------------------------------------------------------------------------------

# Table for ionchannels_membraneproteins balanced
results_ionchannels_membraneproteins_balanced = [
    f for f in results_ionchannels_membraneproteins if f.split("_")[4] == "balanced"]
df_table_ionchannels_membraneproteins_balanced = pd.DataFrame()
for rep_typ in representer_models:
    files = [f for f in results_ionchannels_membraneproteins_balanced if f.endswith(
        f"{rep_typ}.csv")]
    df_dict = {}
    for file in files:
        df = pd.read_csv(os.path.join(settings.RESULTS_PATH, file))
        if "svm" in df_dict:
            df_dict["svm"].append(eval(df["svm"][3])["Val"])  # 3 is the MCC
        else:
            df_dict["svm"] = [eval(df["svm"][3])["Val"]]
        if "rf" in df_dict:
            df_dict["rf"].append(eval(df["rf"][3])["Val"])
        else:
            df_dict["rf"] = [eval(df["rf"][3])["Val"]]
        if "knn" in df_dict:
            df_dict["knn"].append(eval(df["knn"][3])["Val"])
        else:
            df_dict["knn"] = [eval(df["knn"][3])["Val"]]
        if "lr" in df_dict:
            df_dict["lr"].append(eval(df["lr"][3])["Val"])
        else:
            df_dict["lr"] = [eval(df["lr"][3])["Val"]]
        if "mlp" in df_dict:
            df_dict["mlp"].append(eval(df["mlp"][3])["Val"])
        else:
            df_dict["mlp"] = [eval(df["mlp"][3])["Val"]]

    # We take average of mean +- std for each model in df_dict
    for key in df_dict:
        mean_sd = [value.split("±") for value in df_dict[key]]
        # We take the mean of the mean and the mean of the std
        mean = np.mean([float(value[0]) for value in mean_sd])
        std = np.mean([float(value[1]) for value in mean_sd])
        df_dict[key] = f"{mean:.2f}±{std:.2f}"

    # We make a dataframe with the df_dict with the keys as columns
    df_model_mcc = pd.DataFrame.from_dict(
        df_dict, orient="index", columns=["MCC"])
    # We transpose the dataframe, so rows are the columns and columns are the rows
    df_model_mcc = df_model_mcc.transpose()
    # We reset the index
    df_model_mcc = df_model_mcc.reset_index()
    # We remove Metric column
    df_model_mcc = df_model_mcc.drop(columns=["index"])
    # We change the column names
    df_model_mcc = df_model_mcc.rename(
        columns={"svm": "SVM", "rf": "RF", "knn": "kNN", "lr": "LR", "mlp": "FFNN"})
    # We make the latex table as follows: 1st column is rep, 2nd column is typ, and the rest of columns are the df_model_mcc
    representation = rep_typ.split("_")[1] if len(rep_typ.split(
        "_")) == 2 else rep_typ.split("_")[1] + "_" + rep_typ.split("_")[2]
    type = rep_typ.split("_")[0]
    df_table_ionchannels_membraneproteins_balanced = pd.concat(
        [df_table_ionchannels_membraneproteins_balanced, pd.concat([pd.DataFrame([[representation, type]], columns=["Representation", "Type"]), df_model_mcc], axis=1)], axis=0)
# We reset the index
df_table_ionchannels_membraneproteins_balanced = df_table_ionchannels_membraneproteins_balanced.reset_index()
# We make the latex table
df_table_ionchannels_membraneproteins_balanced.to_latex(
    os.path.join(settings.LATEX_PATH, "table_ionchannels_membraneproteins_balanced.tex"), index=False, escape=False)


# --------------------- iontransporters_membraneproteins imbalanced ---------------------
# -----------------------------------------------------------------------------------

# Table for iontransporters_membraneproteins balanced
results_iontransporters_membraneproteins_balanced = [
    f for f in results_iontransporters_membraneproteins if f.split("_")[4] == "balanced"]
df_table_iontransporters_membraneproteins_balanced = pd.DataFrame()
for rep_typ in representer_models:
    files = [f for f in results_iontransporters_membraneproteins_balanced if f.endswith(
        f"{rep_typ}.csv")]
    df_dict = {}
    for file in files:
        df = pd.read_csv(os.path.join(settings.RESULTS_PATH, file))
        if "svm" in df_dict:
            df_dict["svm"].append(eval(df["svm"][3])["Val"])
        else:
            df_dict["svm"] = [eval(df["svm"][3])["Val"]]
        if "rf" in df_dict:
            df_dict["rf"].append(eval(df["rf"][3])["Val"])
        else:
            df_dict["rf"] = [eval(df["rf"][3])["Val"]]
        if "knn" in df_dict:
            df_dict["knn"].append(eval(df["knn"][3])["Val"])
        else:
            df_dict["knn"] = [eval(df["knn"][3])["Val"]]
        if "lr" in df_dict:
            df_dict["lr"].append(eval(df["lr"][3])["Val"])
        else:
            df_dict["lr"] = [eval(df["lr"][3])["Val"]]
        if "mlp" in df_dict:
            df_dict["mlp"].append(eval(df["mlp"][3])["Val"])
        else:
            df_dict["mlp"] = [eval(df["mlp"][3])["Val"]]

    # We take average of mean +- std for each model in df_dict
    for key in df_dict:
        mean_sd = [value.split("±") for value in df_dict[key]]
        # We take the mean of the mean and the mean of the std
        mean = np.mean([float(value[0]) for value in mean_sd])
        std = np.mean([float(value[1]) for value in mean_sd])
        df_dict[key] = f"{mean:.2f}±{std:.2f}"

    # We make a dataframe with the df_dict with the keys as columns
    df_model_mcc = pd.DataFrame.from_dict(
        df_dict, orient="index", columns=["MCC"])
    # We transpose the dataframe, so rows are the columns and columns are the rows
    df_model_mcc = df_model_mcc.transpose()
    # We reset the index
    df_model_mcc = df_model_mcc.reset_index()
    # We remove Metric column
    df_model_mcc = df_model_mcc.drop(columns=["index"])
    # We change the column names
    df_model_mcc = df_model_mcc.rename(
        columns={"svm": "SVM", "rf": "RF", "knn": "kNN", "lr": "LR", "mlp": "FFNN"})
    
    # We make the latex table as follows: 1st column is rep, 2nd column is typ, and the rest of columns are the df_model_mcc
    representation = rep_typ.split("_")[1] if len(rep_typ.split(
        "_")) == 2 else rep_typ.split("_")[1] + "_" + rep_typ.split("_")[2]
    type = rep_typ.split("_")[0]
    df_table_iontransporters_membraneproteins_balanced = pd.concat(
        [df_table_iontransporters_membraneproteins_balanced, pd.concat([pd.DataFrame([[representation, type]], columns=["Representation", "Type"]), df_model_mcc], axis=1)], axis=0)

# We reset the index
df_table_iontransporters_membraneproteins_balanced = df_table_iontransporters_membraneproteins_balanced.reset_index()
# We make the latex table
df_table_iontransporters_membraneproteins_balanced.to_latex(
    os.path.join(settings.LATEX_PATH, "table_iontransporters_membraneproteins_balanced.tex"), index=False, escape=False)

# --------------------- iontransporters_membraneproteins imbalanced ---------------------
# ---------------------------------------------------------------------------------------

# Table for iontransporters_membraneproteins imbalanced
results_iontransporters_membraneproteins_imbalanced = [
    f for f in results_iontransporters_membraneproteins if "imbalanced" in f]
df_table_iontransporters_membraneproteins_imbalanced = pd.DataFrame()
for rep_typ in representer_models:
    # Load csv file that ends with rep_typ.csv
    file = [f for f in results_iontransporters_membraneproteins_imbalanced if f.endswith(
        f"{rep_typ}.csv")][0]
    df = pd.read_csv(os.path.join(settings.RESULTS_PATH, file), names=[
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
    # We make the latex table as follows: 1st column is rep, 2nd column is typ, and the rest of columns are the df_model_mcc
    representation = rep_typ.split("_")[1] if len(rep_typ.split(
        "_")) == 2 else rep_typ.split("_")[1] + "_" + rep_typ.split("_")[2]
    type = rep_typ.split("_")[0]
    df_table_iontransporters_membraneproteins_imbalanced = pd.concat(
        [df_table_iontransporters_membraneproteins_imbalanced, pd.concat([pd.DataFrame([[representation, type]], columns=["Representation", "Type"]), df_model_mcc], axis=1)], axis=0)
# We reset the index
df_table_iontransporters_membraneproteins_imbalanced = df_table_iontransporters_membraneproteins_imbalanced.reset_index()
# We make the latex table
df_table_iontransporters_membraneproteins_imbalanced.to_latex(
    os.path.join(settings.LATEX_PATH, "table_iontransporters_membraneproteins_imbalanced.tex"), index=False, escape=False)


# --------------------- ionchannels_iontransporters ---------------------
# ----------------------------------------------------------------------

# Table for ionchannels_iontransporters
df_table_ionchannels_iontransporters = pd.DataFrame()
for rep_typ in representer_models:
    # Load csv file that ends with rep_typ.csv
    file = [f for f in results_ionchannels_iontransporters if f.endswith(
        f"{rep_typ}.csv")][0]
    df = pd.read_csv(os.path.join(settings.RESULTS_PATH, file), names=[
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

    # We make the latex table as follows: 1st column is rep, 2nd column is typ, and the rest of columns are the df_model_mcc
    representation = rep_typ.split("_")[1] if len(rep_typ.split(
        "_")) == 2 else rep_typ.split("_")[1] + "_" + rep_typ.split("_")[2]
    type = rep_typ.split("_")[0]
    df_table_ionchannels_iontransporters = pd.concat(
        [df_table_ionchannels_iontransporters, pd.concat([pd.DataFrame([[representation, type]], columns=["Representation", "Type"]), df_model_mcc], axis=1)], axis=0)

# We reset the index
df_table_ionchannels_iontransporters = df_table_ionchannels_iontransporters.reset_index(
    drop=True)

# We make the latex table
df_table_ionchannels_iontransporters.to_latex(os.path.join(
    settings.LATEX_PATH, "table_ionchannels_iontransporters.tex"), index=False, escape=False)
