import pandas as pd
import os
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt
# import svm, ffnn and lr, knn, rf
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import h5py
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score
import ast

import numpy as np
import random
import sklearn

# We set the random seed for reproducibility
random.seed(settings.SEED)
np.random.seed(settings.SEED)
sklearn.utils.check_random_state(settings.SEED)

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

# For each task, we find the three train and test sets from REPRESENTATIONS_FILTERED_PATH with the information in df_table, then we train the models based on the best params and test them on the test set
# we make a list of only h5 files that contains only train in the representations folder
results_list = []
config = []
for row in df_table.itertuples():
    task = row.Task
    dataset = row.Dataset
    representer = row.Representer
    representation_type = row.Representation
    precision = row.Precision
    prec = "_" + precision if precision == "full" else ""
    classifier = row.Classifier

    # We check if the config exists in the list
    #if [task, dataset, representation_type, representer, precision] in config:
    #    continue
    #else:
    #    config.append(
    #        [task, dataset, representation_type, representer, precision])

    params = pd.read_csv(os.path.join(settings.RESULTS_PATH, "gridsearch_best_params_" + task +
                         "_" + dataset + "_" + "na" + "_" + representation_type + "_" + representer + prec + ".csv"))
    if task == settings.IONCHANNELS_IONTRANSPORTERS:
        if representation_type == settings.FINETUNED and precision == "full":
            # The file name is : ionchannels_iontransporters_test_finetuned_representations_full_ESM-2_ionchannels_iontransporters.h5
            representation_train = f"{task}_train_{representation_type}_representations_{precision}_{representer}_{task}.h5"
            representation_test = f"{task}_test_{representation_type}_representations_{precision}_{representer}_{task}.h5"
        elif representation_type == settings.FINETUNED and precision == "half":
            # The file name is : ionchannels_iontransporters_test_finetuned_representations_ESM-2_ionchannels_iontransporters_imbalanced.h5
            representation_train = f"{task}_train_{representation_type}_representations_{representer}_{task}_imbalanced.h5"
            representation_test = f"{task}_test_{representation_type}_representations_{representer}_{task}_imbalanced.h5"
        elif representation_type == settings.FROZEN and precision == "full":
            # The file name is : ionchannels_iontransporters_test_frozen_representations_full_ESM-1b.h5
            representation_train = f"{task}_train_{representation_type}_representations_{precision}_{representer}.h5"
            representation_test = f"{task}_test_{representation_type}_representations_{precision}_{representer}.h5"
        elif representation_type == settings.FROZEN and precision == "half":
            # The file name is : ionchannels_iontransporters_test_frozen_representations_ESM-2_15B.h5
            representation_train = f"{task}_train_{representation_type}_representations_{representer}.h5"
            representation_test = f"{task}_test_{representation_type}_representations_{representer}.h5"
    else:
        if representation_type == settings.FINETUNED and precision == "full":
            # The file name is : ionchannels_membraneproteins_imbalanced_train_finetuned_representations_full_ProtBERT-BFD_ionchannels_membraneproteins_imbalanced.h5
            representation_train = f"{task}_{dataset}_train_{representation_type}_representations_{precision}_{representer}_{task}_imbalanced.h5"
            representation_test = f"{task}_{dataset}_test_{representation_type}_representations_{precision}_{representer}_{task}_imbalanced.h5"
        elif representation_type == settings.FINETUNED and precision == "half":
            # The file name is : ionchannels_membraneproteins_imbalanced_train_finetuned_representations_ProtBERT-BFD_ionchannels_membraneproteins_imbalanced.h5
            representation_train = f"{task}_{dataset}_train_{representation_type}_representations_{representer}_{task}_imbalanced.h5"
            representation_test = f"{task}_{dataset}_test_{representation_type}_representations_{representer}_{task}_imbalanced.h5"
        elif representation_type == settings.FROZEN and precision == "full":
            # The file name is : ionchannels_membraneproteins_imbalanced_train_frozen_representations_full_ESM-1b.h5
            representation_train = f"{task}_{dataset}_train_{representation_type}_representations_{precision}_{representer}.h5"
            representation_test = f"{task}_{dataset}_test_{representation_type}_representations_{precision}_{representer}.h5"
        elif representation_type == settings.FROZEN and precision == "half":
            # The file name is : ionchannels_membraneproteins_imbalanced_train_frozen_representations_ProtBERT-BFD.h5
            representation_train = f"{task}_{dataset}_train_{representation_type}_representations_{representer}.h5"
            representation_test = f"{task}_{dataset}_test_{representation_type}_representations_{representer}.h5"

    # We take the best params for the dataset
    svm_param_grid = {
        'C': float(params["svm"][0]),
        'gamma': float(params["svm"][1]),
        'kernel': params["svm"][2],
        'random_state': settings.SEED
    }

    lr_param_grid = {
        'penalty': params["lr"][9],
        'C': float(params["lr"][0]),
        'solver': params["lr"][10],
        'random_state': settings.SEED
    }

    mlp_param_grid = {
        'hidden_layer_sizes': ast.literal_eval(params["mlp"][12]),
        'activation': params["mlp"][11],
        'solver': params["mlp"][10],
        'random_state': settings.SEED
    }

    rf_param_grid = {
        'n_estimators': int(params["rf"][5]),
        'max_depth': int(params["rf"][3]) if pd.notnull(params["rf"][3]) else None,
        'min_samples_split': int(params["rf"][4]),
        'random_state': settings.SEED
    }

    knn_param_grid = {
        'n_neighbors': int(params["knn"][7]),
        'weights': params["knn"][8],
        'algorithm': params["knn"][6]
    }

    # We train the models with the best params and test them on the test set
    with h5py.File(settings.REPRESENTATIONS_FILTERED_PATH + representation_train, "r") as f:
        # We put the id, representation and label together in a list. The saved data is : (str(csv_id), data=representation), [str(csv_id)].attrs["label"] = label. And the representation is a numpy array
        train_data = [(id, representation, label) for id, representation in zip(
            f.keys(), f.values()) for label in f[id].attrs.values()]

        # We convert the representations to a numpy array
        for i in range(len(train_data)):
            train_data[i] = (train_data[i][0], np.array(
                train_data[i][1]), train_data[i][2])

        X_train = []
        y_train = []
        # We separate the id, representation and label in different lists
        for id, rep, label in train_data:
            X_train.append(rep)
            y_train.append(label)

        if task == "ionchannels_membraneproteins":
            # We convert labels to 0 and 1. 0 for ionchannels and 1 for membraneproteins
            y_train = [1 if label ==
                       settings.IONCHANNELS else 0 for label in y_train]
        elif task == "ionchannels_iontransporters":
            # We convert labels to 0 and 1. 0 for ionchannels and 1 for iontransporters
            y_train = [1 if label ==
                       settings.IONCHANNELS else 0 for label in y_train]
        elif task == "iontransporters_membraneproteins":
            # We convert labels to 0 and 1. 0 for iontransporters and 1 for membraneproteins
            y_train = [1 if label ==
                       settings.IONTRANSPORTERS else 0 for label in y_train]

        X_train = [np.mean(np.array(x), axis=0) for x in X_train]

        y_train = np.array(y_train)

    with h5py.File(settings.REPRESENTATIONS_FILTERED_PATH + representation_test, "r") as f:
        # We put the id, representation and label together in a list. The saved data is : (str(csv_id), data=representation), [str(csv_id)].attrs["label"] = label. And the representation is a numpy array
        test_data = [(id, representation, label) for id, representation in zip(
            f.keys(), f.values()) for label in f[id].attrs.values()]

        # We convert the representations to a numpy array
        for i in range(len(test_data)):
            test_data[i] = (test_data[i][0], np.array(
                test_data[i][1]), test_data[i][2])

        X_test = []
        y_test = []
        # We separate the id, representation and label in different lists
        for id, rep, label in test_data:
            X_test.append(rep)
            y_test.append(label)

        if task == "ionchannels_membraneproteins":
            # We convert labels to 0 and 1. 0 for ionchannels and 1 for membraneproteins
            y_test = [1 if label ==
                      settings.IONCHANNELS else 0 for label in y_test]
        elif task == "ionchannels_iontransporters":
            # We convert labels to 0 and 1. 0 for ionchannels and 1 for iontransporters
            y_test = [1 if label ==
                      settings.IONCHANNELS else 0 for label in y_test]
        elif task == "iontransporters_membraneproteins":
            # We convert labels to 0 and 1. 0 for iontransporters and 1 for membraneproteins
            y_test = [1 if label ==
                      settings.IONTRANSPORTERS else 0 for label in y_test]

        X_test = [np.mean(np.array(x), axis=0) for x in X_test]

        y_test = np.array(y_test)

    if classifier == "RF":
        rf_model = RandomForestClassifier(**rf_param_grid)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, rf_predictions)
        mcc = matthews_corrcoef(y_test, rf_predictions)
        sensitivity = recall_score(y_test, rf_predictions, pos_label=1)
        specificity = recall_score(y_test, rf_predictions, pos_label=0)
    elif classifier == "kNN":
        knn_model = KNeighborsClassifier(**knn_param_grid)
        knn_model.fit(X_train, y_train)
        knn_predictions = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, knn_predictions)
        mcc = matthews_corrcoef(y_test, knn_predictions)
        sensitivity = recall_score(y_test, knn_predictions, pos_label=1)
        specificity = recall_score(y_test, knn_predictions, pos_label=0)
    elif classifier == "SVM":
        svm_model = SVC(**svm_param_grid)
        svm_model.fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, svm_predictions)
        mcc = matthews_corrcoef(y_test, svm_predictions)
        sensitivity = recall_score(y_test, svm_predictions, pos_label=1)
        specificity = recall_score(y_test, svm_predictions, pos_label=0)
    elif classifier == "LR":
        lr_model = LogisticRegression(**lr_param_grid)
        lr_model.fit(X_train, y_train)
        lr_predictions = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, lr_predictions)
        mcc = matthews_corrcoef(y_test, lr_predictions)
        sensitivity = recall_score(y_test, lr_predictions, pos_label=1)
        specificity = recall_score(y_test, lr_predictions, pos_label=0)
    elif classifier == "FFNN":
        mlp_model = MLPClassifier(**mlp_param_grid)
        mlp_model.fit(X_train, y_train)
        mlp_predictions = mlp_model.predict(X_test)
        accuracy = accuracy_score(y_test, mlp_predictions)
        mcc = matthews_corrcoef(y_test, mlp_predictions)
        sensitivity = recall_score(y_test, mlp_predictions, pos_label=1)
        specificity = recall_score(y_test, mlp_predictions, pos_label=0)

    # We save the results in the results list, for each task, dataset and representation, representer, precision, classifier, mcc, accuracy, sensitivity, specificity
    results_list.append([task, dataset, representation_type, representer, precision, classifier, mcc, accuracy, sensitivity, specificity])

# We save the results in a csv file
results_df = pd.DataFrame(results_list, columns=["Task", "Dataset", "Representation",
                          "Representer", "Precision", "Classifier", "MCC", "Accuracy", "Sensitivity", "Specificity"])
results_df.to_csv(settings.RESULTS_PATH + "results_best_test_trad.csv", index=False)
