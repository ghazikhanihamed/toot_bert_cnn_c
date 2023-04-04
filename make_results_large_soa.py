import pandas as pd
import os
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt
# import svm, ffnn and lr
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
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

# For each task, we find the three train and test sets from REPRESENTATIONS_FILTERED_PATH with the information in df_table, then we train the models based on the best params and test them on the test set
# we make a list of only h5 files that contains only train in the representations folder
results_list = []
config = []
for task in tasks:
    dataset = "imbalanced" if task == settings.IONCHANNELS_MEMBRANEPROTEINS or settings.IONTRANSPORTERS_MEMBRANEPROTEINS else "na"
    representer = row.Representer
    representation_type = row.Representation
    precision = row.Precision
    prec = "_" + precision if precision == "full" else ""

    # We check if the config exists in the list
    if [task, dataset, representation_type, representer, precision] in config:
        continue
    else:
        config.append(
            [task, dataset, representation_type, representer, precision])

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

    # We train the SVM model
    svm_model = SVC(**svm_param_grid)
    svm_model.fit(X_train, y_train)

    # We train the Logistic Regression model
    lr_model = LogisticRegression(**lr_param_grid)
    lr_model.fit(X_train, y_train)

    # We train the MLP model
    mlp_model = MLPClassifier(**mlp_param_grid)
    mlp_model.fit(X_train, y_train)

    # We predict the labels for the test set
    svm_predictions = svm_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)
    mlp_predictions = mlp_model.predict(X_test)

    # We compute the accuracy for the test set
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    mlp_accuracy = accuracy_score(y_test, mlp_predictions)

    # We compute the mcc score for the test set
    svm_mcc = matthews_corrcoef(y_test, svm_predictions)
    lr_mcc = matthews_corrcoef(y_test, lr_predictions)
    mlp_mcc = matthews_corrcoef(y_test, mlp_predictions)

    # We compute the sensitivity for the test set
    svm_sensitivity = recall_score(y_test, svm_predictions, pos_label=1)
    lr_sensitivity = recall_score(y_test, lr_predictions, pos_label=1)
    mlp_sensitivity = recall_score(y_test, mlp_predictions, pos_label=1)

    # We compute the specificity for the test set
    svm_specificity = recall_score(y_test, svm_predictions, pos_label=0)
    lr_specificity = recall_score(y_test, lr_predictions, pos_label=0)
    mlp_specificity = recall_score(y_test, mlp_predictions, pos_label=0)

    # We save the results in the results list, for each task, dataset and representation, representer, precision, classifier, sensitivity, specificity accuracy, mcc
    results_list.append([task, dataset, representation_type, representer, precision,
                        "SVM", svm_sensitivity, svm_specificity, svm_accuracy, svm_mcc])
    results_list.append([task, dataset, representation_type, representer,
                        precision, "LR", lr_sensitivity, lr_specificity, lr_accuracy, lr_mcc])
    results_list.append([task, dataset, representation_type, representer, precision,
                        "FFNN", mlp_sensitivity, mlp_specificity, mlp_accuracy, mlp_mcc])

# We save the results in a csv file
results_df = pd.DataFrame(results_list, columns=["Task", "Dataset", "Representation",
                          "Representer", "Precision", "Classifier", "Sensitivity", "Specificity", "Accuracy", "MCC"])
results_df.to_csv(settings.RESULTS_PATH + "results_best_test.csv", index=False)
