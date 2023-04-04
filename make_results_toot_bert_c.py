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

ds_best_mcc = []
for task in tasks:
    df_temp = df[df["Task"] == task]
    if not df_temp.empty:
        # We take the first three rows of the df_temp2 sorted by MCC value split by "±" and take the first element
        three_best_mcc = df_temp["MCC"].str.split(
            "±").str[0].astype(float).nlargest(20).index.tolist()
        df_three_best_mcc = df_temp.loc[three_best_mcc]

        # We take the best MCC value for each category of "Task", "Dataset" and "Representer"
        best_mcc_id = df_temp["MCC"].str.split(
            "±").str[0].astype(float).idxmax()

        df_best_mcc = df_temp.loc[best_mcc_id]
        ds_best_mcc.append(df_three_best_mcc)

df_table = pd.concat(ds_best_mcc)

# For each task, we find the three train and test sets from REPRESENTATIONS_FILTERED_PATH with the information in df_table, then we train the models based on the best params and test them on the test set
# we make a list of only h5 files that contains only train in the representations folder
results_list = []
accpeted_config = [[settings.IONCHANNELS_IONTRANSPORTERS, "imbalanced", "half", settings.FINETUNED, "ProtBERT-BFD"],
                   [settings.IONCHANNELS_MEMBRANEPROTEINS, "na", "half", settings.FINETUNED, "ProtBERT-BFD"]]
for task, dataset, precision, representation_type, representer in accpeted_config:
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


    lr_param_grid = {
        'random_state': 1
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

    # We train the Logistic Regression model
    lr_model = LogisticRegression(**lr_param_grid)
    lr_model.fit(X_train, y_train)

    # We predict the labels for the test set
    lr_predictions = lr_model.predict(X_test)

    # We compute the accuracy for the test set
    lr_accuracy = accuracy_score(y_test, lr_predictions)

    # We compute the mcc score for the test set
    lr_mcc = matthews_corrcoef(y_test, lr_predictions)

    # We compute the sensitivity for the test set
    lr_sensitivity = recall_score(y_test, lr_predictions, pos_label=1)

    # We compute the specificity for the test set
    lr_specificity = recall_score(y_test, lr_predictions, pos_label=0)

    # We save the results in the results list, for each task, dataset and representation, representer, precision, classifier, sensitivity, specificity accuracy, mcc
    results_list.append([task, dataset, representation_type, representer,
                        precision, "LR", lr_sensitivity, lr_specificity, lr_accuracy, lr_mcc])

# We save the results in a csv file
results_df = pd.DataFrame(results_list, columns=["Task", "Dataset", "Representation",
                          "Representer", "Precision", "Classifier", "Sensitivity", "Specificity", "Accuracy", "MCC"])
results_df.to_csv(settings.RESULTS_PATH + "results_toot_bert_c_test.csv", index=False)
