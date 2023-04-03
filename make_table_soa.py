import pandas as pd
import os
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt
# import svm, ffnn and lr
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

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
            "±").str[0].astype(float).nlargest(3).index.tolist()
        df_three_best_mcc = df_temp.loc[three_best_mcc]

        # We take the best MCC value for each category of "Task", "Dataset" and "Representer"
        best_mcc_id = df_temp["MCC"].str.split(
            "±").str[0].astype(float).idxmax()

        df_best_mcc = df_temp.loc[best_mcc_id]
        ds_best_mcc.append(df_three_best_mcc)
        # ds_best_mcc.append(df_best_mcc)

df_table = pd.concat(ds_best_mcc)

lr = LogisticRegression(random_state=settings.SEED)
svm = SVC(random_state=settings.SEED)
ffnn = MLPClassifier(random_state=settings.SEED)

# We extract the best params for each classifier from results folder
# The params files are structured as follows: "gridsearch_best_params_" + dataset_name + "_" + dataset_type +"_" + dataset_number + "_" + representation_type + "_" + representer_model + "_" + precision_type + ".csv"

# We take the best params of the best MCC for each task in df_table
# We will use the best params to train the models and test them on the test set

# We create a new dataframe with the best params for each task
ds_best_params = {}
for row in df_table.itertuples():
    task = row.Task
    dataset = row.Dataset
    representer = row.Representer
    representation = row.Representation
    precision = row.Precision
    prec = "_" + precision if precision=="full" else ""
    params = pd.read_csv(os.path.join(settings.RESULTS_PATH, "gridsearch_best_params_" + task + "_" + dataset + "_" + "na" + "_" + representation + "_" + representer + prec + ".csv"))
    if task not in ds_best_params:
        ds_best_params[task] = [params]
    else:
        ds_best_params[task].append(params)


#  Define the parameter grids for each model
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'sigmoid']
}

lr_param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

mlp_param_grid = {
    'hidden_layer_sizes': [(512, 256, 64), (512,), (256,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd']
}

# For each task, we find the three train and test sets from REPRESENTATIONS_FILTERED_PATH with the information in df_table, then we train the models based on the best params and test them on the test set