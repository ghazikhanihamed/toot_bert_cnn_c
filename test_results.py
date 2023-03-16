import torch
import h5py
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from settings import settings
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, matthews_corrcoef, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import fisher_exact
import random
import sklearn
import warnings
warnings.filterwarnings("ignore")


# We set the random seed for reproducibility
random.seed(settings.SEED)
np.random.seed(settings.SEED)
sklearn.utils.check_random_state(settings.SEED)

# we make a list of only h5 files that contains only train in the representations folder
representations = [representation for representation in os.listdir(
    settings.REPRESENTATIONS_FILTERED_PATH) if representation.endswith(".h5") and "train" in representation and "imbalanced" in representation
    and "ESM-1b" in representation and "ionchannels_membraneproteins" in representation and "finetuned" in representation]

# We take the best params of logistic regression from the results folder with the same conditions as the representations
results = [result for result in os.listdir(
    settings.RESULTS_PATH) if result.endswith(".csv") and "best_params" in result and "ESM-1b" in result and "ionchannels_membraneproteins" in result and "finetuned" in result and "imbalanced" in result]

# lr_params = pd.read_csv(os.path.join(
#     settings.RESULTS_PATH, results[0]), index_col=0)
lr_param_grid = {
    'penalty': 'l2',
    'C': 1,
    'solver': 'liblinear',
    'random_state': settings.SEED
}

# Now we train the logistic regression with the best params and then we test the logistic regression with the test data and then we save the results in the results folder as a csv file
training_representation = representations[0]

# We open the h5 file
with h5py.File(os.path.join(settings.REPRESENTATIONS_FILTERED_PATH, training_representation), "r") as f:
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

    y_train = [1 if label ==
               settings.IONCHANNELS else 0 for label in y_train]

    X_train = [np.mean(np.array(x), axis=0) for x in X_train]

    y_train = np.array(y_train)

    # We train the logistic regression with the best params
    lr = LogisticRegression(**lr_param_grid)

    lr.fit(X_train, y_train)


# we make a list of only h5 files that contains only test in the representations folder
test_representations = [representation for representation in os.listdir(
    settings.REPRESENTATIONS_FILTERED_PATH) if representation.endswith(".h5") and "test" in representation and "imbalanced" in representation
    and "ESM-1b" in representation and "ionchannels_membraneproteins" in representation and "finetuned" in representation]

# We open the h5 file
with h5py.File(os.path.join(settings.REPRESENTATIONS_FILTERED_PATH, test_representations[0]), "r") as f:
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

    y_test = [1 if label ==
               settings.IONCHANNELS else 0 for label in y_test]

    X_test = [np.mean(np.array(x), axis=0) for x in X_test]

    y_test = np.array(y_test)

    # We test the logistic regression with the test data
    y_pred = lr.predict(X_test)

    # We save the results in the results folder as a csv file
    df = pd.DataFrame({"id": [i for i in range(len(y_pred))],
                       "y_pred": y_pred, "y_true": y_test})
    df.to_csv(os.path.join(settings.RESULTS_PATH, "test_results.csv"))

    # We print MCC, accuracy and recall
    print("MCC: ", matthews_corrcoef(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))


# We take the train representations from representations folder and then filter the best params of logistic regression and then we train the logistic regression with the best params and then we test the logistic regression with the test data and then we save the results in the results folder as a csv file.
