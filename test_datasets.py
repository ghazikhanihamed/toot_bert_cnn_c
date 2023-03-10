import sklearn
import random
from scipy.stats import fisher_exact
from sklearn.metrics import make_scorer, matthews_corrcoef, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
from settings import settings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import h5py


# We set the random seed for reproducibility
random.seed(settings.SEED)
np.random.seed(settings.SEED)
sklearn.utils.check_random_state(settings.SEED)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                      random_state=settings.SEED)

# we make a list of only h5 files that contains only train in the representations folder
representations = [representation for representation in os.listdir(
    settings.REPRESENTATIONS_FILTERED_PATH) if representation.endswith(".h5") and "train" in representation]

print(representations)

# For each representation we take id, representation and label
for representation in representations:
    dataset_name = ""
    dataset_type = "na"
    # dataset_split = ""
    dataset_number = "na"
    representation_type = ""
    representer_model = ""

    information = representation.split("_")
    # We separate the information from the name of the representation
    # We get the name of the dataset which is the two first words in the name of the representation separated by _
    # ionchannels_membraneproteins or ionchannels_iontransporters or iontrasnporters_membraneproteins
    dataset_name = information[0] + "_" + information[1]
    # If frozen is in the name of the representation, then the dataset is frozen
    if "frozen" in representation:
        representation_type = "frozen"
        if information[1] == "membraneproteins":
            dataset_type = information[2]  # Balanced or imbalanced
            # dataset_split = information[3] # train or test
            if dataset_type == "balanced":
                dataset_number = information[4]  # 1-10
                if len(information) == 8:
                    representer_model = information[7][:-3]
                else:
                    representer_model = information[7] + \
                        "_" + information[8][:-3]
            else:
                if len(information) == 7:
                    representer_model = information[6][:-3]
                else:
                    representer_model = information[6] + \
                        "_" + information[7][:-3]
        else:
            # dataset_split = information[2] # train or test
            if len(information) == 6:
                representer_model = information[5][:-3]
            else:
                representer_model = information[5] + "_" + information[6][:-3]
    else:
        representation_type = "finetuned"
        if information[1] == "membraneproteins":
            dataset_type = information[2]  # Balanced or imbalanced
            # dataset_split = information[3] # train or test
            if dataset_type == "balanced":
                dataset_number = information[4]  # 1-10
                representer_model = information[7][:-3]
            else:
                representer_model = information[6][:-3]
        else:
            # dataset_split = information[2]
            representer_model = information[5][:-3]

    # Print the information
    print("-"*50)
    print("-"*50)
    print("Dataset name: ", dataset_name)
    print("Dataset type: ", dataset_type) if information[1] == "membraneproteins" else print(
        "Dataset type: ", "N/A")
    # print("Dataset split: ", dataset_split)
    print("Dataset number: ", dataset_number) if dataset_type == "balanced" and information[1] == "membraneproteins" else print(
        "Dataset number: ", "N/A")
    print("Representation type: ", representation_type)
    print("Representer model: ", representer_model)

    # We open the h5 file
    with h5py.File(settings.REPRESENTATIONS_FILTERED_PATH + representation, "r") as f:
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
        for id, representation, label in train_data:
            X_train.append(representation)
            y_train.append(label)

        # We convert labels to 0 and 1. 0 for membrane_proteins and 1 for ionchannels
        y_train = [0 if label == settings.MEMBRANE_PROTEINS or label ==
                   settings.IONTRANSPORTERS else 1 for label in y_train]

        X_train = [np.array(x) for x in X_train]
        y_train = np.array(y_train)

        # We print the information about the dataset
        print("Number of samples: ", len(X_train))
        print("Number of labels: ", len(y_train))
        print("Number of different labels: ", len(set(y_train)))

        x_train, x_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=settings.SEED, stratify=y_train)
        
        lr = LogisticRegression(random_state=settings.SEED)
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)
        print("Accuracy: ", accuracy_score(y_test, y_pred))

        print("-"*50)