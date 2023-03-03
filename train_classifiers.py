import h5py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from settings import settings
from classes.Classifier import CNN
import os

import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from skorch import NeuralNetClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef, accuracy_score, recall_score

from scipy.stats import fisher_exact

# We set the random seed for reproducibility
import random
import numpy as np
import torch
import pandas as pd
import sklearn
random.seed(settings.SEED)
np.random.seed(settings.SEED)
torch.manual_seed(settings.SEED)
torch.cuda.manual_seed(settings.SEED)
sklearn.utils.check_random_state(settings.SEED)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom scorer that converts data to CUDA tensors
def pytorch_scorer(estimator, X, y):
    # Convert model to CUDA tensor
    estimator.to(device)
    
    # Convert data to CUDA tensors
    X = X.to(device)
    y = y.to(device)
    
    # Make predictions
    y_pred = estimator.predict(X)
    
    # Calculate and return multiple scores as a dictionary
    scores = {
        "Sensitivity": make_scorer(recall_score, pos_label=1),
        "Specificity": make_scorer(recall_score, pos_label=0),
        "Accuracy": make_scorer(accuracy_score),
        "MCC": make_scorer(matthews_corrcoef)
    }
    return scores


# we make a list of only h5 files in the representations folder
representations = [f for f in os.listdir(settings.REPRESENTATIONS_PATH) if os.path.isfile(os.path.join(settings.REPRESENTATIONS_PATH, f)) and f.endswith(".h5") and "train" in f]

# For each representation we take id, representation and label
for representation in representations:
    # We separate the information from the name of the representation
    # We get the name of the dataset which is the two first words in the name of the representation separated by _
    dataset_name = representation.split("_")[0] + "_" + representation.split("_")[1]
    # We get the name of the type of dataset which is the third word in the name of the representation separated by _
    dataset_type = representation.split("_")[2]
    # We get the split of the dataset which is the fourth word in the name of the representation separated by _
    dataset_split = representation.split("_")[3]
    # We get the number of the dataset if the type is "balanced" which is the fifth word in the name of the representation separated by _
    if dataset_type == "balanced":
        dataset_number = representation.split("_")[4]
        # We get the type of the representations which is the sixth word in the name of the representation separated by _
        representation_type = representation.split("_")[5]
        # And we get the name of the model which is the eighth word in the name of the representation separated by _ + 9th word if exists without the .h5
        if len(representation.split("_")) == 9:
            model_name = representation.split("_")[7] + "_" + representation.split("_")[8][:-3]
        else:
            model_name = representation.split("_")[7][:-3]
    else:
        # We get the type of the representations which is the fifth word in the name of the representation separated by _
        representation_type = representation.split("_")[4]
        # And we get the name of the model which is the seventh word in the name of the representation separated by _
        if len(representation.split("_")) == 8:
            model_name = representation.split("_")[6] + "_" + representation.split("_")[7][:-3]
        else:
            model_name = representation.split("_")[6][:-3]
    # We open the h5 file
    with h5py.File(settings.REPRESENTATIONS_PATH + representation, "r") as f:
        # We put the id, representation and label together in a list
        train_data = list(zip(f["id"], f["representation"], f["label"]))

        X_train = []
        y_train = []
        # We separate the id, representation and label in different lists
        for id, representation, label in train_data:
            X_train.append(representation)
            y_train.append(label)
        
        # We convert labels to 0 and 1. 0 for membrane_proteins and 1 for ionchannels
        y_train = [0 if label == "membrane_proteins" else 1 for label in y_train]

        # Create the models
        svm_model = SVC(random_state=settings.SEED)
        rf_model = RandomForestClassifier(random_state=settings.SEED)
        knn_model = KNeighborsClassifier()
        mlp_model = MLPClassifier(random_state=settings.SEED)
        lr_model = LogisticRegression(random_state=settings.SEED)

        # Create the neural net classifier with scorch
        cnn = NeuralNetClassifier(
            CNN,
            max_epochs=10,
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            verbose=0
        )

        # Define the parameter grids for each model
        svm_param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'sigmoid']
        }

        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }

        mlp_param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01]
        }

        knn_param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute']
        }

        lr_param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        }

        # Define the parameter grid for the grid search
        cnn_param_grid = {
            'module__hidden_layers': [[100, 50], [200, 100], [300, 150]],
            'module__kernel_sizes': [[3, 5, 7, 9], [3, 5, 7]],
            'module__filters': [[32, 64], [64, 128], [128, 256]],
            'lr': [0.01, 0.001, 0.0001, 0.00001]
        }

        # Create a dictionary of models and their corresponding parameter grids
        models = {
            'svm': (svm_model, svm_param_grid),
            'rf': (rf_model, rf_param_grid),
            'mlp': (mlp_model, mlp_param_grid),
            'knn': (knn_model, knn_param_grid),
            'lr': (lr_model, lr_param_grid),
            'cnn': (cnn, cnn_param_grid)
        }

        # We make a dictionary of scores
        scores = {
            "Sensitivity": make_scorer(recall_score, pos_label=1),
            "Specificity": make_scorer(recall_score, pos_label=0),
            "Accuracy": make_scorer(accuracy_score),
            "MCC": make_scorer(matthews_corrcoef)
        }

        results_dict = {}
        best_models = {}
        best_params = {}
        # Perform the grid search for each model
        for name, (model, param_grid) in models.items():
            if name == "cnn":
                x_train = torch.tensor(X_train, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.long)
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring=make_scorer(pytorch_scorer), return_train_score=True, refit="MCC")
            else:
                x_train = X_train
                y_train = y_train
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scores, return_train_score=True, n_jobs=-1, refit="MCC")

            grid_search.fit(x_train, y_train)

            # We save the best parameters
            best_params[name] = grid_search.best_params_

            # We save the best model
            best_models[name] = grid_search.best_estimator_

            # We save the results of the grid search in a csv file
            results = pd.DataFrame(grid_search.cv_results_)
            results.to_csv(settings.RESULTS_PATH + "gridsearch_detail_results_" + dataset_type + "_" + dataset_split + "_" + dataset_number + "_" + representation_type + "_" + model_name + "_" + name + ".csv", index=False)

            # We save a table of results for each model as rows and the different metrics as columns. Each metric has two columns which are train (mean +- std) and test (mean +- std) scores
            results_dict[name] = {
                "Sensitivity": {"Train": str(round(grid_search.best_score_["train_Sensitivity"], 2)) + u"\u00B1" + str(round(grid_search.cv_results_["std_train_Sensitivity"][grid_search.best_index_], 2)), "Val": str(round(grid_search.best_score_["test_Sensitivity"], 2)) + u"\u00B1" + str(round(grid_search.cv_results_["std_test_Sensitivity"][grid_search.best_index_], 2))},
                "Specificity": {"Train": str(round(grid_search.best_score_["train_Specificity"], 2)) + u"\u00B1" + str(round(grid_search.cv_results_["std_train_Specificity"][grid_search.best_index_], 2)), "Val": str(round(grid_search.best_score_["test_Specificity"], 2)) + u"\u00B1" + str(round(grid_search.cv_results_["std_test_Specificity"][grid_search.best_index_], 2))},
                "Accuracy": {"Train": str(round(grid_search.best_score_["train_Accuracy"], 2)) + u"\u00B1" + str(round(grid_search.cv_results_["std_train_Accuracy"][grid_search.best_index_], 2)), "Val": str(round(grid_search.best_score_["test_Accuracy"], 2)) + u"\u00B1" + str(round(grid_search.cv_results_["std_test_Accuracy"][grid_search.best_index_], 2))},
                "MCC": {"Train": str(round(grid_search.best_score_["train_MCC"], 2)) + u"\u00B1" + str(round(grid_search.cv_results_["std_train_MCC"][grid_search.best_index_], 2)), "Val": str(round(grid_search.best_score_["test_MCC"], 2)) + u"\u00B1" + str(round(grid_search.cv_results_["std_test_MCC"][grid_search.best_index_], 2))},
            }
        
        # We save the best parameters for each model in a csv file
        best_params_df = pd.DataFrame(best_params)
        best_params_df.to_csv(settings.RESULTS_PATH + "gridsearch_best_params_" + dataset_type + "_" + dataset_split + "_" + dataset_number + "_" + representation_type + "_" + model_name + ".csv", index=False)

        # We apply Fisher's exact test to the best models and report the p-values in a matrix with rows and columns corresponding to the models
        train_data, val_data, train_labels, val_labels = train_test_split(X_train, y_train, test_size=0.2, random_state=settings.SEED, stratify=y_train)
        p_values = np.zeros((len(models), len(models)))
        for i, (name1, model1) in enumerate(best_models.items()):
            for j, (name2, model2) in enumerate(best_models.items()):
                if i == j:
                    p_values[i, j] = 1
                else:
                    if name1 == "cnn" or name2 == "cnn":
                        x_train = torch.tensor(train_data, dtype=torch.float32)
                        y_train = torch.tensor(train_labels, dtype=torch.long)
                        x_val = torch.tensor(val_data, dtype=torch.float32)
                        y_val = torch.tensor(val_labels, dtype=torch.long)
                    else:
                        x_train = train_data
                        y_train = train_labels
                        x_val = val_data
                        y_val = val_labels
                    model1.fit(x_train, y_train)
                    model2.fit(x_train, y_train)
                    y_pred1 = model1.predict(x_val)
                    y_pred2 = model2.predict(x_val)
                    p_values[i, j] = fisher_exact([y_pred1, y_pred2])[1]

        # We save the p-values in a csv file with rows and columns corresponding to the models
        p_values_df = pd.DataFrame(p_values, index=best_models.keys(), columns=best_models.keys())
        p_values_df.to_csv(settings.RESULTS_PATH + "gridsearch_pvalues_" + dataset_type + "_" + dataset_split + "_" + dataset_number + "_" + representation_type + "_" + model_name + ".csv")


        # We save the results in a csv file
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(settings.RESULTS_PATH + "gridsearch_results_" + dataset_type + "_" + dataset_split + "_" + dataset_number + "_" + representation_type + "_" + model_name + ".csv", index=False)