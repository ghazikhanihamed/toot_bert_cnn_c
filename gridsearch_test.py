import torch
from skorch import NeuralNetClassifier
from classes.PLMDataset import SliceDatasetX
from classes.PLMDataset import GridDataset
import h5py
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from settings import settings
from sklearn.svm import SVC
import torch.optim as optim
import torch.nn as nn
from classes.Classifier import CNN
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
torch.manual_seed(settings.SEED)
torch.cuda.manual_seed(settings.SEED)
sklearn.utils.check_random_state(settings.SEED)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                      random_state=settings.SEED)

# we make a list of only h5 files that contains only train in the representations folder
representations = [representation for representation in os.listdir(
    settings.REPRESENTATIONS_FILTERED_PATH) if representation.endswith(".h5") and "train" in representation]

print("Number of representations: ", len(representations))

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
                representer_model = information[7]
            else:
                representer_model = information[6]
        else:
            # dataset_split = information[2]
            representer_model = information[5]

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
        for id, rep, label in train_data:
            X_train.append(rep)
            y_train.append(label)

        if dataset_name == "ionchannels_membraneproteins":
            # We convert labels to 0 and 1. 0 for ionchannels and 1 for membraneproteins
            y_train = [1 if label ==
                       settings.IONCHANNELS else 0 for label in y_train]
        elif dataset_name == "ionchannels_iontransporters":
            # We convert labels to 0 and 1. 0 for ionchannels and 1 for iontransporters
            y_train = [1 if label ==
                       settings.IONCHANNELS else 0 for label in y_train]
        elif dataset_name == "iontransporters_membraneproteins":
            # We convert labels to 0 and 1. 0 for iontransporters and 1 for membraneproteins
            y_train = [1 if label ==
                       settings.IONTRANSPORTERS else 0 for label in y_train]

        X_train = [np.array(x) for x in X_train]

        y_train = np.array(y_train)

        # Create the models
        svm_model = SVC(random_state=settings.SEED)
        rf_model = RandomForestClassifier(random_state=settings.SEED)
        knn_model = KNeighborsClassifier()
        lr_model = LogisticRegression(random_state=settings.SEED)
        mlp_model = MLPClassifier(random_state=settings.SEED)

        # We take the dimension of the representation
        input_dim = X_train[0].shape[1]

        # Create the neural net classifier with scorch
        cnn = NeuralNetClassifier(
            module=CNN,
            max_epochs=20,
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            verbose=0,
            batch_size=1,
            device=device,
            module__input_size=input_dim,
            train_split=None
        )

        #  Define the parameter grids for each model
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

        mlp_param_grid = {
            'hidden_layer_sizes': [(512, 256, 64), (512,), (256,)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd']
        }

        cnn_param_grid = {
            'module__kernel_sizes': [[3, 5, 7], [3, 5], [3, 7], [5, 7]],
            'module__out_channels': [[64, 32], [64, 32, 16], [64, 32, 16, 8]],
            'lr': [0.001, 0.0001, 0.00001, 0.000001]
        }

        models = {
            'cnn': (cnn, cnn_param_grid),
            'svm': (svm_model, svm_param_grid),
            'rf': (rf_model, rf_param_grid),
            'knn': (knn_model, knn_param_grid),
            'lr': (lr_model, lr_param_grid),
            'mlp': (mlp_model, mlp_param_grid)
        }

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
            print("Performing grid search for model: ", name)

            # We perform the grid search
            grid_search = GridSearchCV(model, param_grid, cv=skf, scoring=scores,
                                       return_train_score=True, n_jobs=5, refit="MCC", error_score='raise')

            if name != "cnn":
                # Take the mean of each representation
                X_train = [np.mean(x, axis=0) for x in X_train]
                grid_search.fit(X_train, y_train)

                # We save the best parameters
                best_params[name] = grid_search.best_params_

                # We save the best model
                best_models[name] = grid_search.best_estimator_

                # We save the results of the grid search in a csv file
                results = pd.DataFrame(grid_search.cv_results_)
                results.to_csv(settings.RESULTS_PATH + "gridsearch_detail_results_" + name + "_" + dataset_name + "_" +
                               dataset_type + "_" + dataset_number + "_" + representation_type + "_" + representer_model + ".csv", index=False)
                
                # We save a table of results for each model as rows and the different metrics as columns. Each metric has two columns which are train (mean +- std) and test (mean +- std) scores
                results_dict[name] = {
                    "Sensitivity": {"Train": '{:.2f}'.format(round(grid_search.cv_results_["mean_train_Sensitivity"][grid_search.best_index_], 2) * 100) + u"\u00B1" + '{:.2f}'.format(round(grid_search.cv_results_["std_train_Sensitivity"][grid_search.best_index_], 2) * 100), "Val": '{:.2f}'.format(round(grid_search.cv_results_["mean_test_Sensitivity"][grid_search.best_index_], 2) * 100) + u"\u00B1" + '{:.2f}'.format(round(grid_search.cv_results_["std_test_Sensitivity"][grid_search.best_index_], 2) * 100)},
                    "Specificity": {"Train": '{:.2f}'.format(round(grid_search.cv_results_["mean_train_Specificity"][grid_search.best_index_], 2) * 100) + u"\u00B1" + '{:.2f}'.format(round(grid_search.cv_results_["std_train_Specificity"][grid_search.best_index_], 2) * 100), "Val": '{:.2f}'.format(round(grid_search.cv_results_["mean_test_Specificity"][grid_search.best_index_], 2) * 100) + u"\u00B1" + '{:.2f}'.format(round(grid_search.cv_results_["std_test_Specificity"][grid_search.best_index_], 2) * 100)},
                    "Accuracy": {"Train": '{:.2f}'.format(round(grid_search.cv_results_["mean_train_Accuracy"][grid_search.best_index_], 2) * 100) + u"\u00B1" + '{:.2f}'.format(round(grid_search.cv_results_["std_train_Accuracy"][grid_search.best_index_], 2) * 100), "Val": '{:.2f}'.format(round(grid_search.cv_results_["mean_test_Accuracy"][grid_search.best_index_], 2) * 100) + u"\u00B1" + '{:.2f}'.format(round(grid_search.cv_results_["std_test_Accuracy"][grid_search.best_index_], 2) * 100)},
                    "MCC": {"Train": '{:.2f}'.format(round(grid_search.cv_results_["mean_train_MCC"][grid_search.best_index_], 2)) + u"\u00B1" + '{:.2f}'.format(round(grid_search.cv_results_["std_train_MCC"][grid_search.best_index_], 2)), "Val": '{:.2f}'.format(round(grid_search.cv_results_["mean_test_MCC"][grid_search.best_index_], 2)) + u"\u00B1" + '{:.2f}'.format(round(grid_search.cv_results_["std_test_MCC"][grid_search.best_index_], 2))}
                }

            else:
                # We make grid search for the CNN without using scikit-learn grid search, but we make the same output as scikit-learn grid search
                cnn_results = []
                train_accuracy_list = []
                test_accuracy_list = []
                train_sensitivity_list = []
                test_sensitivity_list = []
                train_specificity_list = []
                test_specificity_list = []
                train_mcc_list = []
                test_mcc_list = []
                best_cnn_params = {}
                best_cnn_mcc = -1


                for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):

                    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
                    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

                    for kernel_sizes in param_grid["module__kernel_sizes"]:
                        for out_channels in param_grid["module__out_channels"]:
                            for lr in param_grid["lr"]:
                                # Create the neural net classifier with scorch
                                cnn = NeuralNetClassifier(
                                    module=CNN,
                                    max_epochs=20,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=optim.Adam,
                                    verbose=0,
                                    batch_size=1,
                                    device=device,
                                    module__input_size=input_dim,
                                    train_split=None,
                                    module__kernel_sizes=kernel_sizes,
                                    module__out_channels=out_channels,
                                    lr=lr
                                )
                                cnn.fit(X_train_fold, y_train_fold)
                                train_preds = cnn.predict(X_train_fold)
                                test_preds = cnn.predict(X_test_fold)

                                train_accuracy_list.append(accuracy_score(y_train_fold, train_preds))
                                test_accuracy_list.append(accuracy_score(y_test_fold, test_preds))
                                train_sensitivity_list.append(recall_score(y_train_fold, train_preds, pos_label=1))
                                test_sensitivity_list.append(recall_score(y_test_fold, test_preds, pos_label=1))
                                train_specificity_list.append(recall_score(y_train_fold, train_preds, pos_label=0))
                                test_specificity_list.append(recall_score(y_test_fold, test_preds, pos_label=0))
                                train_mcc_list.append(matthews_corrcoef(y_train_fold, train_preds))
                                test_mcc_list.append(matthews_corrcoef(y_test_fold, test_preds))

                                cnn_results.append({'fold': fold,
                                                    'parameters': {'module__kernel_sizes': kernel_sizes, 'module__out_channels': out_channels, 'lr': lr},
                                                    'train_Accuracy': accuracy_score(y_train_fold, train_preds),
                                                    'test_Accuracy': accuracy_score(y_test_fold, test_preds),
                                                    'train_Sensitivity': recall_score(y_train_fold, train_preds, pos_label=1),
                                                    'test_Sensitivity': recall_score(y_test_fold, test_preds, pos_label=1),
                                                    'train_Specificity': recall_score(y_train_fold, train_preds, pos_label=0),
                                                    'test_Specificity': recall_score(y_test_fold, test_preds, pos_label=0),
                                                    'train_MCC': matthews_corrcoef(y_train_fold, train_preds),
                                                    'test_MCC': matthews_corrcoef(y_test_fold, test_preds)})
                                
                                if matthews_corrcoef(y_test_fold, test_preds) > best_cnn_mcc:
                                    best_cnn_mcc = matthews_corrcoef(y_test_fold, test_preds)
                                    best_cnn_params = {'module__kernel_sizes': kernel_sizes, 'module__out_channels': out_channels, 'lr': lr}
                                



        # We save the best parameters for each model in a csv file
        best_params_df = pd.DataFrame(best_params)
        best_params_df.to_csv(settings.RESULTS_PATH + "gridsearch_best_params_" + dataset_name + "_" + dataset_type +
                              "_" + dataset_number + "_" + representation_type + "_" + representer_model + ".csv", index=False)

        # We apply Fisher's exact test to the best models and report the p-values in a matrix with rows and columns corresponding to the models
        train_data, val_data, train_labels, val_labels = train_test_split(
            X_train, y_train, test_size=0.2, random_state=settings.SEED, stratify=y_train)
        p_values = np.zeros((len(models), len(models)))
        for i, (name1, model1) in enumerate(best_models.items()):
            for j, (name2, model2) in enumerate(best_models.items()):
                if i == j:
                    p_values[i, j] = 1
                else:
                    model1.fit(train_data, train_labels)
                    y_pred1 = model1.predict(val_data)

                    model2.fit(train_data, train_labels)
                    y_pred2 = model2.predict(val_data)

                    table = [[np.sum((y_pred1 == 0) & (y_pred2 == 0)), np.sum((y_pred1 == 0) & (y_pred2 == 1))],
                             [np.sum((y_pred1 == 1) & (y_pred2 == 0)), np.sum((y_pred1 == 1) & (y_pred2 == 1))]]

                    p_values[i, j] = fisher_exact(table)[1]

        # We save the p-values in a csv file with rows and columns corresponding to the models
        p_values_df = pd.DataFrame(
            p_values, index=best_models.keys(), columns=best_models.keys())
        p_values_df.to_csv(settings.RESULTS_PATH + "gridsearch_pvalues_" + dataset_name + "_" + dataset_type +
                           "_" + dataset_number + "_" + representation_type + "_" + representer_model + ".csv")

        # We save the results of the grid search in a csv file
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(settings.RESULTS_PATH + "gridsearch_results_" + dataset_name + "_" + dataset_type +
                          "_" + dataset_number + "_" + representation_type + "_" + representer_model + ".csv")
