from classes.Classifier import CNN
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import os
from classes.PLMDataset import GridDataset
from settings import settings
import numpy as np
import random
from sklearn import utils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score
import h5py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def custom_print(*args, **kwargs):
    message = ' '.join([str(arg) for arg in args])
    logging.info(message)


print = custom_print


def train(network, optimizer):
    """
    Trains the model on the training data.

    Parameters:
        - network (torch.nn.Module): The neural network model.
        - optimizer (torch.optim.Optimizer): The optimizer for the model.
    """
    network.train()  # Set the module in training mode (only affects certain modules)
    for batch_i, (data, target) in enumerate(train_loader):  # For each batch
        optimizer.zero_grad()                                 # Clear gradients

        # Forward propagation
        output = network(data.to(device))

        # Compute loss (negative log likelihood: −log(y))
        loss = F.nll_loss(output, target.to(device))

        loss.backward()                                       # Compute gradients
        optimizer.step()                                      # Update weights


def validate(network):
    """
    Tests the model on the validation set and computes the MCC.

    Parameters:
        - network (torch.nn.Module): The neural network model.

    Returns:
        - mcc (float): The Matthews Correlation Coefficient on the validation set.
    """

    network.eval()         # Set the module in evaluation mode (only affects certain modules)

    y_true = []
    y_pred = []

    with torch.no_grad():  # Disable gradient calculation
        for batch_i, (data, target) in enumerate(test_loader):  # For each batch
            # Forward propagation
            output = network(data.to(device))

            # Find max value in each row, return indexes of max values
            pred = output.data.max(1, keepdim=True)[1]

            y_true.append(target)
            y_pred.append(pred)

    # Convert the lists of tensors to a single tensor and move it to the CPU
    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    # Compute the Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    accuracy = accuracy_score(y_true, y_pred)

    return sensitivity, specificity, accuracy, mcc


# -------------------------------------------------------------------------
# Optimization study for a PyTorch CNN with Optuna
# -------------------------------------------------------------------------

# Use cuda if available for faster computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Parameters ----------------------------------------------------------
n_epochs = 10                         # Number of training epochs
batch_size_train = 1                 # Batch size for training data
batch_size_test = 1                # Batch size for testing data
number_of_trials = 100                # Number of Optuna trials

# -------------------------------------------------------------------------

# Make runs repeatable
random_seed = settings.SEED
# Disable cuDNN use of nondeterministic algorithms
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
random.seed(settings.SEED)
np.random.seed(settings.SEED)
utils.check_random_state(settings.SEED)

# We read the csv file of the full results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH,
                 "mean_balanced_imbalanced_results_trad_cnn.csv"))

# The dataframe has the following columns: "Task", "Dataset", "Representation", "Representer", "Precision", "Classifier",
# We will find the best MCC for each task

tasks = settings.TASKS

ds_best_mcc = []
for task in tasks:
    # We take the the rows that have the task and the classifier CNN
    df_temp = df[(df["Task"] == task) & (df["Classifier"] == "CNN")]
    if not df_temp.empty:
        # We take the first three rows of the df_temp2 sorted by MCC value split by "±" and take the first element
        three_best_mcc = df_temp["MCC"].str.split(
            "±").str[0].astype(float).nlargest(1).index.tolist()
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
config = []
for row in df_table.itertuples():
    task = row.Task
    dataset = row.Dataset
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

    # We open the h5 file
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

        X_train = [torch.tensor(representation, dtype=torch.float)
                   for representation in X_train]
        Y_train = np.array(y_train)

        # We check the number of X_train and Y_train
        print("Number of X_train: ", len(X_train))
        print("Number of Y_train: ", len(Y_train))

        input_dim = X_train[0].shape[1]

        cnn_params = {"ionchannels_membraneproteins_imbalanced": {
            "dropout_prob": 0.26,
            "kernel_sizes": [5, 7],
            "lr": 0.0083,
            "optimizer": "RMSprop",
            "out_channels": [128, 64, 32]
        },
            "ionchannels_membraneproteins_balanced": {
            "dropout_prob": 0.32,
            "kernel_sizes": [7, 7, 7],
            "lr": 0.0028,
            "optimizer": "Adam",
            "out_channels": [128, 64, 32]
        },
            "iontransporters_membraneproteins_imbalanced": {
            "dropout_prob": 0.37,
            "kernel_sizes": [3, 7, 9],
            "lr": 0.00029,
            "optimizer": "Adam",
            "out_channels": [128, 64, 32]
        },
            "iontransporters_membraneproteins_balanced": {
            "dropout_prob": 0.26,
            "kernel_sizes": [5, 7],
            "lr": 0.00025,
            "optimizer": "Adam",
            "out_channels": [128, 64, 32]
        },
            "ionchannels_iontransporters": {
            "dropout_prob": 0.27,
            "kernel_sizes": [3, 7, 9],
            "lr": 0.00021,
            "optimizer": "RMSprop",
            "out_channels": [128, 64, 32]
        }
        }

        # We check if the file starts with one of the keys in cnn_params, then we use the corresponding parameters for the CNN for that dataset
        for key in cnn_params.keys():
            if representation_train.startswith(key):
                best_params = cnn_params[key]

        print("-"*80)
        print("-"*80)

        result_folds_dict = {}

        train_dataset = GridDataset(X_train, Y_train)

        # We create the dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)

        # We create the CNN model with the best hyperparameters for each fold
        model = CNN(best_params['kernel_sizes'], best_params['out_channels'],
                    best_params['dropout_prob'], input_dim).to(device)

        # We create the optimizer with the best hyperparameters for each fold
        optimizer_name = best_params['optimizer']
        lr = best_params['lr']
        optimizer = getattr(optim, optimizer_name)(
            model.parameters(), lr=lr)

        # Training of the model
        for epoch in range(n_epochs):
            train(model, optimizer)  # Train the model

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

        X_test = [torch.tensor(representation, dtype=torch.float)
                  for representation in X_test]
        Y_test = np.array(y_test)

        # We check the number of X_test and y_test
        print("Number of X_test: ", len(X_test))
        print("Number of y_test: ", len(Y_test))

        input_dim = X_test[0].shape[1]

        # We create the test dataset
        test_dataset = GridDataset(X_test, Y_test)

        # We create the dataloaders
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)

        # We test the model
        sensitivity, specificity, accuracy, mcc = validate(model)

    # We save the results in the results list, for each task, dataset and representation, representer, precision, classifier, sensitivity, specificity accuracy, mcc
    results_list.append([task, dataset, representation_type, representer, precision,
                        "CNN", sensitivity, specificity, accuracy, mcc])

# We save the results in a csv file
results_df = pd.DataFrame(results_list, columns=["Task", "Dataset", "Representation",
                          "Representer", "Precision", "Classifier", "Sensitivity", "Specificity", "Accuracy", "MCC"])
results_df.to_csv(settings.RESULTS_PATH +
                  "results_best_test_cnn.csv", index=False)
