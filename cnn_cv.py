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
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def custom_print(*args, **kwargs):
    message = " ".join([str(arg) for arg in args])
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
        optimizer.zero_grad()  # Clear gradients

        # Forward propagation
        output = network(data.to(device))

        # Compute loss (negative log likelihood: −log(y))
        loss = F.nll_loss(output, target.to(device))

        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights


def validate(network):
    """
    Tests the model on the validation set and computes the MCC.

    Parameters:
        - network (torch.nn.Module): The neural network model.

    Returns:
        - mcc (float): The Matthews Correlation Coefficient on the validation set.
    """

    network.eval()  # Set the module in evaluation mode (only affects certain modules)

    y_true = []
    y_pred = []

    with torch.no_grad():  # Disable gradient calculation
        for batch_i, (data, target) in enumerate(validation_loader):  # For each batch
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
n_epochs = 10  # Number of training epochs
batch_size_train = 1  # Batch size for training data
batch_size_test = 1  # Batch size for testing data
number_of_trials = 100  # Number of Optuna trials

# -------------------------------------------------------------------------

# Make runs repeatable
random_seed = settings.SEED
# Disable cuDNN use of nondeterministic algorithms
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
random.seed(settings.SEED)
np.random.seed(settings.SEED)
utils.check_random_state(settings.SEED)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=settings.SEED)

# we make a list of only h5 files that contains only train in the representations folder
representations = [
    representation
    for representation in os.listdir(settings.REPRESENTATIONS_FILTERED_PATH)
    if representation.endswith(".h5") and "train" in representation
]

print("Number of representations: ", len(representations))

# For each representation we take id, representation and label
for representation in representations:
    dataset_name = ""
    dataset_type = "na"
    # dataset_split = ""
    dataset_number = "na"
    representation_type = ""
    representer_model = ""
    precision_type = "half"

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
                if len(information) == 9:
                    if information[7] == "full":
                        precision_type = information[7]
                        representer_model = information[8][:-3]
                    else:
                        representer_model = information[7] + "_" + information[8][:-3]
                else:
                    representer_model = information[7][:-3]
            else:
                if len(information) == 8:
                    if information[6] == "full":
                        precision_type = information[6]
                        representer_model = information[7][:-3]
                    else:
                        representer_model = information[6] + "_" + information[7][:-3]
                else:
                    representer_model = information[6][:-3]

        else:
            # dataset_split = information[2] # train or test
            if len(information) == 7:
                if information[5] == "full":
                    precision_type = information[5]
                    representer_model = information[6][:-3]
                else:
                    representer_model = information[5] + "_" + information[6][:-3]
            else:
                representer_model = information[5][:-3]

    else:
        representation_type = "finetuned"
        if information[1] == "membraneproteins":
            dataset_type = information[2]  # Balanced or imbalanced
            # dataset_split = information[3] # train or test
            if dataset_type == "balanced":
                dataset_number = information[4]  # 1-10
                if len(information) == 12:
                    precision_type = information[7]
                    representer_model = information[8]
                else:
                    representer_model = information[7]
            else:
                if len(information) == 11:
                    precision_type = information[6]
                    representer_model = information[7]
                else:
                    representer_model = information[6]
        else:
            # dataset_split = information[2] # train or test
            if information[5] == "full":
                precision_type = information[5]
                representer_model = information[6]
            else:
                representer_model = information[5]

    # We check if the file exists in the results folder and if it does we skip it
    csv_file = (
        settings.RESULTS_PATH
        + "CNN_CV_results_"
        + dataset_name
        + "_"
        + dataset_type
        + "_"
        + dataset_number
        + "_"
        + representation_type
        + "_"
        + representer_model
        + "_"
        + precision_type
        + ".csv"
    )
    if os.path.exists(csv_file):
        print("Skipping ", csv_file)
        continue

    # Print the information
    print("-" * 50)
    print("-" * 50)
    print("Dataset name: ", dataset_name)
    (
        print("Dataset type: ", dataset_type)
        if information[1] == "membraneproteins"
        else print("Dataset type: ", "N/A")
    )
    # print("Dataset split: ", dataset_split)
    (
        print("Dataset number: ", dataset_number)
        if dataset_type == "balanced" and information[1] == "membraneproteins"
        else print("Dataset number: ", "N/A")
    )
    print("Representation type: ", representation_type)
    print("Representer model: ", representer_model)
    print("Precision type: ", precision_type)

    # We open the h5 file
    with h5py.File(settings.REPRESENTATIONS_FILTERED_PATH + representation, "r") as f:
        # We put the id, representation and label together in a list. The saved data is : (str(csv_id), data=representation), [str(csv_id)].attrs["label"] = label. And the representation is a numpy array
        train_data = [
            (id, representation, label)
            for id, representation in zip(f.keys(), f.values())
            for label in f[id].attrs.values()
        ]

        # We convert the representations to a numpy array
        for i in range(len(train_data)):
            train_data[i] = (
                train_data[i][0],
                np.array(train_data[i][1]),
                train_data[i][2],
            )

        X_train = []
        y_train = []
        # We separate the id, representation and label in different lists
        for id, rep, label in train_data:
            X_train.append(rep)
            y_train.append(label)

        if dataset_name == "ionchannels_membraneproteins":
            # We convert labels to 0 and 1. 0 for ionchannels and 1 for membraneproteins
            y_train = [1 if label == settings.IONCHANNELS else 0 for label in y_train]
        elif dataset_name == "ionchannels_iontransporters":
            # We convert labels to 0 and 1. 0 for ionchannels and 1 for iontransporters
            y_train = [1 if label == settings.IONCHANNELS else 0 for label in y_train]
        elif dataset_name == "iontransporters_membraneproteins":
            # We convert labels to 0 and 1. 0 for iontransporters and 1 for membraneproteins
            y_train = [
                1 if label == settings.IONTRANSPORTERS else 0 for label in y_train
            ]

        X_train = [
            torch.tensor(representation, dtype=torch.float)
            for representation in X_train
        ]
        Y_train = np.array(y_train)

        # We check the number of X_train and Y_train
        print("Number of X_train: ", len(X_train))
        print("Number of Y_train: ", len(Y_train))

        input_dim = X_train[0].shape[1]

        cnn_params = {
            "ionchannels_membraneproteins_imbalanced": {
                "dropout_prob": 0.26,
                "kernel_sizes": [5, 7],
                "lr": 0.0083,
                "optimizer": "RMSprop",
                "out_channels": [128, 64, 32],
            },
            "ionchannels_membraneproteins_balanced": {
                "dropout_prob": 0.32,
                "kernel_sizes": [7, 7, 7],
                "lr": 0.0028,
                "optimizer": "Adam",
                "out_channels": [128, 64, 32],
            },
            "iontransporters_membraneproteins_imbalanced": {
                "dropout_prob": 0.37,
                "kernel_sizes": [3, 7, 9],
                "lr": 0.00029,
                "optimizer": "Adam",
                "out_channels": [128, 64, 32],
            },
            "iontransporters_membraneproteins_balanced": {
                "dropout_prob": 0.26,
                "kernel_sizes": [5, 7],
                "lr": 0.00025,
                "optimizer": "Adam",
                "out_channels": [128, 64, 32],
            },
            "ionchannels_iontransporters": {
                "dropout_prob": 0.27,
                "kernel_sizes": [3, 7, 9],
                "lr": 0.00021,
                "optimizer": "RMSprop",
                "out_channels": [128, 64, 32],
            },
        }

        # We check if the file starts with one of the keys in cnn_params, then we use the corresponding parameters for the CNN for that dataset
        for key in cnn_params.keys():
            if representation.startswith(key):
                best_params = cnn_params[key]

        print("-" * 80)
        print(
            "\nApplying 5-fold cross validation to the best model on the whole dataset..."
        )
        print("-" * 80)

        result_folds_dict = {}

        # We apply 5-fold cross validation and compute the mean and std of sensitivity, specificity, accuracy and MCC. We also save each fold's results to a csv file.
        for fold, (train_ids, test_ids) in enumerate(skf.split(X_train, Y_train)):
            print(f"\nFold {fold+1}")
            x_train_fold = [X_train[i] for i in train_ids]
            y_train_fold = [Y_train[i] for i in train_ids]
            x_test_fold = [X_train[i] for i in test_ids]
            y_test_fold = [Y_train[i] for i in test_ids]

            train_dataset = GridDataset(x_train_fold, y_train_fold)
            test_dataset = GridDataset(x_test_fold, y_test_fold)

            # We create the dataloaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True
            )
            validation_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=settings.BATCH_SIZE, shuffle=True
            )

            # We create the CNN model with the best hyperparameters for each fold
            model = CNN(
                best_params["kernel_sizes"],
                best_params["out_channels"],
                best_params["dropout_prob"],
                input_dim,
            ).to(device)

            # We create the optimizer with the best hyperparameters for each fold
            optimizer_name = best_params["optimizer"]
            lr = best_params["lr"]
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

            # Training of the model
            for epoch in range(n_epochs):
                train(model, optimizer)  # Train the model
            # Evaluate the model on the validation set
            sensitivity, specificity, accuracy, mcc = validate(model)

            result_folds_dict[fold] = [sensitivity, specificity, accuracy, mcc]

        # We save the results to a csv file
        df = pd.DataFrame.from_dict(
            result_folds_dict,
            orient="index",
            columns=["Sensitivity", "Specificity", "Accuracy", "MCC"],
        )
        df.to_csv(
            settings.RESULTS_PATH
            + "CNN_CV_results_"
            + dataset_name
            + "_"
            + dataset_type
            + "_"
            + dataset_number
            + "_"
            + representation_type
            + "_"
            + representer_model
            + "_"
            + precision_type
            + ".csv",
            index=False,
        )  # Save to csv file
