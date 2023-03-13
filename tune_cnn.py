from classes.Classifier import CNN
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
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
from sklearn.metrics import matthews_corrcoef
import h5py
from sklearn.model_selection import train_test_split


def train(network, optimizer):
    """Trains the model.
    Parameters:
        - network (__main__.Net):              The CNN
        - optimizer (torch.optim.<optimizer>): The optimizer for the CNN
    """
    network.train()  # Set the module in training mode (only affects certain modules)
    for batch_i, (data, target) in enumerate(train_loader):  # For each batch

        # Limit training data for faster computation
        if batch_i * batch_size_train > number_of_train_examples:
            break

        optimizer.zero_grad()                                 # Clear gradients
        # Forward propagation
        output = network(data.to(device))
        # Compute loss (negative log likelihood: −log(y))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()                                       # Compute gradients
        optimizer.step()                                      # Update weights


def test(network):
    """Tests the model.
    Parameters:
        - network (__main__.Net): The CNN
    Returns:
        - accuracy_test (torch.Tensor): The test accuracy
    """
    network.eval()         # Set the module in evaluation mode (only affects certain modules)
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():  # Disable gradient calculation (when you are sure that you will not call Tensor.backward())
        for batch_i, (data, target) in enumerate(test_loader):  # For each batch

            # Limit testing data for faster computation
            if batch_i * batch_size_test > number_of_test_examples:
                break

            # Forward propagation
            output = network(data.to(device))
            # Find max value in each row, return indexes of max values
            pred = output.data.max(1, keepdim=True)[1]

            y_true.append(target)
            y_pred.append(pred)
            # Compute correct predictions
            # correct += pred.eq(target.to(device).data.view_as(pred)).sum()

    # accuracy_test = correct / len(test_loader.dataset)
    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    mcc = matthews_corrcoef(y_true, y_pred)

    return mcc


def objective(trial):
    """Objective function to be optimized by Optuna.
    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.
    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """

    # Define range of values to be tested for the hyperparameters
    kernel_size = trial.suggest_categorical(
        "kernel_sizes", [[3, 5, 7], [3, 5], [5, 7]])
    out_channel = trial.suggest_categorical(
        "out_channels", [[64, 32], [64, 32, 16], [64, 32, 16, 8]])
    dropout = trial.suggest_float(
        "dropout_prob", 0.2, 0.5)         # Dropout for FC1 layer

    # Generate the model
    model = CNN(trial, kernel_size, out_channel, dropout).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    # Learning rates
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model
    for epoch in range(n_epochs):
        train(model, optimizer)  # Train the model
        mcc = test(model)   # Evaluate the model

        # For pruning (stops trial early if not promising)
        trial.report(mcc, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mcc

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
# Limit number of observations for faster computation
limit_obs = True

# *** Note: For more accurate results, do not limit the observations.
#           If not limited, however, it might take a very long time to run.
#           Another option is to limit the number of epochs. ***

if limit_obs:  # Limit number of observations
    number_of_train_examples = 500 * batch_size_train  # Max train observations
    number_of_test_examples = 5 * batch_size_test      # Max test observations
else:
    number_of_train_examples = 60000                   # Max train observations
    number_of_test_examples = 10000                    # Max test observations
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

        x_train, x_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=settings.SEED, stratify=y_train)

        train_dataset = GridDataset(x_train, y_train)
        test_dataset = GridDataset(x_test, y_test)

        # We create the dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)

        # Create an Optuna study to maximize test MCC
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=number_of_trials)

        # -------------------------------------------------------------------------
        # Results
        # -------------------------------------------------------------------------

        # Find number of pruned and completed trials
        pruned_trials = study.get_trials(
            deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE])

        # Display the study statistics
        print("\nStudy statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        trial = study.best_trial
        print("Best trial:")
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # Save results to csv file
        df = study.trials_dataframe().drop(
            ['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
        # Keep only results that did not prune
        df = df.loc[df['state'] == 'COMPLETE']
        df = df.drop('state', axis=1)                 # Exclude state column
        df = df.sort_values('value')                  # Sort based on accuracy
        df.to_csv('optuna_results.csv', index=False)  # Save to csv file

        # Display results in a dataframe
        print("\nOverall Results (ordered by accuracy):\n {}".format(df))

        # Find the most important hyperparameters
        most_important_parameters = optuna.importance.get_param_importances(
            study, target=None)

        # Display the most important hyperparameters
        print('\nMost important hyperparameters:')
        for key, value in most_important_parameters.items():
            print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
