from classes.Classifier import CNN
import optuna
from optuna.trial import TrialState
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
from sklearn.model_selection import train_test_split
import pandas as pd


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


def objective(trial):
    """
    Objective function to be optimized by Optuna.

    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.

    Parameters:
        - trial (optuna.trial._trial.Trial): Optuna trial

    Returns:
        - mcc (float): The validation MCC. Parameter to be maximized.
    """
    # Define range of values to be tested for the hyperparameters
    kernel_size = trial.suggest_categorical(
        "kernel_sizes", [[3, 5, 7], [3, 5], [5, 7], [7]])
    out_channel = trial.suggest_categorical(
        "out_channels", [[64, 32], [128, 64, 32]])
    dropout = trial.suggest_float("dropout_prob", 0.2, 0.6)

    # Generate the model
    model = CNN(kernel_size, out_channel, dropout, input_dim).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model
    try:
        for epoch in range(n_epochs):
            train(model, optimizer)  # Train the model
            # Evaluate the model on the validation set
            _, _, _, mcc = validate(model)

            # For pruning (stops trial early if not promising)
            trial.report(mcc, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    except RuntimeError as e:
        if str(e) == "Encountered zero total variance in all trees.":
            # Return a very low value to make sure this trial is not considered as the best.
            return float('-inf')
        else:
            # Re-raise the exception if it is not the one we want to handle.
            raise e

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
            if len(information) == 6:
                representer_model = information[5][:-3]
            else:
                representer_model = information[5] + "_" + information[6][:-3]
    else:
        representation_type = "finetuned"
        if information[1] == "membraneproteins":
            dataset_type = information[2]  # Balanced or imbalanced
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

        X_train = [torch.tensor(representation, dtype=torch.float).to(
            device) for representation in X_train]
        Y_train = np.array(y_train)

        # We check the number of X_train and Y_train
        print("Number of X_train: ", len(X_train))
        print("Number of Y_train: ", len(Y_train))

        # We select randomly stratified half of the samples for fast training
        X_train_half, _, y_train_half, _ = train_test_split(
            X_train, Y_train, test_size=0.5, random_state=settings.SEED, stratify=y_train)
        
        # We check the number of X_train and Y_train
        print("Number of X_train_half: ", len(X_train_half))
        print("Number of Y_train_half: ", len(y_train_half))

        input_dim = X_train_half[0].shape[1]

        x_train, x_test, y_train, y_test = train_test_split(
            X_train_half, y_train_half, test_size=0.2, random_state=settings.SEED, stratify=y_train_half)

        train_dataset = GridDataset(x_train, y_train)
        test_dataset = GridDataset(x_test, y_test)

        # We create the dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)

        # Create an Optuna study to maximize test MCC
        study = optuna.create_study(direction="maximize")
        # Run the optimization process using multiple GPUs
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
        df.to_csv(settings.RESULTS_PATH + "optuna_results_" + dataset_name + "_" + dataset_type +
                  "_" + dataset_number + "_" + representation_type + "_" + representer_model + ".csv", index=False)  # Save to csv file

        # Display results in a dataframe
        print("\nOverall Results (ordered by MCC):\n {}".format(df))

        # Find the most important hyperparameters
        most_important_parameters = optuna.importance.get_param_importances(
            study, target=None)

        # Display the most important hyperparameters
        print('\nMost important hyperparameters:')
        for key, value in most_important_parameters.items():
            print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))

        best_params = study.best_params

        print("-"*80)
        print(
            "\nApplying 5-fold cross validation to the best model on the whole dataset...")
        print("-"*80)

        result_folds_dict = {}

        # We apply 5-fold cross validation and compute the mean and std of sensitivity, specificity, accuracy and MCC. We also save each fold's results to a csv file.
        for fold, (train_ids, test_ids) in enumerate(skf.split(X_train, Y_train)):
            print(f'\nFold {fold+1}')
            x_train_fold = [X_train[i] for i in train_ids]
            y_train_fold = [Y_train[i] for i in train_ids]
            x_test_fold = [X_train[i] for i in test_ids]
            y_test_fold = [Y_train[i] for i in test_ids]

            train_dataset = GridDataset(x_train_fold, y_train_fold)
            test_dataset = GridDataset(x_test_fold, y_test_fold)

            # We create the dataloaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
            validation_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)

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
                # Evaluate the model on the validation set
                sensitivity, specificity, accuracy, mcc = validate(model)

            result_folds_dict[fold] = [sensitivity, specificity, accuracy, mcc]

        # We save the results to a csv file
        df = pd.DataFrame.from_dict(result_folds_dict, orient='index', columns=[
                                    'Sensitivity', 'Specificity', 'Accuracy', 'MCC'])
        df.to_csv(settings.RESULTS_PATH + "CNN_CV_results_" + dataset_name + "_" + dataset_type +
                  "_" + dataset_number + "_" + representation_type + "_" + representer_model + ".csv", index=False)  # Save to csv file
