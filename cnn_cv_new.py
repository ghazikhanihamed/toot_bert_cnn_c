import pandas as pd
import numpy as np
import h5py
import torch
from torch import nn, optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score
from settings import settings
from classes.Classifier import CNN
from classes.PLMDataset import GridDataset


# Function to load data
def load_data(df, representations_path):
    X, y = [], []
    with h5py.File(f"{settings.REPRESENTATIONS_PATH}{representations_path}", "r") as f:
        for seq_id in df["id"]:
            if str(seq_id) in f:
                # Append the representation as a PyTorch tensor
                X.append(torch.tensor(f[str(seq_id)][()], dtype=torch.float))
                y.append(f[str(seq_id)].attrs["label"])
    # Convert y labels to 0s and 1s
    y = np.array([1 if label == settings.IONCHANNELS else 0 for label in y])
    return X, y


# Function to train the CNN model
def train(network, train_loader, optimizer, device):
    network.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()


# Function to validate the CNN model
def validate(network, validation_loader, device):
    network.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device)
            output = network(data)
            pred = output.data.max(1, keepdim=True)[1]
            y_true.extend(target.numpy())
            y_pred.extend(pred.cpu().numpy().flatten())
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return sensitivity, specificity, accuracy, mcc


# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=settings.SEED)
final_test_results_list = []
tasks = ["IC_IT"]

for task_name in tasks:
    train_df = pd.read_csv(f"{settings.DATASET_PATH}/{task_name}_train.csv")
    test_df = pd.read_csv(f"{settings.DATASET_PATH}/{task_name}_test.csv")

    X_train, y_train = load_data(train_df, f"{task_name}_representations.h5")
    X_test, y_test = load_data(test_df, f"{task_name}_representations.h5")

    input_dim = X_train[0].shape[1]

    cnn_params = {
        "dropout_prob": 0.27,
        "kernel_sizes": [3, 7, 9],
        "lr": 0.00021,
        "optimizer": "RMSprop",
        "out_channels": [128, 64, 32],
    }

    cv_results = []

    for fold, (train_ids, test_ids) in enumerate(skf.split(X_train, y_train)):
        # We create the CNN model with the best hyperparameters for each fold
        model = CNN(
            cnn_params["kernel_sizes"],
            cnn_params["out_channels"],
            cnn_params["dropout_prob"],
            input_dim,
        ).to(device)
        # We create the optimizer with the best hyperparameters for each fold
        optimizer_name = cnn_params["optimizer"]
        lr = cnn_params["lr"]
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        # Split data for this fold
        X_train_fold, y_train_fold = X_train[train_ids], y_train[train_ids]
        X_val_fold, y_val_fold = X_train[test_ids], y_train[test_ids]

        train_dataset = GridDataset(X_train_fold, y_train_fold)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True
        )

        validation_dataset = GridDataset(X_val_fold, y_val_fold)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=settings.BATCH_SIZE, shuffle=False
        )

        # Train and validate the model for this fold
        train(model, train_loader, optimizer, device)
        sensitivity, specificity, accuracy, mcc = validate(
            model, validation_loader, device
        )

        cv_results.append([sensitivity, specificity, accuracy, mcc])

    # Save cross-validation results to CSV
    cv_results_df = pd.DataFrame(
        cv_results, columns=["Sensitivity", "Specificity", "Accuracy", "MCC"]
    )
    cv_results_df.to_csv(
        f"{settings.RESULTS_PATH}/{task_name}_cv_results_ic_it_new.csv", index=False
    )

    # reinitialize the model and optimizer
    # We create the CNN model with the best hyperparameters for each fold
    model = CNN(
        cnn_params["kernel_sizes"],
        cnn_params["out_channels"],
        cnn_params["dropout_prob"],
        input_dim,
    ).to(device)
    # We create the optimizer with the best hyperparameters for each fold
    optimizer_name = cnn_params["optimizer"]
    lr = cnn_params["lr"]
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_dataset = GridDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True
    )

    test_dataset = GridDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False
    )

    train(model, train_loader, optimizer, device)
    sensitivity, specificity, accuracy, mcc = validate(model, test_loader, device)
    final_test_results_list.append(
        {
            "Task": task_name,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Accuracy": accuracy,
            "MCC": mcc,
        }
    )

    # After training the model
    model_path = f"{settings.FINAL_MODELS_PATH}/final_model_{task_name}.pt"

    # Save the model state dictionary
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")

# Save final test set results
final_test_results = pd.DataFrame(final_test_results_list)
final_test_results.to_csv(
    f"{settings.RESULTS_PATH}/final_test_results_ic_it_new.csv", index=False
)


# input_dim = 1280 # Dimension of ESM-2

# cnn_params = {
#     "dropout_prob": 0.27,
#     "kernel_sizes": [3, 7, 9],
#     "lr": 0.00021,
#     "optimizer": "RMSprop",
#     "out_channels": [128, 64, 32],
# }
# To use the saved model:
# Reinitialize the model architecture
# model = CNN(
#     cnn_params["kernel_sizes"],
#     cnn_params["out_channels"],
#     cnn_params["dropout_prob"],
#     input_dim,
# ).to(device)

# # Load the saved state dictionary into the model
# model.load_state_dict(torch.load(model_path))

# # Remember to call model.eval() to set dropout and batch normalization layers to evaluation mode before making predictions
# model.eval()
