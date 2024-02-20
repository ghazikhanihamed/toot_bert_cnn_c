import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score
from settings import settings
from classes.Classifier import CNN
from classes.PLMDataset import GridDataset
from transformers import EsmModel, EsmTokenizer

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to generate representations using ESM-2
def generate_representations(sequences_df, model, tokenizer, device):
    representations, labels = [], []
    for _, row in sequences_df.iterrows():
        sequence, label = row["sequence"], row["label"]
        sequence = (
            sequence.replace("U", "X")
            .replace("Z", "X")
            .replace("O", "X")
            .replace("B", "X")
        )
        inputs = tokenizer(
            sequence,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            representation = outputs.last_hidden_state[0].cpu().numpy()
            representations.append(torch.tensor(representation, dtype=torch.float))
            labels.append(label)
    return representations, np.array(
        [1 if label == settings.IONCHANNELS else 0 for label in labels]
    )


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


def load_esm_model(model_info):
    model = EsmModel.from_pretrained(model_info["model"])
    tokenizer = EsmTokenizer.from_pretrained(model_info["model"], do_lower_case=False)
    return model, tokenizer


# Load ESM-2 model and tokenizer
model_info = settings.ESM2
esm_model, tokenizer = load_esm_model(model_info)
esm_model.to(device)

tasks = ["IC_IT"]
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=settings.SEED)
final_test_results_list = []

for task_name in tasks:
    train_df = pd.read_csv(f"{settings.DATASET_PATH}/{task_name}_train.csv")
    test_df = pd.read_csv(f"{settings.DATASET_PATH}/{task_name}_test.csv")

    # Generate representations using ESM-2
    X_train, y_train = generate_representations(train_df, esm_model, tokenizer, device)
    X_test, y_test = generate_representations(test_df, esm_model, tokenizer, device)

    # Assuming the CNN model can handle the 2D input shape directly
    input_dim = X_train[0].shape[-1]  # Get the last dimension size as input_dim for CNN

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

        # Convert train_ids and test_ids to arrays
        train_ids = np.array(train_ids)
        test_ids = np.array(test_ids)

        # Split data for this fold
        X_train_fold = [X_train[i] for i in train_ids]
        y_train_fold = [y_train[i] for i in train_ids]
        X_val_fold = [X_train[i] for i in test_ids]
        y_val_fold = [y_train[i] for i in test_ids]

        # Convert lists to tensors
        X_train_fold = [torch.tensor(x, dtype=torch.float) for x in X_train_fold]
        y_train_fold = torch.tensor(y_train_fold, dtype=torch.long)
        X_val_fold = [torch.tensor(x, dtype=torch.float) for x in X_val_fold]
        y_val_fold = torch.tensor(y_val_fold, dtype=torch.long)

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
        f"{settings.RESULTS_PATH}{task_name}_cv_results_generated_rep_ic_it_new.csv",
        index=False,
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

    # After training and cross-validation, save the CNN model
    model_path = f"{settings.FINAL_MODELS_PATH}final_model_generated_rep_{task_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Save final test set results
final_test_results = pd.DataFrame(final_test_results_list)
final_test_results.to_csv(
    f"{settings.RESULTS_PATH}final_test_results_ic_it_generated_rep_new.csv",
    index=False,
)
