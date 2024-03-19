import pandas as pd
import numpy as np
import torch
import joblib
import os
from torch import nn, optim
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from settings import settings
from classes.Classifier import CNN
from classes.PLMDataset import GridDataset
from transformers import EsmModel, EsmTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


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
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            representation = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            representations.append(representation)
            labels.append(label)
    return np.array(representations), np.array(labels)


def test_cnn(model, test_loader, device):
    model.eval()
    total = len(test_loader.dataset)
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = correct / total
    mcc = matthews_corrcoef(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    return accuracy, mcc, sensitivity, specificity


def train_cnn(network, train_loader, optimizer, device, epochs=10):
    network.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()


def train_classifier(model, X_train, y_train):
    model.fit(X_train, y_train)


def test_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    return accuracy, mcc, sensitivity, specificity


def load_esm_model_local(model_name, task):
    model_path = f"{settings.FINETUNED_MODELS_PATH}{model_name}_old/{task}"
    model = EsmModel.from_pretrained(model_path)
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    return model, tokenizer


def append_results(task, model_type, accuracy, mcc, sensitivity, specificity):
    results.append(
        {
            "Task": task,
            "Model": model_type,
            "Accuracy": accuracy,
            "MCC": mcc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
        }
    )


# Task-specific settings for Logistic Regression
lr_params = {
    "IC-MP": {"C": 10, "penalty": "l2", "solver": "liblinear"},
    "IT-MP": {"C": 100, "penalty": "l2", "solver": "liblinear"},
}

# Datasets for training
datasets = {
    "IC-MP": settings.IC_MP_Train_DATASET_OLD,
    "IT-MP": settings.IT_MP_Train_DATASET_OLD,
    "IC-IT": settings.IC_IT_Train_DATASET_OLD,
}

# Task:model dictionary
tasks_model = {"IC-MP": settings.ESM1B, "IT-MP": settings.ESM1B, "IC-IT": settings.ESM2}

results = []

# Main workflow
for task in ["IC-MP", "IT-MP", "IC-IT"]:
    # Load training data
    train_df = pd.read_csv(f"./dataset/{datasets[task]}")
    esm_model, esm_tokenizer = load_esm_model_local(tasks_model[task], task)

    # Generate representations for training data
    X_train, y_train = generate_representations(
        train_df, esm_model, esm_tokenizer, device
    )

    # Load novel data for testing
    novel_sequences_df = pd.read_csv(f"./dataset/{task}_novel_sequences.csv")
    X_test, y_test = generate_representations(
        novel_sequences_df, esm_model, esm_tokenizer, device
    )

    if task in lr_params:
        # Train Logistic Regression
        lr_model = LogisticRegression(**lr_params[task])
        train_classifier(lr_model, X_train, y_train)

        # Ensure directory exists
        ensure_dir(f"./trained_models/lr_{task}_old.joblib")

        # Save the trained model
        joblib.dump(lr_model, f"./trained_models/lr_{task}_old.joblib")

        # Test the model
        accuracy, mcc, sensitivity, specificity = test_classifier(
            lr_model, X_test, y_test
        )
    else:
        # Train CNN
        cnn_model = CNN([3, 7, 9], [128, 64, 32], 0.27, X_train.shape[1]).to(device)
        train_dataset = GridDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True
        )
        optimizer = optim.RMSprop(cnn_model.parameters(), lr=0.00021)
        train_cnn(cnn_model, train_loader, optimizer, device)

        # Ensure directory exists
        ensure_dir(f"./trained_models/cnn_{task}_old.pt")

        # Save the trained model
        torch.save(cnn_model.state_dict(), f"./trained_models/cnn_{task}_old.pt")

        # Create DataLoader for testing data
        test_dataset = GridDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False
        )

        # Test the model
        accuracy, mcc, sensitivity, specificity = test_cnn(
            cnn_model, test_loader, device
        )

    # Append results
    append_results(
        task,
        "Logistic Regression" if task in lr_params else "CNN",
        accuracy,
        mcc,
        sensitivity,
        specificity,
    )

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
ensure_dir("./model_performance_results_old_novel.csv")
results_df.to_csv("./model_performance_results_old_novel.csv", index=False)
print("Results saved to model_performance_results_old_novel.csv")
