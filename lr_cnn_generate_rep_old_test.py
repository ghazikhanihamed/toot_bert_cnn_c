import pandas as pd
import numpy as np
import torch
import joblib
import os
from torch import nn
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score
from settings import settings
from classes.Classifier import CNN
from classes.PLMDataset import GridDataset
from transformers import EsmModel, EsmTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_representations_cnn(sequences_df, model, tokenizer, device):
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


def generate_representations_lr(sequences_df, model, tokenizer, device):
    representations, labels = [], []
    for _, row in sequences_df.iterrows():
        sequence, label = row["sequence"], row["label"]
        # Process the sequence as needed, e.g., replacing special characters
        sequence = (
            sequence.replace("U", "X")
            .replace("Z", "X")
            .replace("O", "X")
            .replace("B", "X")
        )

        # Tokenize and generate representations
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
            # Average pooling
            representation = np.mean(representation, axis=0)
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


def test_classifier(model, X_test, y_test, task):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    sensitivity = recall_score(
        y_test,
        y_pred,
        pos_label="ionchannels" if task == "IC-MP" else "iontransporters",
    )
    specificity = recall_score(y_test, y_pred, pos_label="membrane_proteins")
    return accuracy, mcc, sensitivity, specificity


def load_esm_model_local(model_info, task, device):
    model_path = f"{settings.FINETUNED_MODELS_PATH}/{model_info['name']}_old/{task}"
    model = EsmModel.from_pretrained(model_path)
    tokenizer = EsmTokenizer.from_pretrained(model_info["model"], do_lower_case=False)
    model.to(device)
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


# Task-specific settings for Logistic Regression, adjust as per your settings
lr_params = {
    "IC-MP": {"C": 10, "penalty": "l2", "solver": "liblinear"},
    "IT-MP": {"C": 100, "penalty": "l2", "solver": "liblinear"},
}

# Task:model dictionary
tasks_model = {"IC-MP": settings.ESM1B, "IT-MP": settings.ESM1B, "IC-IT": settings.ESM2}

results = []

# Main workflow
for task in ["IC-MP", "IT-MP", "IC-IT"]:
    test_df = pd.read_csv(f"./dataset/{task}_novel_sequences.csv")
    esm_model, esm_tokenizer = load_esm_model_local(tasks_model[task], task, device)

    if task in lr_params:
        # Generate representations for test data
        X_test, y_test = generate_representations_lr(
            test_df, esm_model, esm_tokenizer, device
        )

        # Load the trained Logistic Regression model
        lr_model = joblib.load(f"./trained_models/lr_{task}_old.joblib")

        # Test the model
        accuracy, mcc, sensitivity, specificity = test_classifier(
            lr_model, X_test, y_test, task
        )
    else:
        # Generate representations for test data
        X_test, y_test = generate_representations_cnn(
            test_df, esm_model, esm_tokenizer, device
        )

        # Load the trained CNN model
        cnn_model = CNN([3, 7, 9], [128, 64, 32], 0.27, X_test[0].shape[-1]).to(device)
        cnn_model.load_state_dict(torch.load(f"./trained_models/cnn_{task}_old.pt"))
        cnn_model.eval()

        # Prepare DataLoader for the test set
        X_test = [torch.tensor(x, dtype=torch.float32) for x in X_test]
        y_test = [torch.tensor(y, dtype=torch.long) for y in y_test]
        test_dataset = GridDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False
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
