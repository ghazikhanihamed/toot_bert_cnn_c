import pandas as pd
import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from settings import settings
import torch
from transformers import EsmModel, EsmTokenizer
from joblib import dump
from joblib import load


# Function to load data
def load_data(df, representations_path):
    X, y = [], []
    with h5py.File(
        f"{settings.REPRESENTATIONS_PATH}/{representations_path}_representations.h5",
        "r",
    ) as f:
        for seq_id in df["id"]:
            X.append(np.mean(f[str(seq_id)][()], axis=0))
            y.append(f[str(seq_id)].attrs["label"])
    return np.array(X), np.array(
        [0 if label == settings.MEMBRANE_PROTEINS else 1 for label in y]
    )


# Function to evaluate the model
def evaluate_model(model, X, y, dataset_type="Test"):
    y_pred = model.predict(X)
    print(f"--- {dataset_type} Dataset Evaluation ---")
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


# Function for single sequence prediction
def predict_single_sequence(sequence, tokenizer, esm_model, lr_model):
    sequence = (
        sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    )
    inputs = tokenizer(
        sequence,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    with torch.no_grad():
        outputs = esm_model(**inputs)
        representation = np.mean(outputs.last_hidden_state.cpu().numpy(), axis=1)
    proba = lr_model.predict_proba(representation.reshape(1, -1))
    prediction = lr_model.predict(representation.reshape(1, -1))
    return prediction[0], proba[0]


def load_esm_model(model_info):
    model = EsmModel.from_pretrained(model_info["model"])
    tokenizer = EsmTokenizer.from_pretrained(model_info["model"], do_lower_case=False)
    return model, tokenizer


# Load models and tokenizer
tasks = {"IT_MP": settings.ESM1B}
task_name = "IT_MP"
model_info = tasks[task_name]
esm_model, tokenizer = load_esm_model(model_info)

# Load and prepare data
train_df = pd.read_csv(f"{settings.DATASET_PATH}{task_name}_train.csv")
test_df = pd.read_csv(f"{settings.DATASET_PATH}{task_name}_test.csv")
X_train, y_train = load_data(train_df, task_name)
X_test, y_test = load_data(test_df, task_name)

# Train LR model
lr_model = LogisticRegression(
    random_state=settings.SEED, C=10, penalty="l1", solver="liblinear"
)
lr_model.fit(X_train, y_train)

# Save the model to a file
model_filename = f"final_lr_model{task_name}.joblib"
dump(lr_model, model_filename)
print(f"Model saved to {model_filename}")

# Evaluate model
evaluate_model(lr_model, X_train, y_train, "Train")
evaluate_model(lr_model, X_test, y_test, "Test")

# delete the model
del lr_model

# Load the model from the file
lr_model = load(model_filename)
print("Model loaded successfully")

# Example sequence for prediction
sequence = "MVRCDRGLQMLLTTAGAFAAFSLMAIAIGTDYWLYSSAHICNGTNLTMDDGPPPRRARGDLTHSGLWRVCCIEGIYKGHCFRINHFPEDNDYDHDSSEYLLRIVRASSVFPILSTILLLLGGLCIGAGRIYSRKNNIVLSAGILFVAAGLSNIIGIIVYISSNTGDPSDKRDEDKKNHYNYGWSFYFGALSFIVAETVGVLAVNIYIEKNKELRFKTKREFLKASSSSPYSRMPSYRYRRRRSSSSRSTEASPSRDASPVGLKITGAIPMGELSMYTLSREPLKVTTAASYSPDQDAGFLQMHDFFQQDLKEGFHVSMLNRRTTPV"
prediction, proba = predict_single_sequence(sequence, tokenizer, esm_model, lr_model)
print(
    f"Prediction: {'Ion Channel' if prediction == 1 else 'Non-ionic Membrane Protein'}, Probabilities: {proba}"
)
