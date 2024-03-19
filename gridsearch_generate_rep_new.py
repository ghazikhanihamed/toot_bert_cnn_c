import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score, make_scorer
from settings import settings
import joblib
from transformers import EsmModel, EsmTokenizer

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Function to process sequences and generate representations
def generate_representations(sequences_df, model, tokenizer, device):
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


def save_best_params(grid_search, task_name):
    params_df = pd.DataFrame(
        list(grid_search.best_params_.items()), columns=["Parameter", "Value"]
    )
    params_df.to_csv(
        f"{settings.RESULTS_PATH}/{task_name}_best_params_generated_rep_new.csv",
        index=False,
    )


def save_grid_search_details(grid_search, task_name):
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(
        f"{settings.RESULTS_PATH}/{task_name}_grid_search_details_generated_rep_new.csv",
        index=False,
    )


def save_grid_search_summary(grid_search, task_name):
    summary = {
        "Sensitivity": {
            "Train": f'{grid_search.cv_results_["mean_train_Sensitivity"][grid_search.best_index_]:.2f} ± {grid_search.cv_results_["std_train_Sensitivity"][grid_search.best_index_]:.2f}',
            "Val": f'{grid_search.cv_results_["mean_test_Sensitivity"][grid_search.best_index_]:.2f} ± {grid_search.cv_results_["std_test_Sensitivity"][grid_search.best_index_]:.2f}',
        },
        "Specificity": {
            "Train": f'{grid_search.cv_results_["mean_train_Specificity"][grid_search.best_index_]:.2f} ± {grid_search.cv_results_["std_train_Specificity"][grid_search.best_index_]:.2f}',
            "Val": f'{grid_search.cv_results_["mean_test_Specificity"][grid_search.best_index_]:.2f} ± {grid_search.cv_results_["std_test_Specificity"][grid_search.best_index_]:.2f}',
        },
        "Accuracy": {
            "Train": f'{grid_search.cv_results_["mean_train_Accuracy"][grid_search.best_index_]:.2f} ± {grid_search.cv_results_["std_train_Accuracy"][grid_search.best_index_]:.2f}',
            "Val": f'{grid_search.cv_results_["mean_test_Accuracy"][grid_search.best_index_]:.2f} ± {grid_search.cv_results_["std_test_Accuracy"][grid_search.best_index_]:.2f}',
        },
        "MCC": {
            "Train": f'{grid_search.cv_results_["mean_train_MCC"][grid_search.best_index_]:.2f} ± {grid_search.cv_results_["std_train_MCC"][grid_search.best_index_]:.2f}',
            "Val": f'{grid_search.cv_results_["mean_test_MCC"][grid_search.best_index_]:.2f} ± {grid_search.cv_results_["std_test_MCC"][grid_search.best_index_]:.2f}',
        },
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        f"{settings.RESULTS_PATH}/{task_name}_grid_search_summary_generated_rep_new.csv"
    )


def test_best_model(model, X_test, y_test, task_name):
    y_pred = model.predict(X_test)
    test_results = {
        "Task": task_name,
        "Sensitivity": recall_score(
            y_test,
            y_pred,
            pos_label="ionchannels" if task_name == "IC_MP" else "iontransporters",
        ),
        "Specificity": recall_score(y_test, y_pred, pos_label="membrane_proteins"),
        "Accuracy": accuracy_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    return test_results


def load_esm_model(model_info, device):
    model = EsmModel.from_pretrained(model_info["model"])
    tokenizer = EsmTokenizer.from_pretrained(model_info["model"], do_lower_case=False)
    model.to(device)
    return model, tokenizer


# Define the tasks
tasks = ["IC_MP", "IT_MP"]

# Parameters for GridSearchCV
param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.1, 1, 10, 100],
    "solver": ["liblinear", "saga"],
}

# Initialize a list to store final test set results
final_test_results_list = []

for task_name in tasks:
    # Scoring metrics
    scoring = {
        "Sensitivity": make_scorer(
            recall_score,
            pos_label="ionchannels" if task_name == "IC_MP" else "iontransporters",
        ),
        "Specificity": make_scorer(recall_score, pos_label="membrane_proteins"),
        "Accuracy": make_scorer(accuracy_score),
        "MCC": make_scorer(matthews_corrcoef),
    }
    # Load the sequences
    train_df = pd.read_csv(f"{settings.DATASET_PATH}{task_name}_train.csv")
    test_df = pd.read_csv(f"{settings.DATASET_PATH}{task_name}_test.csv")

    model_info = settings.ESM1B
    esm_model, tokenizer = load_esm_model(model_info, device)

    # Generate representations
    X_train, y_train = generate_representations(train_df, esm_model, tokenizer, device)
    X_test, y_test = generate_representations(test_df, esm_model, tokenizer, device)

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.SEED)

    # Initialize Logistic Regression model
    lr_model = LogisticRegression(random_state=settings.SEED)

    # Perform grid search
    grid_search = GridSearchCV(
        lr_model,
        param_grid,
        cv=skf,
        scoring=scoring,
        refit="MCC",
        return_train_score=True,
        n_jobs=20,
    )
    grid_search.fit(X_train, y_train)

    # Save best parameters and grid search details
    save_best_params(grid_search, task_name)
    save_grid_search_details(grid_search, task_name)
    save_grid_search_summary(grid_search, task_name)

    # Retrain best model on the entire training set
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)  # Retraining on the entire training set

    # Save the best model
    model_filename = (
        f"{settings.FINAL_MODELS_PATH}final_model_generated_rep_{task_name}.joblib"
    )
    joblib.dump(best_model, model_filename)
    print(f"Best Logistic Regression model saved to {model_filename}")

    # Test the best model on the test set and accumulate results
    test_metrics = test_best_model(best_model, X_test, y_test, task_name)
    final_test_results_list.append(test_metrics)

# Convert the accumulated results list to a DataFrame
final_test_results = pd.DataFrame(final_test_results_list)

# Save final test set results
final_test_results.to_csv(
    f"{settings.RESULTS_PATH}final_test_results_generated_rep_new.csv", index=False
)
