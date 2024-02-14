import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score, make_scorer
from settings import settings


def load_data(df, representations_path):
    X, y = [], []
    with h5py.File(representations_path, "r") as f:
        for seq_id in df["id"]:
            if str(seq_id) in f:
                X.append(np.mean(f[str(seq_id)][()], axis=0))  # Average pooling
                y.append(f[str(seq_id)].attrs["label"])
    return np.array(X), np.array(
        [1 if label == settings.MEMBRANE_PROTEINS else 0 for label in y]
    )


def save_best_params(best_params, task_name):
    params_df = pd.DataFrame([best_params])
    params_df.to_csv(
        f"{settings.RESULTS_PATH}/{task_name}_best_params_new.csv", index=False
    )


def save_grid_search_details(grid_search, task_name):
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(
        f"{settings.RESULTS_PATH}/{task_name}_grid_search_details_new.csv", index=False
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
        f"{settings.RESULTS_PATH}/{task_name}_grid_search_summary_new.csv"
    )


def test_best_model(model, X_test, y_test, task_name):
    y_pred = model.predict(X_test)
    test_results = {
        "Task": task_name,
        "Sensitivity": recall_score(y_test, y_pred, pos_label=1),
        "Specificity": recall_score(y_test, y_pred, pos_label=0),
        "Accuracy": accuracy_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    return pd.DataFrame([test_results])


# Define the tasks and corresponding CSV files
csv_files = {
    "IC-MP": "IC-MP_sequences.csv",
    "IT-MP": "IT-MP_sequences.csv",
}

# Define the tasks
tasks = ["IC_MP", "IT_MP"]

# Parameters for GridSearchCV
param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.1, 1, 10, 100],
    "solver": ["liblinear", "saga"],
}

# Scoring metrics
scoring = {
    "Sensitivity": make_scorer(recall_score, pos_label=1),
    "Specificity": make_scorer(recall_score, pos_label=0),
    "Accuracy": make_scorer(accuracy_score),
    "MCC": make_scorer(matthews_corrcoef),
}

# Initialize a DataFrame to store final test set results
final_test_results = pd.DataFrame(
    columns=["Task", "Sensitivity", "Specificity", "Accuracy", "MCC"]
)

for task_name in tasks:
    # Load the training and test sequences
    train_df = pd.read_csv(f"{settings.DATASET_PATH}/{task_name}_train.csv")
    test_df = pd.read_csv(f"{settings.DATASET_PATH}/{task_name}_test.csv")

    # Load representations and prepare data
    X_train, y_train = load_data(train_df, task_name, settings)
    X_test, y_test = load_data(test_df, task_name, settings)

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
    save_best_params(grid_search, task_name, settings)
    save_grid_search_details(grid_search, task_name, settings)
    save_grid_search_summary(grid_search, task_name, settings)

    # Test the best model on the test set and save results
    test_metrics = test_best_model(
        grid_search.best_estimator_, X_test, y_test, task_name
    )
    final_test_results = final_test_results.append(test_metrics, ignore_index=True)

# Save final test set results
final_test_results.to_csv(
    f"{settings.RESULTS_PATH}/final_test_results_new.csv", index=False
)
