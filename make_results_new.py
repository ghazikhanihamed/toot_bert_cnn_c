from scipy.stats import f_oneway, ttest_rel
import numpy as np
import pandas as pd
from settings import settings

# Load the CSV file to examine its contents
file_path = settings.RESULTS_PATH + "IT_MP_grid_search_details_new.csv"
data = pd.read_csv(file_path)


# Function to compute p-values using ANOVA or paired t-test
def compute_p_values(data, metric):
    # Extract unique param combinations to define groups
    unique_params = data["params"].unique()
    groups = [
        data[data["params"] == param][f"split{i}_test_{metric}"]
        for i in range(5)
        for param in unique_params
    ]

    # ANOVA for comparing more than two groups
    if len(unique_params) > 2:
        anova_p_value = f_oneway(*groups).pvalue
        return anova_p_value
    # Paired t-test for comparing two related groups
    elif len(unique_params) == 2:
        ttest_p_value = ttest_rel(groups[0], groups[1]).pvalue
        return ttest_p_value
    else:
        return np.nan


# Metrics to include in the table
metrics = ["MCC", "Accuracy", "Sensitivity", "Specificity"]

# Initialize an empty DataFrame for the LaTeX table data
latex_table_data = pd.DataFrame(
    columns=["Task", "MCC", "Accuracy", "Sensitivity", "Specificity", "P-value"]
)

# Assuming 'IT-MP' is the task for this dataset, we'll add it to our table data
task = "IT-MP"

# Compute mean±std and p-values for each metric
for metric in metrics:
    mean_std = f"{data['mean_test_' + metric].mean():.2f}±{data['mean_test_' + metric].std():.2f}"
    p_value = compute_p_values(data, metric)
    latex_table_data = latex_table_data._append(
        {"Task": task, metric: mean_std, "P-value": p_value}, ignore_index=True
    )

# Since the task is the same for all rows, we'll clean up the DataFrame
latex_table_data = latex_table_data.fillna(method="ffill").drop_duplicates()

print(latex_table_data)
