import pandas as pd
from Bio import SeqIO
from settings import settings

# Define the paths to your dataset directories
path = "./dataset/feb112024/raw/"

# Define the tasks and their corresponding classes
tasks = {
    "IC-MP": [settings.IONCHANNELS, settings.MEMBRANE_PROTEINS],
    "IT-MP": [settings.IONTRANSPORTERS, settings.MEMBRANE_PROTEINS],
    "IC-IT": [settings.IONCHANNELS, settings.IONTRANSPORTERS],
}

# Iterate over each task to process train and test datasets
for task_name, classes in tasks.items():
    # Initialize a list to store all rows for the current task
    task_rows = []

    for class_folder in classes:
        # Define file paths for train and test datasets
        train_file = f"{path}{class_folder}/{class_folder}_train.fasta"
        test_file = f"{path}{class_folder}/{class_folder}_test.fasta"

        # Process train file
        for record in SeqIO.parse(train_file, "fasta"):
            task_rows.append({"sequence": str(record.seq), "label": class_folder, "id": record.id})

        # Process test file
        for record in SeqIO.parse(test_file, "fasta"):
            task_rows.append({"sequence": str(record.seq), "label": class_folder, "id": record.id})

    # Convert the list to a DataFrame for the current task
    task_sequences_df = pd.DataFrame(task_rows, columns=["sequence", "label", "id"])

    # Save the DataFrame to a CSV file named after the task
    task_csv_file = f"{settings.DATASET_PATH}/{task_name}_sequences.csv"
    task_sequences_df.to_csv(task_csv_file, index=False)

    print(f"Saved sequences for task {task_name} to {task_csv_file}.")
