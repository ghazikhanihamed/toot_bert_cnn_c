import pandas as pd
from Bio import SeqIO
import os
from settings import settings

# Define the paths to your dataset directories
path = "./dataset/feb112024/raw/"

# Define the tasks and their corresponding classes
tasks = {
    "IC-MP": [settings.IONCHANNELS, settings.MEMBRANE_PROTEINS],
    "IT-MP": [settings.IONTRANSPORTERS, settings.MEMBRANE_PROTEINS],
    "IC-IT": [settings.IONCHANNELS, settings.IONTRANSPORTERS],
}

# Iterate over each task to create train and test CSV files
for task_name, classes in tasks.items():
    # Lists to store rows before appending to DataFrame
    train_rows = []
    test_rows = []

    for class_folder in classes:
        train_file = f"{path}{class_folder}/{class_folder}_train.fasta"
        test_file = f"{path}{class_folder}/{class_folder}_test.fasta"

        # Process train file
        for record in SeqIO.parse(train_file, "fasta"):
            train_rows.append({"sequence": str(record.seq), "label": class_folder, "id": record.id})

        # Process test file
        for record in SeqIO.parse(test_file, "fasta"):
            test_rows.append({"sequence": str(record.seq), "label": class_folder, "id": record.id})

    # Convert lists to DataFrames and append to existing DataFrame
    train_df = pd.DataFrame(train_rows, columns=["sequence", "label", "id"])
    test_df = pd.DataFrame(test_rows, columns=["sequence", "label", "id"])

    # Save to CSV
    train_df.to_csv(f"{settings.DATASET_PATH}/{task_name}_train.csv", index=False)
    test_df.to_csv(f"{settings.DATASET_PATH}/{task_name}_test.csv", index=False)

    print(f"Saved {task_name} train and test datasets to CSV files.")
