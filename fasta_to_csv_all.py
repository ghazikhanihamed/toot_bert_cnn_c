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

# Initialize a list to store all rows before appending to DataFrame
all_rows = []

# Iterate over each task to process train and test datasets
for task_name, classes in tasks.items():

    for class_folder in classes:
        train_file = f"{path}{class_folder}/{class_folder}_train.fasta"
        test_file = f"{path}{class_folder}/{class_folder}_test.fasta"

        # Process train file
        for record in SeqIO.parse(train_file, "fasta"):
            all_rows.append(
                {"sequence": str(record.seq), "label": class_folder, "id": record.id}
            )

        # Process test file
        for record in SeqIO.parse(test_file, "fasta"):
            all_rows.append(
                {"sequence": str(record.seq), "label": class_folder, "id": record.id}
            )

# Convert the list to a DataFrame
all_sequences_df = pd.DataFrame(all_rows, columns=["sequence", "label", "id"])

# Save the consolidated DataFrame to a CSV file
all_sequences_df.to_csv(f"{settings.DATASET_PATH}/all_sequences_new.csv", index=False)

print("Saved all sequences to all_sequences_new.csv.")
