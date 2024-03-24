# import pandas as pd
from settings import settings

# # Load the CSV file into a DataFrame
# df = pd.read_csv(settings.DATASET_PATH + "/IC-MP_new.csv")  # Ensure the path is correctly concatenated

# # Check the total number of sequences
# total_sequences = len(df)
# print(f"Total number of sequences: {total_sequences}")

# # Check for duplicate sequences
# duplicates = df[df.duplicated(subset='sequence', keep=False)]

# # Report the number of duplicates found
# print(f"Number of duplicate sequences: {len(duplicates)}")


import pandas as pd

# List of files to process
files = [
    "IC-IT_new.csv",
    "IC-MP_new.csv",
    "IT-MP_new.csv"
]

# Loop through each file, remove duplicates, and save to a new file
for file_name in files:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(f"{settings.DATASET_PATH}{file_name}")

    # Remove duplicate sequences, keeping the first occurrence
    cleaned_df = df.drop_duplicates(subset='sequence', keep='first')

    # Save the cleaned DataFrame to a new CSV file
    cleaned_df.to_csv(f"{settings.DATASET_PATH}cleaned_{file_name}", index=False)

    # Report the number of sequences removed
    num_removed = len(df) - len(cleaned_df)
    print(f"Removed {num_removed} duplicate sequences from {file_name}")
    
    # Count the number of each label in the cleaned DataFrame
    label_counts = cleaned_df['label'].value_counts()

    # Print the counts for each label in the cleaned file
    print(f"Label counts in cleaned {file_name}:")
    print(label_counts, "\n")