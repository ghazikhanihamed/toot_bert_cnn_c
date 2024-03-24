import pandas as pd
from settings import settings


# List of files to process
files = ["IC-IT_new.csv", "IC-MP_new.csv", "IT-MP_new.csv", "all_novel_sequences.csv", "IC_IT_train.csv", "IC_MP_train.csv", "IT_MP_train.csv", "IC_IT_test.csv", "IC_MP_test.csv", "IT_MP_test.csv"]

# Iterate through each file and count the labels
for file in files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(settings.DATASET_PATH + file)
    
    # Count the occurrences of each label
    label_counts = df['label'].value_counts()
    
    # Print the counts for each label in the current file
    print(f"Counts for {file}:")
    print(label_counts)
    print()  # Print a blank line for better readability between files