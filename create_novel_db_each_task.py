import pandas as pd

# Load the dataset
df = pd.read_csv("./dataset/all_novel_sequences.csv")

# Define the tasks and their corresponding labels
tasks = {
    "IC-MP": ["ionchannels", "membrane_proteins"],
    "IT-MP": ["iontransporters", "membrane_proteins"],
    "IC-IT": ["ionchannels", "iontransporters"]
}

# Iterate over each task and combine sequences from the relevant labels
for task, labels in tasks.items():
    # Filter the DataFrame for the relevant labels
    filtered_df = df[df["label"].isin(labels)]
    
    # Save the combined DataFrame to a CSV file
    filtered_df.to_csv(f"./dataset/{task}_novel_sequences.csv", index=False)
    
    # Print the number of unique sequences for this task
    print(f"Number of unique sequences for {task}: {len(filtered_df['sequence'].unique())}")
