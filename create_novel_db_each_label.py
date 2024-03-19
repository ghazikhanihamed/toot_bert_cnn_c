import pandas as pd

# Load the dataset
df = pd.read_csv("./dataset/all_novel_sequences.csv")

# List of unique labels
labels = {"ionchannels": "IC", "membrane_proteins": "MP", "iontransporters": "IT"}

# Separate and save sequences based on labels
for label in labels:
    filtered_df = df[df["label"] == label]
    filtered_df.to_csv(f"./dataset/{labels[label]}_novel_sequences.csv", index=False)
    # print the number of unique sequences
    print(
        f"Number of unique sequences for {labels[label]}: {len(filtered_df['sequence'].unique())}"
    )
