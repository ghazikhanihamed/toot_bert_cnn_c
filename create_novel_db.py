import pandas as pd

path = "./dataset/"
# # Load the CSV files
# all_novel_sequences = pd.read_csv(path + 'all_novel_sequences.csv')

# # print the number of unique sequences
# print(len(all_novel_sequences['sequence'].unique()))


# Load the CSV files
all_sequence_df = pd.read_csv(path + 'all_sequences.csv')
all_sequence_new_df = pd.read_csv(path + 'all_sequences_new.csv')

# Convert the 'sequence' columns to sets for efficient comparison
sequences_old = set(all_sequence_df['sequence'])
sequences_new = set(all_sequence_new_df['sequence'])

# Find the unique sequences in the new file that are not in the old file
unique_sequences = sequences_new - sequences_old

# Filter the new DataFrame to only include these unique sequences
unique_sequences_df = all_sequence_new_df[all_sequence_new_df['sequence'].isin(unique_sequences)]

# Save the filtered DataFrame to a new CSV file
unique_sequences_df.to_csv(path + 'all_novel_sequences.csv', index=False)