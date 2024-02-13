import json
from Bio import SeqIO
from settings import settings

# Path to your datasets
path = "./dataset/feb112024/raw/"

# Folders representing each class
folders = ["ionchannels", "iontransporters", "mp"]

# Initialize an empty set to hold all unique sequence IDs
all_sequence_ids = set()

# Iterate over each folder and collect all sequence IDs
for folder in folders:
    # Define the path to the filtered dataset
    filtered_file = f"{path}{folder}/{folder}_filtered.fasta"
    
    # Read sequences from the filtered dataset
    for record in SeqIO.parse(filtered_file, "fasta"):
        # Add the sequence ID to the set
        all_sequence_ids.add(record.id)

# Convert the set of sequence IDs to a list and sort it
sequence_ids = sorted(list(all_sequence_ids))

# Create a dictionary mapping each sequence ID to a unique numeric ID
seqid_dict = {seq_id: i for i, seq_id in enumerate(sequence_ids)}

# Define the path where the dictionary will be saved
# Replace 'settings.SEQUENCE_IDS_DICT_PATH' with the actual path where you want to save the dictionary
sequence_ids_dict_path = settings.SEQUENCE_IDS_DICT_PATH

# Save the dictionary as a JSON file
with open(sequence_ids_dict_path, "w") as f:
    json.dump(seqid_dict, f)

print(f"Saved sequence ID dictionary to {sequence_ids_dict_path}")
