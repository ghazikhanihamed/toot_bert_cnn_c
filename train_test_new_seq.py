from Bio import SeqIO
import random

path = "./dataset/feb112024/raw/"

folders = ["ionchannels", "iontransporters", "mp"]

# Seed the random number generator for reproducibility
random.seed(42)

for folder in folders:
    input_file = f"{path}{folder}/{folder}_filtered.fasta"
    
    # Read all sequences into a list
    sequences = list(SeqIO.parse(input_file, "fasta"))
    
    # Shuffle the list of sequences randomly
    random.shuffle(sequences)
    
    # Calculate the split index for 80% train, 20% test
    split_index = int(len(sequences) * 0.8)
    
    # Split the sequences
    train_sequences = sequences[:split_index]
    test_sequences = sequences[split_index:]
    
    # Write the train and test sequences to separate files
    train_file = f"{path}{folder}/{folder}_train.fasta"
    test_file = f"{path}{folder}/{folder}_test.fasta"
    
    with open(train_file, "w") as output_handle:
        SeqIO.write(train_sequences, output_handle, "fasta")
        
    with open(test_file, "w") as output_handle:
        SeqIO.write(test_sequences, output_handle, "fasta")
        
    print(f"{folder}: {len(train_sequences)} train sequences, {len(test_sequences)} test sequences")
