from Bio import SeqIO

path = "./dataset/feb112024/raw/"

folders = ["ionchannels", "iontransporters", "mp"]

for folder in folders:
    input_file = f"{path}{folder}/{folder}_filtered.fasta"
    num_sequences = len(list(SeqIO.parse(input_file, "fasta")))
    print(f"{folder}: {num_sequences} sequences")