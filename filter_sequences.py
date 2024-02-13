from Bio import SeqIO

path = "./dataset/feb112024/raw/"

folders = ["ionchannels", "iontransporters", "mp"]


def filter_sequences(psiblast_output, threshold=20):
    # Read PSI-BLAST output and find sequences with >20% similarity
    similar_sequences = set()
    with open(psiblast_output) as f:
        for line in f:
            parts = line.strip().split()
            percent_identity = float(parts[2])
            if percent_identity > threshold:
                similar_sequences.add(parts[1])  # subject id is in the second column

    return similar_sequences


for folder in folders:
    input_file = f"{path}{folder}/{folder}_cleaned.fasta"
    psiblast_output = f"{path}{folder}/psiblast_results.txt"
    sequences_to_exclude = filter_sequences(psiblast_output)

    # Read the original cleaned sequences and exclude the identified similar sequences
    filtered_sequences = []
    for record in SeqIO.parse(input_file, "fasta"):
        if record.id not in sequences_to_exclude:
            filtered_sequences.append(record)

    # Count and print the number of sequences after exclusion
    num_sequences_after_exclusion = len(filtered_sequences)
    print(
        f"{folder}: {num_sequences_after_exclusion} sequences after excluding similar ones"
    )

    # Write the filtered sequences to a new file
    output_file = f"{path}{folder}/{folder}_filtered.fasta"
    with open(output_file, "w") as output_handle:
        SeqIO.write(filtered_sequences, output_handle, "fasta")
