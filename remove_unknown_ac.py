from Bio import SeqIO

path = "./dataset/feb112024/raw/"

folders = ["ionchannels", "iontransporters", "mp"]
unknown_amino_acids = set("BOUXZ")

for folder in folders:
    input_file = f"{path}{folder}/{folder}.fasta"
    output_file = f"{path}{folder}/{folder}_cleaned.fasta"

    cleaned_count = 0  # Initialize counter for cleaned sequences

    with open(output_file, 'w') as cleaned_fasta:
        for record in SeqIO.parse(input_file, 'fasta'):
            # Check if the sequence contains any unknown amino acids
            if not set(record.seq.upper()).intersection(unknown_amino_acids):
                SeqIO.write(record, cleaned_fasta, 'fasta')
                cleaned_count += 1  # Increment counter for each cleaned sequence
        
    print(f'{folder}: {cleaned_count} sequences cleaned')  # Display the count of cleaned sequences