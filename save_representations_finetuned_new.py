import pandas as pd
import torch
import h5py
from settings import settings
from transformers import EsmModel, EsmTokenizer

torch.cuda.empty_cache()

# Assuming you have a way to map task names to their corresponding CSV files
csv_files = {
    "IC-MP": "IC-MP_sequences.csv",
    "IT-MP": "IT-MP_sequences.csv",
    "IC-IT": "IC-IT_sequences.csv",
}

# Define the tasks and corresponding models
tasks = {"IC-MP": settings.ESM1B, "IT-MP": settings.ESM1B, "IC-IT": settings.ESM2}

for task, model_info in tasks.items():
    # Load the sequences for the current task
    sequences_df = pd.read_csv(f"{settings.DATASET_PATH}/{csv_files[task]}")

    # Define the output directory for the fine-tuned model
    finetuned_model_path = (
        f"{settings.FINETUNED_MODELS_PATH}/{model_info['name']}/{task}"
    )

    # Load the fine-tuned model and tokenizer
    model = EsmModel.from_pretrained(finetuned_model_path)
    tokenizer = EsmTokenizer.from_pretrained(model_info["model"], do_lower_case=False)

    print("Finetuned model: ", task)
    print("-" * 50)

    # Set up the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    representations = []

    # Process each sequence in the dataframe
    for _, row in sequences_df.iterrows():
        sequence, label, seq_id = row["sequence"], row["label"], row["id"]
        # Replace UZOB with X
        sequence = (
            sequence.replace("U", "X")
            .replace("Z", "X")
            .replace("O", "X")
            .replace("B", "X")
        )

        # Tokenize the sequence
        inputs = tokenizer(
            sequence,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Obtain representations
        with torch.no_grad():
            outputs = model(**inputs)
            representation = outputs.last_hidden_state[0].cpu().numpy()
            representations.append((representation, label, seq_id))

    # Save the representations to an HDF5 file
    with h5py.File(
        f"{settings.REPRESENTATIONS_PATH}/{task}_representations.h5", "w"
    ) as f:
        for representation, label, seq_id in representations:
            f.create_dataset(str(seq_id), data=representation)
            f[str(seq_id)].attrs["label"] = label

    print(f"Saved representations for task {task}.")
