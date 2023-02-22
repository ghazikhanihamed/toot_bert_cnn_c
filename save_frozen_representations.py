import h5py
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from transformers import EsmModel, EsmTokenizer
from transformers import T5EncoderModel, T5Tokenizer
from settings import settings
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load all the sequences
sequences = pd.read_csv(settings.ALL_SEQUENCES_PATH)

for rep in settings.REPRESENTATIONS:

    if rep["name"] == "ProtBERT":
        model = BertModel.from_pretrained(settings.PROTBERT["model"])
        tokenizer = BertTokenizer.from_pretrained(settings.PROTBERT["model"], do_lower_case=False)
    elif rep["name"] == "ProtBERT-BFD":
        model = BertModel.from_pretrained(settings.PROTBERTBFD["model"])
        tokenizer = BertTokenizer.from_pretrained(settings.PROTBERTBFD["model"], do_lower_case=False)
    elif rep["name"] == "ProtT5":
        model = T5EncoderModel.from_pretrained(settings.PROTT5["model"])
        tokenizer = T5Tokenizer.from_pretrained(settings.PROTT5["model"], do_lower_case=False)
    elif rep["name"] == "ESM-1b":
        model = EsmModel.from_pretrained(settings.ESM1B["model"])
        tokenizer = EsmTokenizer.from_pretrained(settings.ESM1B["model"], do_lower_case=False)
    elif rep["name"] == "ESM-2":
        model = EsmModel.from_pretrained(settings.ESM2["model"])
        tokenizer = EsmTokenizer.from_pretrained(settings.ESM2["model"], do_lower_case=False)
    else:
        model = EsmModel.from_pretrained(settings.ESM2_15B["model"])
        tokenizer = EsmTokenizer.from_pretrained(settings.ESM2_15B["model"], do_lower_case=False)

    model = model.to(device)
    model = model.eval()

    representations = []

    # For each sequence, label and id in the dataframe we take frozen representations from each rep
    for sequence, label, id in zip(sequences["sequence"], sequences["label"], sequences["id"]):
        # We put a space between each amino acid
        sequence = " ".join(sequence)
        # Replace UZOB with X
        sequence = sequence.replace("U", "X")
        sequence = sequence.replace("Z", "X")
        sequence = sequence.replace("O", "X")
        sequence = sequence.replace("B", "X")
        # Tokenize the sequence
        inputs = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=settings.MAX_LENGTH_FROZEN, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Obtain frozen representations
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            representation = outputs.last_hidden_state[0].detach().numpy()
            representations.append((representation, label, id))

    # Save the frozen representations
    with h5py.File(settings.FROZEN_REPRESENTATIONS_PATH + "_" + rep["name"] + ".h5", "w") as f:
        for representation, label, id in representations:
            f.create_dataset(str(id), data=representation)
            f[str(id)].attrs["label"] = label

