import h5py
import os
import torch
from transformers import BertModel, BertTokenizer
from transformers import EsmModel, EsmTokenizer
from transformers import T5EncoderModel, T5Tokenizer
from settings import settings
import pandas as pd

torch.cuda.empty_cache()

# Load all the sequences
sequences = pd.read_csv(settings.ALL_SEQUENCES_PATH)

for rep in settings.REPRESENTATIONS:
    # We check the saved frozen representations if exists then we skip this rep
    if os.path.exists(settings.FROZEN_REPRESENTATIONS_PATH + "_" + rep["name"] + ".h5"):
        continue

    if rep["name"] == "ProtBERT":
        model = BertModel.from_pretrained(settings.PROTBERT["model"])
        tokenizer = BertTokenizer.from_pretrained(settings.PROTBERT["model"])
    elif rep["name"] == "ProtBERT-BFD":
        model = BertModel.from_pretrained(settings.PROTBERTBFD["model"])
        tokenizer = BertTokenizer.from_pretrained(settings.PROTBERTBFD["model"])
    elif rep["name"] == "ProtT5":
        model = T5EncoderModel.from_pretrained(settings.PROTT5["model"])
        tokenizer = T5Tokenizer.from_pretrained(settings.PROTT5["model"])
    elif rep["name"] == "ESM-1b":
        model = EsmModel.from_pretrained(settings.ESM1B["model"])
        tokenizer = EsmTokenizer.from_pretrained(settings.ESM1B["model"])
    elif rep["name"] == "ESM-2":
        model = EsmModel.from_pretrained(settings.ESM2["model"])
        tokenizer = EsmTokenizer.from_pretrained(settings.ESM2["model"])
    else:
        model = EsmModel.from_pretrained(settings.ESM2_15B["model"])
        tokenizer = EsmTokenizer.from_pretrained(settings.ESM2_15B["model"])
    
    print("Model: ", rep["name"])
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), " GPUs.")
            model = torch.nn.DataParallel(model)

    model.half()        
    model.to(device)
    model = model.eval()

    representations = []

    # For each sequence, label and id in the dataframe we take frozen representations from each rep
    for sequence, label, id in zip(sequences["sequence"], sequences["label"], sequences["id"]):
        torch.cuda.empty_cache()
        if rep["name"] == "ProtBERT" or rep["name"] == "ProtBERT-BFD" or rep["name"] == "ProtT5":
            # We put a space between each amino acid
            sequence = " ".join(sequence)
        
        # Replace UZOB with X
        sequence = sequence.replace("U", "X")
        sequence = sequence.replace("Z", "X")
        sequence = sequence.replace("O", "X")
        sequence = sequence.replace("B", "X")

        # Tokenize the sequence
        inputs = tokenizer(sequence, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=1024)
        
        # Obtain frozen representations
        with torch.no_grad():
            outputs = model(**inputs)
            representation = outputs.last_hidden_state[0].detach().cpu().numpy()
            representations.append((representation, label, id))

    # Save the frozen representations
    with h5py.File(settings.FROZEN_REPRESENTATIONS_PATH + "_" + rep["name"] + ".h5", "w") as f:
        for representation, label, id in representations:
            f.create_dataset(str(id), data=representation)
            f[str(id)].attrs["label"] = label



