import h5py
import os
import torch
import ankh
from settings import settings
import pandas as pd

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load all the sequences
sequences = pd.read_csv(settings.ALL_SEQUENCES_PATH)

representations = []

# For each sequence, label and id in the dataframe we take frozen representations from each rep
for sequence, label, id in zip(sequences["sequence"], sequences["label"], sequences["id"]):

    # To load base model.
    model, tokenizer = ankh.load_base_model()
    model.to(device)
    model.eval()

    outputs = tokenizer.batch_encode_plus(list(sequence), 
                                add_special_tokens=True, 
                                is_split_into_words=True, 
                                return_tensors="pt")
    
    outputs = {k: v.to(device) for k, v in outputs.items()}
    
    # Obtain frozen representations
    with torch.no_grad():
        embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
        representation = embeddings.last_hidden_state[0].detach().cpu().numpy()
        representations.append((representation, label, id))

# Save the frozen representations
with h5py.File(settings.FROZEN_REPRESENTATIONS_PATH + "_ankh.h5", "w") as f:
    for representation, label, id in representations:
        f.create_dataset(str(id), data=representation)
        f[str(id)].attrs["label"] = label


