import h5py
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from transformers import EsmModel, EsmTokenizer
from transformers import T5EncoderModel, T5Tokenizer
from settings import settings
import pandas as pd
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

# init the distributed world with world_size 1
url = "tcp://localhost:23456"
torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

fsdp_params = dict(mixed_precision=True, flatten_parameters=True, state_dict_device=torch.device("cpu"), cpu_offload=True)

# Load all the sequences
sequences = pd.read_csv(settings.ALL_SEQUENCES_PATH)

for rep in settings.REPRESENTATIONS:

    if rep["name"] == "ProtBERT" or rep["name"] == "ProtBERT-BFD" or rep["name"] == "ProtT5" or rep["name"] == "ESM-1b" or rep["name"] == "ESM-2":
        continue

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
        with enable_wrap(wrapper_cls=FSDP,**fsdp_params):
            print("here")
            model = EsmModel.from_pretrained(settings.ESM2_15B["model"])
            tokenizer = EsmTokenizer.from_pretrained(settings.ESM2_15B["model"], do_lower_case=False)

            #model = model.to(device)
            model = model.eval()

            for name, child in model.named_children():
                if name == "layers":
                    for layer_name, layer in child.named_children():
                        wrapped_layer = wrap(layer)
                        setattr(child, layer_name, wrapped_layer)

            model = wrap(model)

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
        inputs = tokenizer.encode_plus(sequence, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Obtain frozen representations
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            representation = outputs.last_hidden_state[0].detach().cpu().numpy()
            representations.append((representation, label, id))

    # Save the frozen representations
    with h5py.File(settings.FROZEN_REPRESENTATIONS_PATH + "_" + rep["name"] + ".h5", "w") as f:
        for representation, label, id in representations:
            f.create_dataset(str(id), data=representation)
            f[str(id)].attrs["label"] = label

