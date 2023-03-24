import h5py
import os
import torch
from transformers import BertModel, BertTokenizer
from transformers import EsmModel, EsmTokenizer
from settings import settings
import pandas as pd

torch.cuda.empty_cache()

# Load all the sequences
sequences = pd.read_csv(settings.ALL_SEQUENCES_PATH)

# We take all the folders in finetuned_models and not the files
finetuned_models = [f for f in os.listdir(settings.FINETUNED_MODELS_PATH) if os.path.isdir(os.path.join(settings.FINETUNED_MODELS_PATH, f))]

for finetuned_model in finetuned_models:

    model_name = finetuned_model.split("_")[0]

    # We load the model and the tokenizer
    if model_name == settings.PROTBERT["name"]:
        model = BertModel.from_pretrained(settings.FINETUNED_MODELS_PATH + finetuned_model)
        tokenizer = BertTokenizer.from_pretrained(settings.PROTBERT["model"], do_lower_case=False)
    elif model_name == settings.PROTBERTBFD["name"]:
        model = BertModel.from_pretrained(settings.FINETUNED_MODELS_PATH + finetuned_model)
        tokenizer = BertTokenizer.from_pretrained(settings.PROTBERTBFD["model"], do_lower_case=False)
    elif model_name == settings.ESM1B["name"]:
        model = EsmModel.from_pretrained(settings.FINETUNED_MODELS_PATH + finetuned_model)
        tokenizer = EsmTokenizer.from_pretrained(settings.ESM1B["model"], do_lower_case=False)
    elif model_name == settings.ESM2["name"]:
        model = EsmModel.from_pretrained(settings.FINETUNED_MODELS_PATH + finetuned_model)
        tokenizer = EsmTokenizer.from_pretrained(settings.ESM2["model"], do_lower_case=False)

    print("Finetuned model: ", finetuned_model)
    print("-"*50)
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), " GPUs.")
            model = torch.nn.DataParallel(model)

    # model.half()        
    model.to(device)
    model = model.eval()

    representations = []

    # For each sequence, label and id in the dataframe we take frozen representations from each rep
    for sequence, label, id in zip(sequences["sequence"], sequences["label"], sequences["id"]):
        if model_name == settings.PROTBERT["name"] or model_name == settings.PROTBERTBFD["name"]:
            # We put a space between each amino acid
            sequence = " ".join(sequence)
        
        # Replace UZOB with X
        sequence = sequence.replace("U", "X")
        sequence = sequence.replace("Z", "X")
        sequence = sequence.replace("O", "X")
        sequence = sequence.replace("B", "X")

        # Tokenize the sequence
        if model_name == settings.PROTBERT["name"] or model_name == settings.PROTBERTBFD["name"]:
            inputs = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
        else:
            inputs = tokenizer(sequence, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=1024)
        
        # Move the inputs to the device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Obtain frozen representations
        with torch.no_grad():
            outputs = model(**inputs)
            representation = outputs.last_hidden_state[0].detach().cpu().numpy()
            representations.append((representation, label, id))

    # Save the frozen representations
    with h5py.File(settings.PLM_REPRESENTATIONS_PATH + "full" + "_" + finetuned_model + ".h5", "w") as f:
        for representation, label, id in representations:
            f.create_dataset(str(id), data=representation)
            f[str(id)].attrs["label"] = label
