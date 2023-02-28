import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef
import re
from settings import settings


class PLMDataset(Dataset):
    def __init__(self, model, dataset, seqid_dict):
        self.model = model
        self.datasetFolderPath = settings.DATASET_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(model["model"], do_lower_case=False)
        self.seqid_dict = seqid_dict

        self.seqs, self.labels, self.seq_ids = self.load_dataset(dataset)

        self.max_length = 1024
    
    def load_dataset(self, dataset):
        trainFilePath = os.path.join(self.datasetFolderPath, dataset)
        df = pd.read_csv(trainFilePath, names=["sequence", "label", "id"], skiprows=1)
        # We extract the labels from the file name. The first and the second part split by _ are the labels. If the second part is "membraneproteins" then the label is "membrane_proteins".
        labels = [dataset.split("_")[0], dataset.split("_")[1] if dataset.split("_")[1] != "membraneproteins" else "membrane_proteins"]
        df["labels"] = np.where(df["label"] == labels[0], 1, 0)
        df["ids"] = df["id"].map(self.seqid_dict)

        seq = list(df["sequence"])
        label = list(df["labels"])
        seq_id = list(df["ids"])

        assert len(seq) == len(label) == len(seq_id)

        return seq, label, seq_id


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.model["name"] == "ProtBERT" or self.model["name"] == "ProtBERT-BFD":
            # We put a space between each amino acid
            seq = " ".join("".join(self.seqs[idx].split()))
        else:
            seq = "".join(self.seqs[idx].split())
        
        # Replace UZOB with X
        seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}

        sample["labels"] = torch.tensor(self.labels[idx])

        sample["seq_id"] = torch.tensor(self.seq_ids[idx])

        return sample
