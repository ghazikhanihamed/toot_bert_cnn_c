import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import re
from settings import settings


class PLMDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, truncation=True, max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        
        sample['labels'] = torch.tensor(self.labels[idx])

        sample['seq_ids'] = torch.tensor(self.seq_ids[idx])

        return sample