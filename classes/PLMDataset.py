import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import re
from settings import settings
from torch.utils.data.dataloader import default_collate
from scipy import sparse


class PLMDataset(Dataset):
    def __init__(self, model, dataset, seqid_dict):
        self.model = model
        self.datasetFolderPath = settings.DATASET_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(
            model["model"], do_lower_case=False)
        self.seqid_dict = seqid_dict

        self.seqs, self.labels = self.load_dataset(dataset)

        self.max_length = 1024

    def load_dataset(self, dataset):
        trainFilePath = os.path.join(self.datasetFolderPath, dataset)
        df = pd.read_csv(trainFilePath, names=[
                         "sequence", "label", "id"], skiprows=1)
        # We extract the labels from the file name. The first and the second part split by _ are the labels. If the second part is "membraneproteins" then the label is "membrane_proteins".
        labels = [dataset.split("_")[0], dataset.split("_")[1] if dataset.split("_")[
            1] != "membraneproteins" else "membrane_proteins"]
        df["labels"] = np.where(df["label"] == labels[0], 1, 0)

        seq = list(df["sequence"])
        label = list(df["labels"])

        assert len(seq) == len(label)

        return seq, label

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

        seq_ids = self.tokenizer(seq, truncation=True,
                                 max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}

        sample["labels"] = torch.tensor(self.labels[idx])

        return sample


class GridDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def transform(self, X, y):
        # pylint: disable=anomalous-backslash-in-string
        """Additional transformations on ``X`` and ``y``.
        By default, they are cast to PyTorch :class:`~torch.Tensor`\s.
        Override this if you want a different behavior.
        Note: If you use this in conjuction with PyTorch
        :class:`~torch.utils.data.DataLoader`, the latter will call
        the dataset for each row separately, which means that the
        incoming ``X`` and ``y`` each are single rows.
        """
        # pytorch DataLoader cannot deal with None so we use 0 as a
        # placeholder value. We only return a Tensor with one value
        # (as opposed to ``batchsz`` values) since the pytorch
        # DataLoader calls __getitem__ for each row in the batch
        # anyway, which results in a dummy ``y`` value for each row in
        # the batch.
        y = torch.Tensor([0]) if y is None else y

        # pytorch cannot convert sparse matrices, for now just make it
        # dense; squeeze because X[i].shape is (1, n) for csr matrices
        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        return X, y

    def __getitem__(self, i):
        Xi = self.X[i]
        yi = self.y[i]
        return self.transform(Xi, yi)


class SliceDatasetX(Dataset):
    """Helper class that wraps a torch dataset to make it work with sklearn"""

    def __init__(self, dataset, collate_fn=default_collate):
        self.dataset = dataset
        self.collate_fn = collate_fn

        self._indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)
    
    def transform(self, X, y):
        # pylint: disable=anomalous-backslash-in-string
        """Additional transformations on ``X`` and ``y``.
        By default, they are cast to PyTorch :class:`~torch.Tensor`\s.
        Override this if you want a different behavior.
        Note: If you use this in conjuction with PyTorch
        :class:`~torch.utils.data.DataLoader`, the latter will call
        the dataset for each row separately, which means that the
        incoming ``X`` and ``y`` each are single rows.
        """
        # pytorch DataLoader cannot deal with None so we use 0 as a
        # placeholder value. We only return a Tensor with one value
        # (as opposed to ``batchsz`` values) since the pytorch
        # DataLoader calls __getitem__ for each row in the batch
        # anyway, which results in a dummy ``y`` value for each row in
        # the batch.
        y = torch.Tensor([0]) if y is None else y

        # pytorch cannot convert sparse matrices, for now just make it
        # dense; squeeze because X[i].shape is (1, n) for csr matrices
        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        return X, y

    @property
    def shape(self):
        return len(self),

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            Xb = self.transform(*self.dataset[i])[0]
            return Xb

        if isinstance(i, slice):
            i = self._indices[i]

        Xb = self.collate_fn([self.transform(*self.dataset[j])[0] for j in i])
        return Xb
