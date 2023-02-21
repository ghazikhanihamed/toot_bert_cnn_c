import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import re
from settings import settings


class ProtBertCNNDataset(Dataset):
    def __init__(self, data, is_evolutionary, split, max_length, seqid_dict):

        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        self.split = split
        self.max_length = max_length
        self.is_evolutionary = is_evolutionary
        self.seqid_dict = seqid_dict

        if self.is_evolutionary:
            if self.split == settings.TRAIN:
                self.training, self.phmm_training = data
            elif self.split == settings.VAL:
                self.validation, self.phmm_validation = data
            else:
                self.test, self.phmm_test = data
            self.seqs, self.hmms, self.labels, self.seq_ids = self.load_dataset_ion_non()
            
        else:
            if self.split == settings.TRAIN:
                self.training = data
            elif self.split == settings.VAL:
                self.validation = data
            else:
                self.test = data
            self.seqs, self.labels, self.seq_ids = self.load_dataset_ion_non()
    
    def load_dataset_ion_non(self):

        if self.split == settings.TRAIN:    
            dataset = self.training
            # We make a label dictionary. If it starts with ion then it is 1, otherwise it is 0
            label_dict = {"ion": 1, "non_ion": 0}
            seqs = []
            labels = []
            seq_ids = []
            if self.is_evolutionary:
                hmms = []
            for i in range(len(dataset)):
                seq = dataset.iloc[i, 0]
                label = dataset.iloc[i, 1]
                if self.is_evolutionary:
                    # We filter hmm where the label column is equal to label and we take the first 47 columns
                    hmm = self.phmm_training[self.phmm_training['label'] == label].iloc[:, 0:47].values[0]
                    hmms.append(hmm)
                seqs.append(seq)
                # Append if it staerts with ion or non_ion
                labels.append(label_dict[label.split("|")[0]])
                # We add the seq_id to the seq_ids list
                seq_ids.append(self.seqid_dict[label])

        elif self.split == settings.VAL:
            dataset = self.validation
            label_dict = {"ion": 1, "non_ion": 0}
            seqs = []
            labels = []
            seq_ids = []
            if self.is_evolutionary:
                hmms = []
            for i in range(len(dataset)):
                seq = dataset.iloc[i, 0]
                label = dataset.iloc[i, 1]
                if self.is_evolutionary:
                    # We filter hmm where the label column is equal to label and we take the first 47 columns
                    hmm = self.phmm_validation[self.phmm_validation['label'] == label].iloc[:, 0:47].values[0]
                    hmms.append(hmm)
                seqs.append(seq)
                # Append if it staerts with ion or non_ion
                labels.append(label_dict[label.split("|")[0]])
                seq_ids.append(self.seqid_dict[label])

        else:
            dataset = self.test
            label_dict = {"ionchannel": 1, "non_ion": 0}
            seqs = []
            labels = []
            seq_ids = []
            if self.is_evolutionary:
                hmms = []
            for i in range(len(dataset)):
                seq = dataset.iloc[i, 0]
                label = dataset.iloc[i, 1]
                if self.is_evolutionary:
                    # We filter hmm where the label column is equal to label and we take the first 47 columns
                    hmm = self.phmm_test[self.phmm_test['label'] == label].iloc[:, 0:47].values[0]
                    hmms.append(hmm)
                seqs.append(seq)
                # Append if it staerts with ion or non_ion
                labels.append(label_dict[label.split("|")[1].split("-")[0]])
                seq_ids.append(self.seqid_dict[label])

        if self.is_evolutionary:
            return seqs, hmms, labels, seq_ids        
        return seqs, labels, seq_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, truncation=True, max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}

        if self.is_evolutionary:
            sample['pHMMs'] = torch.tensor(self.hmms[idx])
        
        sample['labels'] = torch.tensor(self.labels[idx])

        sample['seq_ids'] = torch.tensor(self.seq_ids[idx])

        return sample