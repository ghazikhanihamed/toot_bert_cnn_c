import torch.nn as nn
from settings import settings
from transformers import BertModel, BertTokenizer
from transformers import EsmModel, EsmTokenizer
from transformers import T5EncoderModel, T5Tokenizer

class ProtBertClassifier(nn.Module):
    def __init__(self, task):
        super(ProtBertClassifier, self).__init__()

        self.max_length = task["max_length"]
        self.representation = task["representation"]

        self.protbert = BertModel.from_pretrained(settings.PROTBERT["model"])

        if self.representation == settings.FROZEN:
            for param in self.protbert.parameters():
                param.requires_grad = False
        else:
            classifier = nn.Linear(self.protbert.config.hidden_size, 2)
            for param in self.protbert.parameters():
                param.requires_grad = True

    
    def forward(self, input_ids, attention_mask):
        outputs = self.protbert(input_ids, attention_mask=attention_mask).last_hidden_state
        
        return outputs

class ProtBertBFDClassifier(nn.Module):
    def __init__(self, task):
        super(ProtBertBFDClassifier, self).__init__()

        self.max_length = task["max_length"]
        self.representation = task["representation"]

        self.protbertbfd = BertModel.from_pretrained(settings.PROTBERTBFD["model"])

        if self.representation == settings.FROZEN:
            for param in self.protbertbfd.parameters():
                param.requires_grad = False
        else:
            for param in self.protbertbfd.parameters():
                param.requires_grad = True

    def forward(self, x):
        return x
    
class ProtT5Classifier(nn.Module):
    def __init__(self, task):
        super(ProtT5Classifier, self).__init__()

        self.max_length = task["max_length"]
        self.representation = task["representation"]

        self.prott5 = T5EncoderModel.from_pretrained(settings.PROTT5["model"])

        for param in self.prott5.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return x
    
class Esm1bClassifier(nn.Module):
    def __init__(self, task):
        super(Esm1bClassifier, self).__init__()

        self.max_length = task["max_length"]
        self.representation = task["representation"]

        self.esm1b = EsmModel.from_pretrained(settings.ESM1B["model"])

        if self.representation == settings.FROZEN:
            for param in self.esm1b.parameters():
                param.requires_grad = False
        else:
            for param in self.esm1b.parameters():
                param.requires_grad = True

    def forward(self, x):
        return x
    
class Esm2TClassifier(nn.Module):
    def __init__(self, task):
        super(Esm2TClassifier, self).__init__()

        self.max_length = task["max_length"]
        self.representation = task["representation"]

        self.esm2 = EsmModel.from_pretrained(settings.ESM2T["model"])

    def forward(self, x):
        return x

class Esm2_15BClassifier(nn.Module):
    def __init__(self, task):
        super(Esm2_15BClassifier, self).__init__()

        self.max_length = task["max_length"]
        self.representation = task["representation"]

        self.esm2_15b = EsmModel.from_pretrained(settings.ESM2_15B["model"])

    def forward(self, x):
        return x