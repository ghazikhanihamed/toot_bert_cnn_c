from settings import settings
from classes.PLMClassifier import PLMClassifier
from classes.PLMDataset import PLMDataset
import torch
import torch.nn as nn
import random
import numpy as np
import json
from transformers import AdamW
from methods.methods import train, evaluate, test

# We set the seed for reproducibility
random.seed(settings.SEED)
np.random.seed(settings.SEED)
torch.manual_seed(settings.SEED)
torch.cuda.manual_seed_all(settings.SEED)

dataset_path = settings.DATASET_PATH
# load the json dictionary of mappings
with open(settings.SEQUENCE_IDS_DICT_PATH, 'r') as f:
    dataset = json.load(f)

plms = settings.REPRESENTATIONS
# classifiers = [LR(), RF(), KNN(), SVM(), FFNN(), CNN()]
datasets = ["balanced", "imbalanced"]
representations = ["frozen", "fine-tuned"]

# for i, plm in enumerate(plms):
#     for j, dataset in enumerate(datasets):
#         for k, representation in enumerate(representations):
#             for l, classifier in enumerate(classifiers):
#                 print("PLM: ", plm, "Dataset: ", dataset, "Representation: ", representation, "Classifier: ", classifier)
#                 # We create the dataset
#                 dataset = PLMDataset(dataset_path, dataset, plm, representation)
#                 # We create the classifier
#                 classifier = PLMClassifier(classifier, plm, representation, dataset)
#                 # We train the classifier
#                 train(classifier, dataset)
#                 # We evaluate the classifier
#                 evaluate(classifier, dataset)
#                 # We test the classifier
#                 test(classifier, dataset)

