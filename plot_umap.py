import h5py
from settings import settings
import pandas as pd
from methods.methods import plot_umap_datasets
import numpy as np

datasets = [
    'ionchannels_iontransporters_train_finetuned_representations_ESM-1b_ionchannels_iontransporters_imbalanced.h5',
    'ionchannels_iontransporters_train_finetuned_representations_ESM-2_ionchannels_iontransporters_imbalanced.h5',
    'ionchannels_iontransporters_train_finetuned_representations_full_ESM-1b_ionchannels_iontransporters.h5',
    'ionchannels_iontransporters_train_finetuned_representations_full_ESM-2_ionchannels_iontransporters.h5',
    'ionchannels_iontransporters_train_finetuned_representations_full_ProtBERT-BFD_ionchannels_iontransporters.h5',
    'ionchannels_iontransporters_train_finetuned_representations_full_ProtBERT_ionchannels_iontransporters.h5',
    'ionchannels_iontransporters_train_finetuned_representations_ProtBERT-BFD_ionchannels_iontransporters_imbalanced.h5',
    'ionchannels_iontransporters_train_finetuned_representations_ProtBERT_ionchannels_iontransporters_imbalanced.h5',
    'ionchannels_iontransporters_train_frozen_representations_ESM-1b.h5',
    'ionchannels_iontransporters_train_frozen_representations_ESM-2_15B.h5',
    'ionchannels_iontransporters_train_frozen_representations_ESM-2.h5',
    'ionchannels_iontransporters_train_frozen_representations_full_ESM-1b.h5',
    'ionchannels_iontransporters_train_frozen_representations_full_ESM-2.h5',
    'ionchannels_iontransporters_train_frozen_representations_full_ProtBERT-BFD.h5',
    'ionchannels_iontransporters_train_frozen_representations_full_ProtBERT.h5',
    'ionchannels_iontransporters_train_frozen_representations_ProtBERT-BFD.h5',
    'ionchannels_iontransporters_train_frozen_representations_ProtBERT.h5',
    'ionchannels_iontransporters_train_frozen_representations_ProtT5.h5',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_ESM-1b_ionchannels_membraneproteins_imbalanced.h5',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_ESM-2_ionchannels_membraneproteins_imbalanced.h5',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_full_ESM-1b_ionchannels_membraneproteins_imbalanced.h5',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_full_ESM-2_ionchannels_membraneproteins_imbalanced.h5',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_full_ProtBERT-BFD_ionchannels_membraneproteins_imbalanced.h5',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_full_ProtBERT_ionchannels_membraneproteins_imbalanced.h5',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_ProtBERT-BFD_ionchannels_membraneproteins_imbalanced.h5',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_ProtBERT_ionchannels_membraneproteins_imbalanced.h5',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ESM-1b.h5',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ESM-2_15B.h5',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ESM-2.h5',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_full_ESM-1b.h5',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_full_ESM-2.h5',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_full_ProtBERT-BFD.h5',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_full_ProtBERT.h5',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ProtBERT-BFD.h5',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ProtBERT.h5',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ProtT5.h5',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_ESM-1b_iontransporters_membraneproteins_imbalanced.h5',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_ESM-2_iontransporters_membraneproteins_imbalanced.h5',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_full_ESM-1b_iontransporters_membraneproteins_imbalanced.h5',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_full_ESM-2_iontransporters_membraneproteins_imbalanced.h5',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_full_ProtBERT-BFD_iontransporters_membraneproteins_imbalanced.h5',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_full_ProtBERT_iontransporters_membraneproteins_imbalanced.h5',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_ProtBERT-BFD_iontransporters_membraneproteins_imbalanced.h5',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_ProtBERT_iontransporters_membraneproteins_imbalanced.h5',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ESM-1b.h5',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ESM-2_15B.h5',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ESM-2.h5',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_full_ESM-1b.h5',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_full_ESM-2.h5',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_full_ProtBERT-BFD.h5',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_full_ProtBERT.h5',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ProtBERT-BFD.h5',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ProtBERT.h5',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ProtT5.h5',
]

for dataset in datasets:
    dataset_name = dataset.split("_")[0] + "_" + dataset.split("_")[1]
    # We open the h5 file
    with h5py.File(settings.REPRESENTATIONS_FILTERED_PATH + dataset, "r") as f:
        # We put the id, representation and label together in a list. The saved data is : (str(csv_id), data=representation), [str(csv_id)].attrs["label"] = label. And the representation is a numpy array
        train_data = [(id, representation, label) for id, representation in zip(
            f.keys(), f.values()) for label in f[id].attrs.values()]

        # We convert the representations to a numpy array
        for i in range(len(train_data)):
            train_data[i] = (train_data[i][0], np.array(
                train_data[i][1]), train_data[i][2])

        X_train = []
        y_train = []
        ids = []
        # We separate the id, representation and label in different lists
        for id, rep, label in train_data:
            X_train.append(rep)
            y_train.append(label)
            ids.append(id)

        X_train = [np.mean(np.array(x), axis=0) for x in X_train]
        y_train = np.array(y_train)
        ids = np.array(ids)

        if dataset_name == "ionchannels_membraneproteins":
            # We convert labels to 0 and 1. 0 for ionchannels and 1 for membraneproteins
            y_train = [1 if label ==
                       settings.IONCHANNELS else 0 for label in y_train]
            label_dict = {0: settings.IONCHANNELS,
                          1: settings.MEMBRANEPROTEINS}
        elif dataset_name == "ionchannels_iontransporters":
            # We convert labels to 0 and 1. 0 for ionchannels and 1 for iontransporters
            y_train = [1 if label ==
                       settings.IONCHANNELS else 0 for label in y_train]
            label_dict = {0: settings.IONCHANNELS, 1: settings.IONTRANSPORTERS}
        elif dataset_name == "iontransporters_membraneproteins":
            # We convert labels to 0 and 1. 0 for iontransporters and 1 for membraneproteins
            y_train = [1 if label ==
                       settings.IONTRANSPORTERS else 0 for label in y_train]
            label_dict = {0: settings.IONTRANSPORTERS,
                          1: settings.MEMBRANEPROTEINS}

        # We make a pandas dataframe with the id, representation and label columns
        df = pd.DataFrame(
            {"id": ids, "representation": X_train, "label": y_train})

        # We plot the umap
        plot_umap_datasets([df], label_dict=label_dict, task=dataset_name,
                           filename=dataset.split(".")[0] + "_umap.png")
