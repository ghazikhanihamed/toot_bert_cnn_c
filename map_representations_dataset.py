import h5py
from settings import settings
import os
import pandas as pd

# We load all frozen and finetuned representations h5 files in the plm_representations folder
frozen_representations = [f for f in os.listdir(settings.PLM_REPRESENTATIONS_PATH) if os.path.isfile(
    os.path.join(settings.PLM_REPRESENTATIONS_PATH, f)) and "frozen" in f]
finetuned_representations = [f for f in os.listdir(settings.PLM_REPRESENTATIONS_PATH) if os.path.isfile(
    os.path.join(settings.PLM_REPRESENTATIONS_PATH, f)) and "finetuned" in f]

# We create a dictionary with the frozen representations
frozen_representations_dict = {}
for frozen_representation in frozen_representations:
    frozen_representations_dict[frozen_representation] = h5py.File(
        settings.PLM_REPRESENTATIONS_PATH + frozen_representation, "r")

# We create a dictionary with the finetuned representations
finetuned_representations_dict = {}
for finetuned_representation in finetuned_representations:
    finetuned_representations_dict[finetuned_representation] = h5py.File(
        settings.PLM_REPRESENTATIONS_PATH + finetuned_representation, "r")

# For each frozen representation we create datasets with the names in dataset folder with frozen at the end, by mapping the id in h5 file to the id in the csv dataset. So for each new frozen datasets,
# there are frozen representations for each dataset in the dataset folder

# We take the list of datasets in the dataset folder
# We make a list of csv datasets. We add all the files in the datasets folder except all_sequences.csv and sequence_ids_dict.jsn and not the directories
datasets = [f for f in os.listdir(settings.DATASET_PATH) if os.path.isfile(os.path.join(
    settings.DATASET_PATH, f)) and f != "all_sequences.csv" and f != "sequence_ids_dict.jsn"]

# For each dataset and each frozen representation we create a new h5 dataset with the frozen representations, labels and ids
for dataset in datasets:
    for frozen_representation in frozen_representations:
        # We create a new h5 file with the name of the dataset and frozen representation
        new_dataset = h5py.File(
            settings.PLM_REPRESENTATIONS_PATH + dataset[:-4] + "_" + frozen_representation, "w")

        # The saved format is: f.create_dataset(str(id), data=representation) \ f[str(id)].attrs["label"] = label

        # We open the dataset csv file
        df = pd.read_csv(settings.DATASET_PATH + dataset, names=["sequence", "label", "id"], skiprows=1)

        # We map the id in the csv file to the id in the h5 file
        for index, row in df.iterrows():
            # We get the id from the csv file
            csv_id = row["id"]
            # We get the label from the csv file
            label = row["label"]
            # We get the id from the h5 file
            h5_id = frozen_representations_dict[frozen_representation][str(csv_id)].attrs["id"]

            # We create a new dataset with the id from the h5 file
            new_dataset.create_dataset(str(h5_id), data=frozen_representations_dict[frozen_representation][str(csv_id)])
            # We add the label to the new dataset
            new_dataset[str(h5_id)].attrs["label"] = label

        # We close the new dataset
        new_dataset.close()

# For each finetuned representation we create datasets with the names in dataset folder with finetuned at the end, by mapping the id in h5 file to the id in the csv dataset. So for each new finetuned datasets,
# there are finetuned representations for each dataset in the dataset folder

# For each dataset and each finetuned representation we create a new h5 dataset with the finetuned representations, labels and ids
for dataset in datasets:
    for finetuned_representation in finetuned_representations:
        # We create a new h5 file with the name of the dataset and finetuned representation
        new_dataset = h5py.File(
            settings.PLM_REPRESENTATIONS_PATH + dataset[:-4] + "_" + finetuned_representation, "w")

        # The saved format is: f.create_dataset(str(id), data=representation) \ f[str(id)].attrs["label"] = label

        # We open the dataset csv file
        df = pd.read_csv(settings.DATASET_PATH + dataset, names=["sequence", "label", "id"], skiprows=1)

        # We map the id in the csv file to the id in the h5 file
        for index, row in df.iterrows():
            # We get the id from the csv file
            csv_id = row["id"]
            # We get the label from the csv file
            label = row["label"]
            # We get the id from the h5 file
            h5_id = finetuned_representations_dict[finetuned_representation][str(csv_id)].attrs["id"]

            # We create a new dataset with the id from the h5 file
            new_dataset.create_dataset(str(h5_id), data=finetuned_representations_dict[finetuned_representation][str(csv_id)])
            # We add the label to the new dataset
            new_dataset[str(h5_id)].attrs["label"] = label

        # We close the new dataset
        new_dataset.close()

# We close all the frozen representations
for frozen_representation in frozen_representations:
    frozen_representations_dict[frozen_representation].close()

# We close all the finetuned representations
for finetuned_representation in finetuned_representations:
    finetuned_representations_dict[finetuned_representation].close()

# Path: map_representations_dataset.py
