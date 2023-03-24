import h5py
from settings import settings
import os
import pandas as pd

# We load all frozen and finetuned representations h5 files in the plm_representations folder
frozen_representations = [f for f in os.listdir(settings.PLM_REPRESENTATIONS_PATH) if os.path.isfile(
    os.path.join(settings.PLM_REPRESENTATIONS_PATH, f)) and "frozen" in f]

# We create a dictionary with the frozen representations
frozen_representations_dict = {}
for frozen_representation in frozen_representations:
    frozen_representations_dict[frozen_representation] = h5py.File(
        settings.PLM_REPRESENTATIONS_PATH + frozen_representation, "r")

# For each frozen representation we create datasets with the names in dataset folder with frozen at the end, by mapping the id in h5 file to the id in the csv dataset. So for each new frozen datasets,
# there are frozen representations for each dataset in the dataset folder

# We take the list of datasets in the dataset folder
# We make a list of csv datasets. We add all the files in the datasets folder except all_sequences.csv and sequence_ids_dict.jsn and not the directories
datasets = [f for f in os.listdir(settings.DATASET_PATH) if os.path.isfile(os.path.join(
    settings.DATASET_PATH, f)) and f != "all_sequences.csv" and f != "sequence_ids_dict.jsn"]

print(len(datasets))

# For each dataset and each frozen representation we create a new h5 dataset with the frozen representations, labels and ids
for dataset in datasets:
    for frozen_representation in frozen_representations:
        # We create a new h5 file with the name of the dataset and frozen representation
        new_dataset = h5py.File(
            settings.REPRESENTATIONS_FILTERED_PATH + dataset[:-4] + "_" + frozen_representation + "_full.h5", "w")

        # The saved format is: f.create_dataset(str(id), data=representation) \ f[str(id)].attrs["label"] = label

        # We open the dataset csv file
        df = pd.read_csv(settings.DATASET_PATH + dataset, names=["sequence", "label", "id"], skiprows=1)

        # We map the id in the csv file to the id in the h5 file
        for index, row in df.iterrows():
            # We get the id from the csv file
            csv_id = row["id"]
            # We get the label from the csv file
            label = row["label"]
            # We get the representation from the h5 file where the id is the same
            representation = frozen_representations_dict[frozen_representation][str(csv_id)][:]
            # We save the representation in the new h5 file
            new_dataset.create_dataset(str(csv_id), data=representation)
            # We save the label in the new h5 file
            new_dataset[str(csv_id)].attrs["label"] = label

        # We close the new h5 file
        new_dataset.close()


