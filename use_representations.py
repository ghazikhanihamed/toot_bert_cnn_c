import h5py
import pandas as pd
from settings import settings
import os


# we make a list of only h5 files in the representations folder
representations = [f for f in os.listdir(settings.REPRESENTATIONS_PATH) if os.path.isfile(os.path.join(settings.REPRESENTATIONS_PATH, f)) and f.endswith(".h5")]

# For each representation we take id, representation and label
for representation in representations:
    # We separate the information from the name of the representation
    # We get the name of the dataset which is the two first words in the name of the representation separated by _
    dataset_name = representation.split("_")[0] + "_" + representation.split("_")[1]
    # We get the name of the type of dataset which is the third word in the name of the representation separated by _
    dataset_type = representation.split("_")[2]
    # We get the split of the dataset which is the fourth word in the name of the representation separated by _
    dataset_split = representation.split("_")[3]
    # We get the number of the dataset if the type is "balanced" which is the fifth word in the name of the representation separated by _
    if dataset_type == "balanced":
        dataset_number = representation.split("_")[4]
        # We get the type of the representations which is the sixth word in the name of the representation separated by _
        representation_type = representation.split("_")[5]
        # And we get the name of the model which is the eighth word in the name of the representation separated by _ + 9th word if exists without the .h5
        if len(representation.split("_")) == 9:
            model_name = representation.split("_")[7] + "_" + representation.split("_")[8][:-3]
        else:
            model_name = representation.split("_")[7][:-3]
    else:
        # We get the type of the representations which is the fifth word in the name of the representation separated by _
        representation_type = representation.split("_")[4]
        # And we get the name of the model which is the seventh word in the name of the representation separated by _
        if len(representation.split("_")) == 8:
            model_name = representation.split("_")[6] + "_" + representation.split("_")[7][:-3]
        else:
            model_name = representation.split("_")[6][:-3]
    # We open the h5 file
    with h5py.File(settings.REPRESENTATIONS_PATH + representation, "r") as f:
        # We get the ids
        ids = list(f.keys())
        # We get the labels
        labels = [f[id].attrs["label"] for id in ids]
        # We get the representations
        representations = [f[id][:] for id in ids]

        aa=1