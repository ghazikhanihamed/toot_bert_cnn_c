from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
from settings import settings
import random
import numpy as np
import os
import json

# We set the seed for reproducibility
random.seed(settings.SEED)
np.random.seed(settings.SEED)

# Read all the fasta files in the train and test folders
ionchannels_train_dir = settings.DATASET_PATH + \
    settings.IONCHANNELS + "/" + settings.TRAIN + "/"
ionchannels_train_fasta_files = [
    ionchannels_train_dir + file for file in os.listdir(ionchannels_train_dir)]

ionchannels_test_dir = settings.DATASET_PATH + \
    settings.IONCHANNELS + "/" + settings.TEST + "/"
ionchannels_test_fasta_files = [
    ionchannels_test_dir + file for file in os.listdir(ionchannels_test_dir)]

# Read all the fasta files in the train and test folders
iontransporters_train_dir = settings.DATASET_PATH + \
    settings.IONTRANSPORTERS + "/" + settings.TRAIN + "/"
iontransporters_train_fasta_files = [
    iontransporters_train_dir + file for file in os.listdir(iontransporters_train_dir)]

iontransporters_test_dir = settings.DATASET_PATH + \
    settings.IONTRANSPORTERS + "/" + settings.TEST + "/"
iontransporters_test_fasta_files = [
    iontransporters_test_dir + file for file in os.listdir(iontransporters_test_dir)]

# Read all the fasta files in the train and test folders
membrane_proteins_train_dir = settings.DATASET_PATH + \
    settings.MEMBRANE_PROTEINS + "/" + settings.TRAIN + "/"
membrane_proteins_train_fasta_files = [
    membrane_proteins_train_dir + file for file in os.listdir(membrane_proteins_train_dir)]

membrane_proteins_test_dir = settings.DATASET_PATH + \
    settings.MEMBRANE_PROTEINS + "/" + settings.TEST + "/"
membrane_proteins_test_fasta_files = [
    membrane_proteins_test_dir + file for file in os.listdir(membrane_proteins_test_dir)]

# Read the fasta files and create a dataframe with the sequences and the labels and sequences ids
ionchannels_train_df = pd.DataFrame(columns=["sequence", "label", "id"])
for file in ionchannels_train_fasta_files:
    for record in SeqIO.parse(file, "fasta"):
        ionchannels_train_df = ionchannels_train_df.append({"sequence": str(
            record.seq), "label": settings.IONCHANNELS, "id": record.id}, ignore_index=True)

ionchannels_test_df = pd.DataFrame(columns=["sequence", "label", "id"])
for file in ionchannels_test_fasta_files:
    for record in SeqIO.parse(file, "fasta"):
        ionchannels_test_df = ionchannels_test_df.append({"sequence": str(
            record.seq), "label": settings.IONCHANNELS, "id": record.id}, ignore_index=True)

iontransporters_train_df = pd.DataFrame(columns=["sequence", "label", "id"])
for file in iontransporters_train_fasta_files:
    for record in SeqIO.parse(file, "fasta"):
        iontransporters_train_df = iontransporters_train_df.append({"sequence": str(
            record.seq), "label": settings.IONTRANSPORTERS, "id": record.id}, ignore_index=True)

iontransporters_test_df = pd.DataFrame(columns=["sequence", "label", "id"])
for file in iontransporters_test_fasta_files:
    for record in SeqIO.parse(file, "fasta"):
        iontransporters_test_df = iontransporters_test_df.append({"sequence": str(
            record.seq), "label": settings.IONTRANSPORTERS, "id": record.id}, ignore_index=True)

membrane_proteins_train_df = pd.DataFrame(columns=["sequence", "label", "id"])
for file in membrane_proteins_train_fasta_files:
    for record in SeqIO.parse(file, "fasta"):
        membrane_proteins_train_df = membrane_proteins_train_df.append({"sequence": str(
            record.seq), "label": settings.MEMBRANE_PROTEINS, "id": record.id}, ignore_index=True)

membrane_proteins_test_df = pd.DataFrame(columns=["sequence", "label", "id"])
for file in membrane_proteins_test_fasta_files:
    for record in SeqIO.parse(file, "fasta"):
        membrane_proteins_test_df = membrane_proteins_test_df.append({"sequence": str(
            record.seq), "label": settings.MEMBRANE_PROTEINS, "id": record.id}, ignore_index=True)

# We make a list of all the sequence ids and map them to a unique id and save the mapping in a dictionary as a json file
sequence_ids = list(ionchannels_train_df["id"]) + list(ionchannels_test_df["id"]) + list(iontransporters_train_df["id"]) + list(
    iontransporters_test_df["id"]) + list(membrane_proteins_train_df["id"]) + list(membrane_proteins_test_df["id"])

sequence_ids = list(sequence_ids)

sequence_ids_dict = {}
for i in range(len(sequence_ids)):
    sequence_ids_dict[sequence_ids[i]] = i

with open(settings.SEQUENCE_IDS_DICT_PATH, "w") as f:
    json.dump(sequence_ids_dict, f)


# Concatenate the dataframes ionchannels_train_df and membrane_proteins_train_df
ionchannel_membrane_train_df = pd.concat(
    [ionchannels_train_df, membrane_proteins_train_df], ignore_index=True)
ionchannel_membrane_test_df = pd.concat(
    [ionchannels_test_df, membrane_proteins_test_df], ignore_index=True)

# Concatenate the dataframes ionchannels_train_df and membrane_proteins_train_df
iontransporter_membrane_train_df = pd.concat(
    [iontransporters_train_df, membrane_proteins_train_df], ignore_index=True)
iontransporter_membrane_test_df = pd.concat(
    [iontransporters_test_df, membrane_proteins_test_df], ignore_index=True)

# Concatenate the dataframes ionchannels_train_df and iontransporters_train_df
ionchannel_iontransporter_train_df = pd.concat(
    [ionchannels_train_df, iontransporters_train_df], ignore_index=True)
ionchannel_iontransporter_test_df = pd.concat(
    [ionchannels_test_df, iontransporters_test_df], ignore_index=True)

# We save the dataframes in csv files
ionchannel_membrane_train_df.to_csv(settings.DATASET_PATH + settings.IONCHANNELS_MEMBRANEPROTEINS +
                                    "_" + "imbalanced" + "_" + settings.TRAIN + ".csv", index=False)
ionchannel_membrane_test_df.to_csv(settings.DATASET_PATH + settings.IONCHANNELS_MEMBRANEPROTEINS +
                                   "_" + "imbalanced" + "_" + settings.TEST + ".csv", index=False)

iontransporter_membrane_train_df.to_csv(settings.DATASET_PATH + settings.IONTRANSPORTERS_MEMBRANEPROTEINS +
                                        "_" + "imbalanced" + "_" + settings.TRAIN + ".csv", index=False)
iontransporter_membrane_test_df.to_csv(settings.DATASET_PATH + settings.IONTRANSPORTERS_MEMBRANEPROTEINS +
                                       "_" + "imbalanced" + "_" + settings.TEST + ".csv", index=False)

ionchannel_iontransporter_train_df.to_csv(
    settings.DATASET_PATH + settings.IONCHANNELS_IONTRANSPORTERS + "_" + settings.TRAIN + ".csv", index=False)
ionchannel_iontransporter_test_df.to_csv(
    settings.DATASET_PATH + settings.IONCHANNELS_IONTRANSPORTERS + "_" + settings.TEST + ".csv", index=False)

# We randomly take 280 sequences from membrane_proteins_train_df and 70 sequences from membrane_proteins_test_df to balance the dataset with 10 different random states
random_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for random_state in random_states:
    membrane_proteins_train_balanced_df = membrane_proteins_train_df.sample(
        n=280, random_state=random_state)
    membrane_proteins_test_balanced_df = membrane_proteins_test_df.sample(
        n=70, random_state=random_state)

    # We concatenate the dataframes ionchannels_train_df and membrane_proteins_train_balanced_df
    ionchannel_membrane_train_balanced_df = pd.concat(
        [ionchannels_train_df, membrane_proteins_train_balanced_df], ignore_index=True)
    ionchannel_membrane_test_balanced_df = pd.concat(
        [ionchannels_test_df, membrane_proteins_test_balanced_df], ignore_index=True)

    # We concatenate the dataframes iontransporters_train_df and membrane_proteins_train_balanced_df
    iontransporter_membrane_train_balanced_df = pd.concat(
        [iontransporters_train_df, membrane_proteins_train_balanced_df], ignore_index=True)
    iontransporter_membrane_test_balanced_df = pd.concat(
        [iontransporters_test_df, membrane_proteins_test_balanced_df], ignore_index=True)

    # We save the dataframes in csv files
    ionchannel_membrane_train_balanced_df.to_csv(settings.DATASET_PATH + settings.IONCHANNELS_MEMBRANEPROTEINS +
                                                 "_" + "balanced" + "_" + settings.TRAIN + "_" + str(random_state) + ".csv", index=False)
    ionchannel_membrane_test_balanced_df.to_csv(settings.DATASET_PATH + settings.IONCHANNELS_MEMBRANEPROTEINS +
                                                "_" + "balanced" + "_" + settings.TEST + "_" + str(random_state) + ".csv", index=False)

    iontransporter_membrane_train_balanced_df.to_csv(settings.DATASET_PATH + settings.IONTRANSPORTERS_MEMBRANEPROTEINS +
                                                     "_" + "balanced" + "_" + settings.TRAIN + "_" + str(random_state) + ".csv", index=False)
    iontransporter_membrane_test_balanced_df.to_csv(settings.DATASET_PATH + settings.IONTRANSPORTERS_MEMBRANEPROTEINS +
                                                    "_" + "balanced" + "_" + settings.TEST + "_" + str(random_state) + ".csv", index=False)
