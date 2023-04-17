
import pandas as pd
import matplotlib.pyplot as plt
from settings import settings
import random
import numpy as np
import seaborn as sns

# We set the seed for reproducibility
random.seed(settings.SEED)
np.random.seed(settings.SEED)

# Read the csv files
ionchannel_membrane_train_df = pd.read_csv(
    settings.DATASET_PATH + settings.IONCHANNELS_MEMBRANEPROTEINS + "_" + "imbalanced" + "_" + settings.TRAIN + ".csv")
ionchannel_membrane_test_df = pd.read_csv(
    settings.DATASET_PATH + settings.IONCHANNELS_MEMBRANEPROTEINS + "_" + "imbalanced" + "_" + settings.TEST + ".csv")
ionchannel_transporter_train_df = pd.read_csv(
    settings.DATASET_PATH + settings.IONCHANNELS_IONTRANSPORTERS + "_" + settings.TRAIN + ".csv")
ionchannel_transporter_test_df = pd.read_csv(
    settings.DATASET_PATH + settings.IONCHANNELS_IONTRANSPORTERS + "_" + settings.TEST + ".csv")
iontransporter_membrane_train_df = pd.read_csv(
    settings.DATASET_PATH + settings.IONTRANSPORTERS_MEMBRANEPROTEINS + "_" + "imbalanced" + "_" + settings.TRAIN + ".csv")
iontransporter_membrane_test_df = pd.read_csv(
    settings.DATASET_PATH + settings.IONTRANSPORTERS_MEMBRANEPROTEINS + "_" + "imbalanced" + "_" + settings.TEST + ".csv")

# We count the number of sequences in each class for each dataset
ionchannels_train = ionchannel_membrane_train_df[ionchannel_membrane_train_df["label"]
                                                 == settings.IONCHANNELS].shape[0]
ionchannels_test = ionchannel_membrane_test_df[ionchannel_membrane_test_df["label"]
                                               == settings.IONCHANNELS].shape[0]
iontransporters_train = ionchannel_transporter_train_df[
    ionchannel_transporter_train_df["label"] == settings.IONTRANSPORTERS].shape[0]
iontransporters_test = ionchannel_transporter_test_df[
    ionchannel_transporter_test_df["label"] == settings.IONTRANSPORTERS].shape[0]
membrane_train = ionchannel_membrane_train_df[ionchannel_membrane_train_df["label"]
                                              == "membrane_proteins"].shape[0]
membrane_test = ionchannel_membrane_test_df[ionchannel_membrane_test_df["label"]
                                            == "membrane_proteins"].shape[0]

# We create bar plots for each dataset. Train test beside each other for each class
bar_width = 0.35
ind = [0, 1, 2]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(ind, [ionchannels_train, iontransporters_train,
       membrane_train], bar_width, color='b', label='Train')
ax.bar([i + bar_width for i in ind], [ionchannels_test,
       iontransporters_test, membrane_test], bar_width, color='r', label='Test')

ax.set_ylabel('Number of sequences')
# ax.set_title('Ion channels, membrane proteins and ion transporters')
ax.set_xticks([i + bar_width * 0.5 for i in ind])
ax.set_xticklabels(('Ion channels', 'Ion transporters', 'Other MPs'))
# ax.set_yticks(np.arange(0, 4000, 100))
ax.legend(loc='best')

# Set y-axis limit
ax.set_ylim(0, 1000)

# Loop through each bar in the plot
for p in ax.patches:
    # Get the bar's x and y coordinates and its height
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    height = p.get_height()
    # Write the value of the bar on top of it
    if height > 1000:
        continue
    ax.text(x, y, height, ha='center', va='bottom')

# plt.show()

plt.savefig(settings.PLOT_PATH + "ionchannels_iontransporters_membrane_imbalanced2.png",
            dpi=300, bbox_inches='tight')
plt.close()

# Read the csv files for the balanced datasets
ionchannel_membrane_train_df = pd.read_csv(
    settings.DATASET_PATH + settings.IONCHANNELS_MEMBRANEPROTEINS + "_" + "balanced" + "_" + settings.TRAIN + "_1" + ".csv")
ionchannel_membrane_test_df = pd.read_csv(
    settings.DATASET_PATH + settings.IONCHANNELS_MEMBRANEPROTEINS + "_" + "balanced" + "_" + settings.TEST + "_1" + ".csv")
iontransporter_membrane_train_df = pd.read_csv(
    settings.DATASET_PATH + settings.IONTRANSPORTERS_MEMBRANEPROTEINS + "_" + "balanced" + "_" + settings.TRAIN + "_1" + ".csv")
iontransporter_membrane_test_df = pd.read_csv(
    settings.DATASET_PATH + settings.IONTRANSPORTERS_MEMBRANEPROTEINS + "_" + "balanced" + "_" + settings.TEST + "_1" + ".csv")

# We count the number of sequences in each class for each dataset
membrane_train = ionchannel_membrane_train_df[ionchannel_membrane_train_df["label"]
                                              == "membrane_proteins"].shape[0]
membrane_test = ionchannel_membrane_test_df[ionchannel_membrane_test_df["label"]
                                            == "membrane_proteins"].shape[0]

# We create bar plots for each dataset. Train test beside each other for each class
bar_width = 0.35
ind = [0, 1, 2]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(ind, [ionchannels_train, iontransporters_train,
             membrane_train], bar_width, color='b', label='Train')
ax.bar([i + bar_width for i in ind], [ionchannels_test,
                                      iontransporters_test, membrane_test], bar_width, color='r', label='Test')

ax.set_ylabel('Number of sequences')
# ax.set_title('Ion channels, membrane proteins and ion transporters')
ax.set_xticks([i + bar_width * 0.5 for i in ind])
ax.set_xticklabels(('Ion channels', 'Ion transporters', 'Other MPs'))
# ax.set_yticks(np.arange(0, 4000, 100))
ax.legend(loc='best')

# Set y-axis limit
ax.set_ylim(0, 1000)

# Loop through each bar in the plot
for p in ax.patches:
    # Get the bar's x and y coordinates and its height
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    height = p.get_height()
    # Write the value of the bar on top of it
    ax.text(x, y, height, ha='center', va='bottom')

plt.savefig(settings.PLOT_PATH + "ionchannels_iontransporters_membrane_balanced2.png",
            dpi=300, bbox_inches='tight')
plt.close()
