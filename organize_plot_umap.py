import matplotlib.pyplot as plt
import numpy as np
from settings import settings

plot_list = [
    'ionchannels_iontransporters_train_finetuned_representations_ESM-1b_ionchannels_iontransporters_imbalanced_umap.png',
    'ionchannels_iontransporters_train_finetuned_representations_ESM-2_ionchannels_iontransporters_imbalanced_umap.png',
    'ionchannels_iontransporters_train_finetuned_representations_ProtBERT-BFD_ionchannels_iontransporters_imbalanced_umap.png',
    'ionchannels_iontransporters_train_finetuned_representations_ProtBERT_ionchannels_iontransporters_imbalanced_umap.png',
    'ionchannels_iontransporters_train_frozen_representations_ESM-1b_umap.png',
    'ionchannels_iontransporters_train_frozen_representations_ESM-2_15B_umap.png',
    'ionchannels_iontransporters_train_frozen_representations_ESM-2_umap.png',
    'ionchannels_iontransporters_train_frozen_representations_ProtBERT-BFD_umap.png',
    'ionchannels_iontransporters_train_frozen_representations_ProtBERT_umap.png',
    'ionchannels_iontransporters_train_frozen_representations_ProtT5_umap.png',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_ESM-1b_ionchannels_membraneproteins_imbalanced_umap.png',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_ESM-2_ionchannels_membraneproteins_imbalanced_umap.png',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_ProtBERT-BFD_ionchannels_membraneproteins_imbalanced_umap.png',
    'ionchannels_membraneproteins_imbalanced_train_finetuned_representations_ProtBERT_ionchannels_membraneproteins_imbalanced_umap.png',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ESM-1b_umap.png',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ESM-2_15B_umap.png',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ESM-2_umap.png',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ProtBERT-BFD_umap.png',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ProtBERT_umap.png',
    'ionchannels_membraneproteins_imbalanced_train_frozen_representations_ProtT5_umap.png',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_ESM-1b_iontransporters_membraneproteins_imbalanced_umap.png',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_ESM-2_iontransporters_membraneproteins_imbalanced_umap.png',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_ProtBERT-BFD_iontransporters_membraneproteins_imbalanced_umap.png',
    'iontransporters_membraneproteins_imbalanced_train_finetuned_representations_ProtBERT_iontransporters_membraneproteins_imbalanced_umap.png',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ESM-1b_umap.png',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ESM-2_15B_umap.png',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ESM-2_umap.png',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ProtBERT-BFD_umap.png',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ProtBERT_umap.png',
    'iontransporters_membraneproteins_imbalanced_train_frozen_representations_ProtT5_umap.png',
]

# define the task names, representation types, and representer names
task_names = ['ionchannels_iontransporters', 'ionchannels_membraneproteins', 'iontransporters_membraneproteins']
rep_types = ['frozen', 'finetuned']
rep_names = ['ProtBERT', 'ProtBERT-BFD', 'ProtT5', 'ESM-1b', 'ESM-2', 'ESM-2_15B']

# create a dictionary to store the filenames for each task, representer, and representation type
file_dict = {}
for task in task_names:
    file_dict[task] = {}
    for rep in rep_names:
        file_dict[task][rep] = {}
        for rep_type in rep_types:
            # Find the file for the given task, representer, and representation type
            filename = [f for f in plot_list if task in f and rep in f and rep_type in f]
            if len(filename) > 0:
                filename = filename[0]

            if filename in plot_list:
                file_dict[task][rep][rep_type] = filename

# create a figure for each task
for task in task_names:
    fig, axs = plt.subplots(len(rep_names), len(rep_types), figsize=(10, 20), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1]})
    fig.suptitle(task)

    # plot each representer for the given task and representation type
    for i, rep in enumerate(rep_names):
        # add representer name to left of row
        axs[i, 0].text(-0.4, 0.5, rep, fontsize=14, fontweight='bold', transform=axs[i, 0].transAxes, va='center')
        for j, rep_type in enumerate(rep_types):
            # check if a file exists for the given task, representer, and representation type
            if rep_type in file_dict[task][rep]:
                # load the image and plot it on the corresponding subplot
                img = plt.imread(settings.PLOT_PATH + file_dict[task][rep][rep_type])
                axs[i, j].imshow(img)
                axs[i, j].axis('off')
            else:
                # if no file exists, plot an empty subplot
                axs[i, j].axis('off')

            # add representation type to top of column
            if i == 0:
                axs[i, j].set_title(rep_type.capitalize(), fontsize=14, fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    plt.tight_layout()
    # plt.show()
    plt.savefig(settings.PLOT_PATH + task + '_umap.png', dpi=300, bbox_inches='tight')