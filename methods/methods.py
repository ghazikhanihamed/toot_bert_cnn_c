
import umap
import matplotlib.pyplot as plt
from settings import settings
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns


def plot_umap_datasets(datasets, label_dict, task, filename, figsize=(10, 5)):
    # Define the color mapping dictionary for the three tasks with specific HEX colors
    task_color_mapping = {
        'ionchannels_membraneproteins': ['#f1a340', '#998ec3'],
        'iontransporters_membraneproteins': ['#ef8a62', '#999999'],
        'ionchannels_iontransporters': ['#d8b365', '#5ab4ac'],
    }

    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, sharey=True)

    if n_datasets == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        representation_array = np.vstack(dataset['representation'].tolist())

        reducer = umap.UMAP()
        embeddings = reducer.fit_transform(representation_array)

        encoded_labels = dataset['label'].values

        unique_encoded_labels = np.unique(encoded_labels)
        unique_labels = [label_dict[encoded_label]
                         for encoded_label in unique_encoded_labels]
        # Assign the same colors for each task using the color mapping dictionary
        colors = task_color_mapping[task]

        for label, encoded_label, color in zip(unique_labels, unique_encoded_labels, colors):
            indices = encoded_labels == encoded_label
            axes[i].scatter(embeddings[indices, 0], embeddings[indices, 1], c=[
                            color], label=label, s=5, alpha=0.7)

        axes[i].set_title('')
        axes[i].set_xlabel('')
        if i == 0:
            axes[i].set_ylabel('')

    patches = [mpatches.Patch(color=color, label=label)
               for label, color in zip(unique_labels, colors)]
    # Adjust legend positioning to place it inside the box
    axes[-1].legend(handles=patches, loc='lower right')

    fig.tight_layout()
    fig.savefig(settings.PLOT_PATH + filename, dpi=300, bbox_inches='tight')


def train(model, train_dataloader, optimizer, criterion, device, task):
    pass


def evaluate(model, val_dataloader, criterion, device, task):
    pass


def test(model, test_dataloader, criterion, device, task):
    pass
