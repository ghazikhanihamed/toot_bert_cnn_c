
import umap
import matplotlib.pyplot as plt
from settings import settings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches


def plot_umap_datasets(datasets, label_dict, title='UMAP Plots', figsize=(15, 5)):
    """
    Takes a list of pandas DataFrames, each with columns 'id', 'representation', and 'label', and
    produces UMAP plots of each dataset as subplots for comparison.

    Args:
        datasets (list): List of pandas DataFrames with columns 'id', 'representation', and 'label'.
        label_dict (dict): Dictionary with integer keys and string values representing labels.
        title (str, optional): Title for the entire plot. Default is 'UMAP Plots'.
        figsize (tuple, optional): Figure size in inches. Default is (15, 5).

    Returns:
        None
    """
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, sharey=True)

    if n_datasets == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        # Convert 'representation' to a homogeneous 2D array
        representation_array = np.vstack(dataset['representation'].tolist())

        # Use UMAP for dimensionality reduction
        reducer = umap.UMAP()
        embeddings = reducer.fit_transform(representation_array)

        # Encode labels as integers
        encoded_labels = dataset['label'].values

        # Get unique labels and their corresponding colors
        unique_encoded_labels = np.unique(encoded_labels)
        unique_labels = [label_dict[encoded_label]
                         for encoded_label in unique_encoded_labels]
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

        # Plot the embeddings with different colors for each unique label
        for label, encoded_label, color in zip(unique_labels, unique_encoded_labels, colors):
            indices = encoded_labels == encoded_label
            axes[i].scatter(embeddings[indices, 0], embeddings[indices, 1], c=[
                            color], label=label, s=5, alpha=0.7)

        axes[i].set_title(f'Dataset {i+1}')
        axes[i].set_xlabel('UMAP 1')
        if i == 0:
            axes[i].set_ylabel('UMAP 2')

    # Create a legend using unique labels and their corresponding colors
    patches = [mpatches.Patch(color=color, label=label)
               for label, color in zip(unique_labels, colors)]
    fig.legend(handles=patches, loc='upper right',
               bbox_to_anchor=(1.1, 1), title='Labels')

    # Set the main title and adjust the layout
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


def train(model, train_dataloader, optimizer, criterion, device, task):
    pass


def evaluate(model, val_dataloader, criterion, device, task):
    pass


def test(model, test_dataloader, criterion, device, task):
    pass
