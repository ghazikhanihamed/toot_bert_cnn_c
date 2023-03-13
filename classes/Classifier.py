

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from torch.functional import F

# Define the CNN model


class CNN(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7], out_channels=[512, 256, 128, 64, 32], input_size=1024, output_size=2, dropout_prob=0.2):
        super(CNN, self).__init__()

        # Define the input channel
        input_channel = 1

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        output_dim = 0
        for i in range(len(kernel_sizes)):
            for j in range(len(out_channels)):
                output_dim += out_channels[j]
                padding = (kernel_sizes[i] - 1) // 2
                self.conv_layers.append(nn.Conv2d(
                    input_channel, out_channels[j], (kernel_sizes[i], input_size), padding=(padding, 0)))
                
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected layer
        self.fc1 = nn.Linear(output_dim, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Convolutional layers
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]

        # Max pooling
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        # Concatenate the outputs of the convolutional layers
        x = torch.cat(x, 1)

        # Dropout
        x = self.dropout(x)

        # Fully connected layer
        x = self.fc1(x)

        x = F.softmax(x, dim=1)

        return x
