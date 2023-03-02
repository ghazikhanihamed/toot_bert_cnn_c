import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_sizes=[3, 5, 7, 9], hidden_layers=[100, 50], filters=[32, 64]):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_sizes = kernel_sizes
        self.hidden_layers = hidden_layers
        self.filters = filters
        
        # Create the convolutional layers
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            for n_filters in filters:
                padding = (kernel_size - 1) // 2
                conv_layer = nn.Conv1d(in_channels=input_dim, out_channels=n_filters, kernel_size=kernel_size, padding=padding)
                self.conv_layers.append(conv_layer)
        
        # Create the fully connected layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_layers)-1):
            fc_layer = nn.Linear(hidden_layers[i], hidden_layers[i+1])
            self.fc_layers.append(fc_layer)
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = nn.functional.relu(conv_layer(x))
            conv_output = nn.functional.max_pool1d(conv_output, kernel_size=conv_output.size()[2])
            conv_outputs.append(conv_output.squeeze())
        x = torch.cat(conv_outputs, dim=1)
        for fc_layer in self.fc_layers:
            x = nn.functional.relu(fc_layer(x))
        output = self.output_layer(x)
        return output
