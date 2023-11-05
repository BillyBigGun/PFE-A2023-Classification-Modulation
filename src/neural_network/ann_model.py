import torch
from torch import nn
import torch.nn.functional as F
from nn_model import NNModel

class ANNModel(NNModel):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, activation_function=F.relu):
        super(ANNModel, self).__init__(input_size, hidden_sizes, output_size, learning_rate, activation_function)
      
        layers = []
        
        # Define the first fully connected layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Define the fully connected layers
        for i in range(1,len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            # layers.append(nn.BatchNorm1D(hidden_sizes[i]))
            
        # Add the output layer
        layers.append(nn.Linear(hidden_sizes[i], output_size))
        # Store all layers in a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, input):
        # Pass data through each layers
        for layer in self.layers[:-1]:
            input = layer(input)
            input = self.activation(input)
        output = self.layers[-1](input)
        return output

# Example usage:
input_size = 10
hidden_sizes = [128, 64, 32]  # List of hidden layers sizes
output_size = 2
learning_rate = 3e-4
activation_function = F.relu

neural_network = ANNModel(input_size, hidden_sizes, output_size, learning_rate, activation_function)
print(neural_network)