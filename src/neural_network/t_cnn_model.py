import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from nn_model import NNModel

# Chump the padding added
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu3(out + res)

class TCNModel(NNModel):
    def __init__(self, hyperparameters):
        super(TCNModel, self).__init__(hyperparameters['input_size'], hyperparameters['output_size'], hyperparameters['learning_rate'])
        layers = []
        num_inputs = hyperparameters['input_size']
        num_channels = hyperparameters['num_channels']
        self.output_size = hyperparameters['output_size']
        kernel_size = hyperparameters['kernel_size']
        dropout = hyperparameters['dropout']
        
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], self.output_size)

    def forward(self, x):
        x = self.network(x)
        # Assuming x has shape [batch_size, num_channels, sequence_length]
        # Apply global average pooling across the temporal dimension
        x = x.mean(dim=2)
        # Pass the result through the linear layer for classification
        return self.linear(x)

if __name__ == "__main__":
    # Example hyperparameters for a TCN
    tcn_hyperparameters = {
        'input_size': 4,  # number of input channels
        'num_channels': [8,8,8,8,8, 8,8,8,8,8, 8,8,8,8,8 ],  # number of output channels for each level of TCN blocks
        'output_size': 2,
        'kernel_size': 3,  # size of the convolutional kernel
        'learning_rate' : 3e-4,
        'dropout': 0.2,  # dropout rate
    }

    # Example usage
    tcn_model = TCNModel(tcn_hyperparameters)
    # with batch size of 32 and sequence length of 128
    # Note: TCN expects the data in the format (batch_size, num_channels, sequence_length)
    x = torch.randn(32, 4, 128)
    output = tcn_model(x)
    print(output.shape)  # Should be (batch_size, 1)
