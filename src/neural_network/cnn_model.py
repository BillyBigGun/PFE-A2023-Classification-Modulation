import torch
from torch import nn
import torch.nn.functional as F
from nn_model import NNModel

class CNNModel(NNModel):
    def __init__(self, hyperparameters):
        super(CNNModel, self).__init__(hyperparameters['input_size'], hyperparameters['output_size'], hyperparameters['learning_rate'])
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # Retrieve the configuration for each block from the config dictionary
        for block in hyperparameters['blocks']:
            for element in block:
                # Check if element is an array (list of layers)
                if isinstance(element, list):
                    for layer in element:
                        if 'conv' in layer:
                            conv_layer = nn.Conv2d(**layer['conv'])
                            self.conv_layers.append(conv_layer)
                            if 'batch_norm' in layer:
                                bn_layer = nn.BatchNorm2d(**layer['batch_norm'])
                                self.conv_layers.append(bn_layer)
                            if 'activation' in layer and layer['activation'] == 'relu':
                                self.conv_layers.append(nn.ReLU())
                else:
                    # Process single layer (pool or fc)
                    layer = element
                    if 'pool' in layer:
                        pool_type = layer['pool']['type']
                        if pool_type == 'max':
                            pool_layer = nn.MaxPool2d(layer['pool']['kernel_size'], layer['pool']['stride'])
                        elif pool_type == 'avg':
                            pool_layer = nn.AvgPool2dlayer(layer['pool']['kernel_size'], layer['pool']['stride'])
                        self.conv_layers.append(pool_layer)
                    elif 'fc' in layer:
                        self.fc_layers.append(nn.Flatten())
                        self.fc_layers.append(nn.Linear(**layer['fc']))

    def forward(self, x):
        # Pass the input through the convolutional and pooling layers
        for layer in self.conv_layers:
            x = layer(x)
            #print(x.size())
        
        # Flatten the tensor before passing it to the fully connected layers
        # x = torch.flatten(x, 1)
        
        # Pass the flattened output through the fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    # in 4x128 --> out before fc = 2x16 = 32
    hyperparameters = {
        'input_size': 128,
        'output_size': 2,
        'learning_rate': 3e-4,
        'blocks': [
            [ 
                {'conv': {'in_channels': 1, 'out_channels': 2, 'kernel_size': (3, 1), 'stride': 1, 'padding': (1, 0)},
                'batch_norm': {'num_features': 2},
                'activation': 'relu'},
                {'conv': {'in_channels': 2, 'out_channels': 1, 'kernel_size': (3, 1), 'stride': 1, 'padding': (1, 0)},
                'batch_norm': {'num_features': 1},
                'activation': 'relu'},
                {'pool': {'type': 'max', 'kernel_size': (2, 1), 'stride': (2, 1)}}
            ],
            [
                {'conv': {'in_channels': 1, 'out_channels': 2, 'kernel_size': (3, 1), 'stride': 1, 'padding': (1, 0)},
                'batch_norm': {'num_features': 2},
                'activation': 'relu'},
                {'conv': {'in_channels': 2, 'out_channels': 1, 'kernel_size': (3, 1), 'stride': 1, 'padding': (1, 0)},
                'batch_norm': {'num_features': 1},
                'activation': 'relu'},
                {'pool': {'type': 'max', 'kernel_size': (2, 1), 'stride': (2, 1)}}
            ],
            [
                {'conv': {'in_channels': 1, 'out_channels': 2, 'kernel_size': (2,2), 'stride': 1, 'padding': (1, 0)},
                'batch_norm': {'num_features': 2},
                'activation': 'relu'},
                {'conv': {'in_channels': 2, 'out_channels': 1, 'kernel_size': (2, 2), 'stride': 1, 'padding': (1, 0)},
                'batch_norm': {'num_features': 1},
                'activation': 'relu'},
                {'pool': {'type': 'max', 'kernel_size': (2, 2), 'stride': (2, 1)}}
            ],
            [
                {'fc':{'in_features': 32, 'out_features': 2}},
            ],
        ]
    }

    model = CNNModel(hyperparameters)
    print(model)