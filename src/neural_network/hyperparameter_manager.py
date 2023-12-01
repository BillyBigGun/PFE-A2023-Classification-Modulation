import torch.nn.functional as F
from ray import tune

OUTPUT_SIZE = 4

class HyperparameterManager():
    def __init__(self) -> None:
        pass
        
    def get_ann_parameters(self):
        hyperparameters_ann = dict(
            input_size = 128,
            hidden_sizes = [128,128, 64,64,64,64, 32,32, 16,16], #[128,128,64,64,32,32,32,32,32,32,32,32,32,32,16,16],
            output_size = OUTPUT_SIZE,
            learning_rate = 0.0005,
            activation = F.relu,
            normalize_batch = False
        )

        return hyperparameters_ann
    
    
    def get_conv_layer(self, base_channels, in_channels, out_channels, kernel_size= (3, 1), padding= (1,0), stride = 1):
        return {
            'conv': {
                'in_channels': in_channels, 
                'out_channels': out_channels, 
                'kernel_size': kernel_size, 
                'padding': padding,
                'stride': stride,
            },
            'batch_norm': {'num_features': base_channels},
            'activation': 'relu'
        }

    def get_layers(self, block_channels, num_layers):
        layers = [self.get_conv_layer(block_channels, 1, block_channels)]
        layers += [self.get_conv_layer(block_channels, block_channels, block_channels) for _ in range(num_layers)]
        layers.append(self.get_conv_layer(1, block_channels, 1))
        return layers
    
    def get_cnn_parameters_2d(self, nb_filter_cnn_A, nb_layer_A, nb_filter_cnn_B, nb_layer_B, nb_filter_cnn_C, nb_layer_C, learning_rate=3e-4):
        # in 4x128 --> out before fc = 2x16 = 32
        fc_inputs = 32  # Output before fully connected layer

        # Setup Blocks
        layers_A = self.get_layers(nb_filter_cnn_A, nb_layer_A)
        layers_B = self.get_layers(nb_filter_cnn_B, nb_layer_B, kernel_size=(3,3), padding = 1)
        layers_C = self.get_layers(nb_filter_cnn_C, nb_layer_C, kernel_size=(3,3), padding= 1)
        
        # Define hyperparameters
        hyperparameters_cnn = {
            'input_size': 128,
            'output_size': OUTPUT_SIZE,
            'learning_rate': learning_rate,
            'blocks': [
                [layers_A, {'pool': {'type': 'max', 'kernel_size': (2, 1), 'stride': (2, 1)}}],
                [layers_B, {'pool': {'type': 'max', 'kernel_size': (2, 1), 'stride': (2, 1)}}],
                [layers_C, {'pool': {'type': 'max', 'kernel_size': (2, 2), 'stride': (2, 2)}}],
                [{'fc': {'in_features': fc_inputs, 'out_features': OUTPUT_SIZE}}],
            ]
        }
    
        return hyperparameters_cnn

    def get_cnn_parameters_1d(self, nb_filter_cnn_A, nb_layer_A, nb_filter_cnn_B, nb_layer_B, nb_filter_cnn_C, nb_layer_C, learning_rate=0.0005):
        fc_inputs = 16  # Output before fully connected layer

        # Setup Blocks
        layers_A = self.get_layers(nb_filter_cnn_A, nb_layer_A)
        layers_B = self.get_layers(nb_filter_cnn_B, nb_layer_B)
        layers_C = self.get_layers(nb_filter_cnn_C, nb_layer_C)
        
        # Define hyperparameters
        hyperparameters_cnn = {
            'input_size': 128,
            'output_size': OUTPUT_SIZE,
            'learning_rate': learning_rate,
            'blocks': [
                [layers_A, {'pool': {'type': 'max', 'kernel_size': (2, 1), 'stride': (2, 1)}}],
                [layers_B, {'pool': {'type': 'max', 'kernel_size': (2, 1), 'stride': (2, 1)}}],
                [layers_C, {'pool': {'type': 'max', 'kernel_size': (2, 1), 'stride': (2, 1)}}],
                [{'fc': {'in_features': fc_inputs, 'out_features': OUTPUT_SIZE}}],
            ]
        }
    
        return hyperparameters_cnn

    def get_t_cnn_parameters(self, in_channels, num_layers, num_channels_each_layer, learning_rate=0.0005):
        
        num_channels = [num_channels_each_layer]*num_layers

        hyperparameters_t_cnn = {
            'input_size': 128, 
            'input_channel': in_channels,  # number of input channels
            'num_channels': num_channels,  # number of output channels for each level of TCN blocks
            'output_size': OUTPUT_SIZE,
            'kernel_size': 3,  # size of the convolutional kernel
            'learning_rate' : learning_rate,
            'dropout': 0.2,  # dropout rate
        }

        return hyperparameters_t_cnn