import torch.nn.functional as F

OUTPUT_SIZE = 3

class HyparameterManager():
    def __init__(self) -> None:
        pass
        
    def get_ann_parameters():
        hyperparameters_ann = dict(
            input_size = 128,
            hidden_sizes = [128,128,64,64,32,32,32,32,32,32,32,32,32,32,16,16],
            output_size = OUTPUT_SIZE,
            learning_rate = 3e-4,
            activation = F.relu
        )

        return hyperparameters_ann
    
    def get_cnn_parameters(nb_filter_cnn_A, nb_filter_cnn_B, nb_filter_cnn_C):
        # in 4x128 --> out before fc = 2x16 = 32

        # Setup Block A layers
        conv_layer_A = {'conv': {'in_channels': nb_filter_cnn_A, 'out_channels': nb_filter_cnn_A, 'kernel_size': (3, 1), 'stride': 1, 'padding': (1, 0)},
                    'batch_norm': {'num_features': nb_filter_cnn_A},
                    'activation': 'relu'}
        conv_layer_A_in = conv_layer_A
        conv_layer_A_in['conv']['in_channels'] = 1
        conv_layer_A_out = conv_layer_A
        conv_layer_A_out['conv']['out_channels'] = 1
        conv_layer_A_out['batch_nomr']['num_features'] = 1

       # Setup block B layers 
        conv_layer_B = {'conv': {'in_channels': nb_filter_cnn_B, 'out_channels': nb_filter_cnn_B, 'kernel_size': (3, 3), 'stride': 1, 'padding': 1},
                    'batch_norm': {'num_features': nb_filter_cnn_B},
                    'activation': 'relu'}
        conv_layer_B_in = conv_layer_B
        conv_layer_B_in['conv']['in_channels'] = 1
        conv_layer_B_out = conv_layer_B
        conv_layer_B_out['conv']['out_channels'] = 1
        conv_layer_B_out['batch_nomr']['num_features'] = 1

        # Setup Block C layers
        conv_layer_C = {'conv': {'in_channels': nb_filter_cnn_C, 'out_channels': nb_filter_cnn_C, 'kernel_size': (3, 3), 'stride': 1, 'padding': 1},
                    'batch_norm': {'num_features': nb_filter_cnn_C},
                    'activation': 'relu'}
        conv_layer_C_in = conv_layer_C
        conv_layer_C_in['conv']['in_channels'] = 1
        conv_layer_C_out = conv_layer_C
        conv_layer_C_out['conv']['out_channels'] = 1
        conv_layer_C_out['batch_nomr']['num_features'] = 1


        hyperparameters_cnn = {
            'input_size': 128,
            'output_size': OUTPUT_SIZE,
            'learning_rate': 3e-4,
            'blocks': [
                [ 
                    conv_layer_A_in, conv_layer_A, conv_layer_A, conv_layer_A, conv_layer_A_out,
                    {'pool': {'type': 'max', 'kernel_size': (2, 1), 'stride': (2, 1)}}
                ],
                [
                    conv_layer_B_in, conv_layer_B, conv_layer_B, conv_layer_B, conv_layer_B_out,
                    {'pool': {'type': 'max', 'kernel_size': (2, 1), 'stride': (2, 1)}}
                ],
                [
                    conv_layer_C_in, conv_layer_C, conv_layer_C,conv_layer_C,conv_layer_C_out,
                    {'pool': {'type': 'max', 'kernel_size': (2, 2), 'stride': (2, 2)}}
                ],
                [
                    # !IMPORTANT in_features needs to be calculated correctly
                    {'fc':{'in_features': 32, 'out_features': OUTPUT_SIZE}},
                ],
            ]
        }

        return hyperparameters_cnn

    def get_t_cnn_parameters(in_channels, num_layers, num_channels_each_layer):
        
        num_channels = [num_channels_each_layer]*num_layers

        hyperparameters_t_cnn = {
            'input_size': in_channels,  # number of input channels
            'num_channels': num_channels,  # number of output channels for each level of TCN blocks
            'output_size': OUTPUT_SIZE,
            'kernel_size': 3,  # size of the convolutional kernel
            'learning_rate' : 3e-4,
            'dropout': 0.2,  # dropout rate
        }

        return hyperparameters_t_cnn