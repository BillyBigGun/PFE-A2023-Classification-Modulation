import torch
from torch import nn
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
from nn_model import NNModel
from ann_model import ANNModel
from cnn_model import CNNModel
from waveform_batch_manager import WaveformBatchManager

MAX_EPOCH = 60
MINI_BATCH_SIZE = 100
LEARNING_RATE = 3E-4
LEARNING_RATE_DROP_PERIOD = 20
LEARNING_RATE_DROP_FACTOR = 0.1
#OPTIMIZER = torch.optim.Adam()

"""
        | Paper         |       | 
        | best          |       | Nb 
        | performance   |       | of
Layer   | output        | test  | Layers
---------------------------------------------
Input   : 4x1024        | 2x128 | 1
A block : 4x512         | 2x64  | 4 + max pool
B block : 4x256         | 2x32  | 5
Pool    : 2x256         | ----  | 1
C block1: 2x256         | 2x32  | 3
C block2: 2x256         | 2x32  | 2
A pool  : 1x128         | 1x16  | 1
Full Con: 1x128         | 1x16  | 1
Softmax : 1x11          | 1x2   | 1

Min good input performance in paper: 4x128
"""

DATA_LOCATION = './data/raw'


class Trainer():
    def __init__(self, model_class : NNModel, hyperparameters, data_dir='data/raw', batch_size_=32, shuffle_=True, num_workers_=2):
        # create the NN model
        self.model = model_class(hyperparameters)
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Example normalization
        # ])

        # Create the train set
        #self.train_set = WaveformBatchManager(data_dir)
        self.train_set = self.rand_train_set()

        # Create the train loader
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size_, shuffle=shuffle_, num_workers=num_workers_)

    def train(self, epochs):
        self.model.train_model(self.train_loader, epochs)
        return
    
    def rand_train_set(self):
        features = torch.randn(100, 1, 128, 4)
        labels = torch.randint(0,2,(100,))

        trainset = torch.utils.data.TensorDataset(features, labels)
        return trainset

hyperparameters_ann = dict(
    input_size = 128,
    hidden_sizes = [128,128,64,64,32,32,32,32,32,32,32,32,32,32,16,16],
    output_size = 2,
    learning_rate = 3e-4,
    activation = F.relu
)

# in 4x128 --> out before fc = 2x16 = 32
hyperparameters_cnn = {
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
            {'conv': {'in_channels': 1, 'out_channels': 2, 'kernel_size': (3, 3), 'stride': 1, 'padding': 1},
             'batch_norm': {'num_features': 2},
             'activation': 'relu'},
            {'conv': {'in_channels': 2, 'out_channels': 1, 'kernel_size': (3, 3), 'stride': 1, 'padding': 1},
             'batch_norm': {'num_features': 1},
             'activation': 'relu'},
            {'pool': {'type': 'max', 'kernel_size': (2, 1), 'stride': (2, 1)}}
        ],
        [
            {'conv': {'in_channels': 1, 'out_channels': 2, 'kernel_size': (3,1), 'stride': 1, 'padding': (1,0)},
            'batch_norm': {'num_features': 2},
             'activation': 'relu'},
            {'conv': {'in_channels': 2, 'out_channels': 1, 'kernel_size': (3, 1), 'stride': 1, 'padding': (1,0)},
             'batch_norm': {'num_features': 1},
             'activation': 'relu'},
            {'pool': {'type': 'max', 'kernel_size': (2, 2), 'stride': (2, 2)}}
        ],
        [
            {'fc':{'in_features': 32, 'out_features': 2}},
        ],
    ]
}

if __name__ == "__main__":
    #trainer_ann = Trainer(ANNModel, hyperparameters_ann)
    #trainer_ann.train(32)

    trainer_cnn = Trainer(CNNModel, hyperparameters_cnn)
    #print(trainer_cnn.model)
    trainer_cnn.train(5)