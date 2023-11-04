import torch
from torch import nn
import torch.nn.functional as F
from nn_model import NNModel
from ann_model import ANNModel
from cnn_model import CNNModel

MAX_EPOCH = 60
MINI_BATCH_SIZE = 100
LEARNING_RATE = 3E-4
LEARNING_RATE_DROP_PERIOD = 20
LEARNING_RATE_DROP_FACTOR = 0.1
OPTIMIZER = torch.optim.Adam()

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

class Trainer():
    def test():
        return