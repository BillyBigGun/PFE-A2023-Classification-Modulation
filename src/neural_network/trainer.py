import torch
from torch import nn
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
from nn_model import NNModel
from ann_model import ANNModel
from cnn_model import CNNModel
from t_cnn_model import TCNModel
from waveform_batch_manager import WaveformBatchManager
from hyperparameter_manager import HyperparameterManager

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
    def __init__(self, model_class : NNModel, hyperparameters, data_dir=None, train_set= None, batch_size_=32, shuffle_=True, num_workers_=2, transform_=None):
        # create the NN model
        self.model = model_class(hyperparameters)
        self.nb_input = hyperparameters['input_size']
        self.batch_size = batch_size_
        self.shuffle = shuffle_
        self.num_workers = num_workers_
        self.train_set = train_set

        # Create the train set
        if train_set is None:
            self.train_set = WaveformBatchManager(data_dir, self.nb_input, eval_ratio=0.1, transform=transform_)
        #self.train_set = self.rand_train_set_cnn()
        #self.train_set = self.rand_train_set_t_cnn()        

    def train(self, epochs):
        # Create the train loader
        self.train_set.training_mode()
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last = True)
        
        self.model.train_model(self.train_loader, epochs)
        return

    def eval(self):
        self.train_set.eval_mode()
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last = True)

        accuracy = self.model.evaluate(self.train_loader)
        print(f"Accuracy : {accuracy}")    

    def eval_dir(self, dir):
        """
        Create a new evaluation set from the directory
        """
        self.eval_set = WaveformBatchManager(data_dir, self.nb_input, eval_ratio=1, transform=transform_)
        self.eval_set.eval_mode()
        self.train_loader = torch.utils.data.DataLoader(self.eval_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last = True)

        accuracy = self.model.evaluate(self.train_loader)
        print(f"Accuracy : {accuracy}")    

    def dry_run_model(self, input_shape):
        """
        Perform a dry run of the model to check input-output dimensions.
    
        Args:
        model (NNModel): NNModel class that inherit torch.nn.module.
        input_shape (tuple): The shape of the input tensor excluding the batch size.
                             For example, for a 1D signal, this might be (1, signal_length).
    
        Returns:
        None. Prints the shape of the tensor at each layer.
        """
        # Set the model to evaluation mode to disable layers like dropout and batch normalization
        self.model.eval()
    
        # Create a dummy input tensor with the specified input shape
        # The first dimension (batch size) is typically 1 for a dry run
        dummy_input = torch.rand(1, *input_shape)
    
        # Print the input shape
        print(f"Input shape: {dummy_input.shape}")

        # Pass the dummy input through each layer of the model
        with torch.no_grad():  # Disable gradient computation for efficiency
            print_layer=True
            for name, layer in self.model.named_modules():
                if print_layer:
                    print(layer)
                    print_layer = False
                    
                # Skip the overall model itself, only apply to layers
                if name:
                    if hasattr(layer, 'forward'):  # Check if the layer has a 'forward' method
                        try:
                            dummy_input = layer(dummy_input)
                            print(f"Layer {name}, output shape: {dummy_input.shape}")
                        except Exception as e:
                            print(f"Layer {name} threw an exception: {e}")


    # def rand_train_set_cnn(self):
    #     features = torch.randn(100, 1, 128, 4)
    #     labels = torch.randint(0,2,(100,))

    #     trainset = torch.utils.data.TensorDataset(features, labels)
    #     return trainset

    # def rand_train_set_t_cnn(self):
    #     features = torch.randn(100, 4, 128)
    #     labels = torch.randint(0,2,(100,))

    #     trainset = torch.utils.data.TensorDataset(features, labels)
    #     return trainset



if __name__ == "__main__":
    manager = HyperparameterManager()
    #trainer_ann = Trainer(ANNModel, manager.get_ann_parameters())
    #trainer_ann.train(32)

    #trainer_cnn = Trainer(CNNModel, manager.get_cnn_parameters(4,4,4))
    #print(trainer_cnn.model)
    #trainer_cnn.train(5)

    trainer_tcnn = Trainer(TCNModel, manager.get_t_cnn_parameters(1, 15, 4))
    trainer_tcnn.train(5)