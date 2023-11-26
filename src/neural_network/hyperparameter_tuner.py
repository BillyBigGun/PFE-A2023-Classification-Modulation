import torch
import torch.nn.functional as F
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import sys
import os
import matplotlib.pyplot as plt
sys.path.append("neural_network")
from trainer import Trainer
from nn_model import NNModel
from ann_model import ANNModel
from cnn_model import CNNModel
from t_cnn_model import TCNModel
from waveform_batch_manager import WaveformBatchManager
from waveform_batch_manager import WaveformBatchManagerHDF5
from hyperparameter_manager import HyperparameterManager


ABS_PATH1 = '~/School/PFE/PFE-A2023-Classification-Modulation/data/raw/PAM_1_10.23'
ABS_PATH2 = '~/School/PFE/PFE-A2023-Classification-Modulation/data/raw/PWM_1_22.27'
ABS_PATH_ONLINE_MOD ='~/School/PFE/PFE-A2023-Classification-Modulation/data/raw/DatasetOnline/Modulation_4ASK_BPSK_QPSK.hdf5' 

class HyperparameterTuner():
    def __init__(self):
        self.nb_input=128

        path1 = os.path.expanduser(ABS_PATH1)
        path2 = os.path.expanduser(ABS_PATH2)
        self.file_mod =  os.path.expanduser(ABS_PATH_ONLINE_MOD)

        self.data_dir = [path1, path2]
        self.batch_size=32
        self.shuffle = True
        self.num_workers = 2
        self.output_size = 4 

    # =========================
    #        ANN LAYERS
    # =========================
    def get_ann_parameters_tuner(self):
        return {
            "input_size": 128,
            "hidden_sizes": tune.choice([[128, 64, 32, 16]*n for n in range(1, 17)]),
            "output_size": self.output_size,  # Définissez OUTPUT_SIZE selon vos besoins
            "learning_rate": tune.grid_search([0.001, 0.0001, 0.00001]),
            "activation": F.relu,  # Vous pourriez également vouloir optimiser cela
            "normalize_batch": True  # Vous pourriez également vouloir optimiser cela
        }

    # ======================
    #       CNN LAYERS
    # ======================
    def get_cnn_parameters_1d_tuner(self):
        search_space = {
            'nb_filter_cnn_A': tune.choice([2,4,16]),
            'nb_layer_A': tune.choice([2, 3, 4]),
            'nb_filter_cnn_B': tune.choice([2,4,16]),
            'nb_layer_B': tune.choice([2, 3, 4]),
            'nb_filter_cnn_C': tune.choice([2,4,16]),
            'nb_layer_C': tune.choice([2, 3, 4]),
            'learning_rate': tune.loguniform(1e-4, 1e-2)
        }


        return search_space 

    # ======================
    #       TCN LAYERS
    # ======================
    def get_t_cnn_parameters_tuner(self):
        
        num_channels = [tune.grid_search(3,5,7)]*tune.grid_search(4,8,12)

        hyperparameters_t_cnn = {
            'input_size': 128, 
            'input_channel': 1,  # number of input channels
            'num_channels': num_channels,  # number of output channels for each level of TCN blocks
            'output_size': self.output_size,
            'kernel_size': 3,  # size of the convolutional kernel
            'learning_rate' : tune.grid_search([0.001, 0.0001, 0.00001]),
            'dropout': tune.grid_search([0.1, 0.2, 0.3]),  # dropout rate
        }

        return hyperparameters_t_cnn
    

    # ==============================
    #           TUNING
    # ==============================
    def train(self, config, nn_model:NNModel):
        #self.train_set = WaveformBatchManager(self.data_dir, self.nb_input, eval_ratio=0.7)
        self.train_set = WaveformBatchManagerHDF5(self.file_mod, ['4ASK', 'BPSK', 'QPSK'], 128)
        #trainer_ann = Trainer(ANNModel, parameters_ANN, train_set=train_set, batch_size_=32, num_workers_=2)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last = True)
        
        if nn_model==CNNModel:
            manager = HyperparameterManager()
            config = manager.get_cnn_parameters_1d(**config)

        model = nn_model(config)

        model.train_tune_model(self.train_loader, 2)
        
    def tune(self, nn_model:NNModel, config):
    
        reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
        scheduler = ASHAScheduler(metric="loss", mode="min")
    
        analysis = tune.run(
            tune.with_parameters(self.train, nn_model=nn_model),
            resources_per_trial={"cpu": 4}, 
            config=config,
            num_samples=10,
            scheduler=scheduler,
            progress_reporter=reporter
        )
    
        best_trial = analysis.get_best_trial("loss", "min", "last")
        print("Meilleurs hyperparamètres trouvés étaient: ", best_trial.config)
        

if __name__ == "__main__":
    
    tuner = HyperparameterTuner()

    tuner.tune(ANNModel, tuner.get_ann_parameters_tuner())
    #tuner.tune(CNNModel, tuner.get_cnn_parameters_1d_tuner())
    #tuner.tune(TCNModel, tuner.get_t_cnn_parameters_tuner())
