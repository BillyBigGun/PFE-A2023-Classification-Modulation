import torch
import torch.nn.functional as F
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

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
import nn_transforms


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
        self.batch_size=100
        self.shuffle = True
        self.num_workers = 2
        self.output_size = 4 

    # =========================
    #        ANN LAYERS
    # =========================
    def get_ann_parameters_tuner(self):
        return {
            "input_size": 128,
            #"hidden_sizes": tune.choice([[128, 64, 32, 16]*n for n in range(1, 17)]),
            "hidden_sizes": tune.grid_search([#[128, 64, 32, 16],
                                             #[32, 32, 32, 32, 16, 16, 16, 16], #not bad
                                             #[128, 64, 64, 32, 32, 16, 16], # not bad
                                             #[128, 128, 64, 64, 32, 32, 16, 16], # not bad
                                             #[128, 128, 64, 64, 32, 32, 32, 32, 16, 16], # not bad
                                             #[128, 128, 64, 64, 32, 32, 16, 16, 16, 16],
                                             [128, 128, 64, 64, 64, 64, 32, 32, 16, 16], # good
                                             #[128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32, 16, 16, 16, 16]
                                             ]),
            "output_size": self.output_size,  # Définissez OUTPUT_SIZE selon vos besoins
            "learning_rate": 0.0005, #tune.grid_search([0.008, 0.005]), #tune.grid_search([0.0001, 0.00001]),
            "activation": F.relu,  # Vous pourriez également vouloir optimiser cela
            "normalize_batch": False
        }

    # ======================
    #       CNN LAYERS
    # ======================
    def get_cnn_parameters_1d_tuner(self):
        search_space = {
            'nb_filter_cnn_A': 12, #tune.grid_search([8, 12]),#tune.choice([2,4,16]),
            'nb_layer_A': 3, #tune.grid_search([3, 7]),
            'nb_filter_cnn_B': 16, #tune.grid_search([4, 16]),#tune.choice([2,4,16]),
            'nb_layer_B': 5, #tune.grid_search([3, 7]),
            'nb_filter_cnn_C': 16, #tune.grid_search([4,8,16]),#tune.choice([2,4,16]),
            'nb_layer_C': 7, #tune.grid_search([3, 7]),
            'learning_rate': 0.001#tune.loguniform(1e-4, 1e-2)
        }


        return search_space 

    # ======================
    #       TCN LAYERS
    # ======================
    def get_t_cnn_parameters_tuner(self):
        
        # num_channels = [4,4,4,4,4]#,[tune.grid_search([3,5,7])]*tune.grid_search([4,8,12])
        # num_channels += [[8]*n for n in [5,6]]
        # num_channels += [[16]*n for n in [4,8,16]]

        num_channels = [12,12,12, 8,8,8,8] #tune.grid_search([[12,12,12,8,8,8,8]])

        hyperparameters_t_cnn = {
            'input_size': 128, 
            'input_channel': 1,  # number of input channels
            'num_channels': num_channels,  # number of output channels for each level of TCN blocks
            'output_size': self.output_size,
            'kernel_size': 3,  # size of the convolutional kernel
            'learning_rate' : 0.001, #tune.grid_search([0.0001, 0.00001]),
            'dropout': 0.05#tune.grid_search([0.05, 0.1, 0.2]),  # dropout rate
        }

        return hyperparameters_t_cnn
    

    # ==============================
    #           TUNING
    # ==============================
    def train(self, config, nn_model:NNModel):
        #self.train_set = WaveformBatchManager(self.data_dir, self.nb_input, eval_ratio=0.7)
        transform = nn_transforms.get_transform_t_cnn()#nn_transforms.get_transform_to_2d()
        self.train_set = WaveformBatchManagerHDF5(self.file_mod, ['4ASK', 'BPSK', 'QPSK'], 128, transform=transform)
        #trainer_ann = Trainer(ANNModel, parameters_ANN, train_set=train_set, batch_size_=32, num_workers_=2)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last = True)
        
        if nn_model==CNNModel:
            manager = HyperparameterManager()
            config = manager.get_cnn_parameters_1d(**config)

        model = nn_model(config)

        model.train_tune_model(self.train_loader, 12)
        
    def tune(self, nn_model:NNModel, config):
    
        reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
        scheduler = ASHAScheduler(
            metric="loss", 
            mode="min", 
            grace_period=3,  # Increase the grace period
            #reduction_factor=2  # Adjust the reduction factor if necessary
        )

        analysis = tune.run(
            tune.with_parameters(self.train, nn_model=nn_model),
            resources_per_trial={"cpu": 2}, 
            config=config,
            num_samples=2,
            scheduler=scheduler,
            progress_reporter=reporter
        )
    
        best_trial = analysis.get_best_trial("loss", "min", "last")
        print("Meilleurs hyperparamètres trouvés étaient: ", best_trial.config)

        # Iterate through all the trials in the analysis object
        for trial in analysis.trials:
            # Check if the trial ended in error
            if trial.status == 'ERROR':
                # Fetch the log files associated with this trial
                logfiles = analysis.fetch_trial_data(trial).logdir
                # You might have to open these log files and read their contents
                print(f"Error in trial {trial.trial_id}: Logs at {logfiles}")

        

if __name__ == "__main__":
    
    tuner = HyperparameterTuner()

    #tuner.tune(ANNModel, tuner.get_ann_parameters_tuner())
    #tuner.tune(CNNModel, tuner.get_cnn_parameters_1d_tuner())
    tuner.tune(TCNModel, tuner.get_t_cnn_parameters_tuner())
