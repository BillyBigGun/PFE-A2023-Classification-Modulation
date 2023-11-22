import torch
import torch.nn.functional as F
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import sys
import matplotlib.pyplot as plt
sys.path.append("neural_network")
from trainer import Trainer
from nn_model import NNModel
from ann_model import ANNModel
from cnn_model import CNNModel
from t_cnn_model import TCNModel
from waveform_batch_manager import WaveformBatchManager



class HyperparameterTuner():
    def __init__(self):
        self.nb_input=128
        self.data_dir = ["../../data/raw/PAM_1_10.23", "../../data/raw/PWM_1_22.27"]
        self.batch_size=32
        self.shuffle = True
        self.num_workers = 2
        self.output_size = 3
        
    def get_ann_parameters_tuner(self):
        return {
            "input_size": 128,
            "hidden_sizes": tune.choice([[128, 64, 32, 16]*n for n in range(1, 17)]),
            "output_size": self.output_size,  # Définissez OUTPUT_SIZE selon vos besoins
            "learning_rate": tune.grid_search([0.001, 0.0001, 0.00001]),
            "activation": F.relu,  # Vous pourriez également vouloir optimiser cela
            "normalize_batch": True  # Vous pourriez également vouloir optimiser cela
        }
    def train_ann(self, config):
        self.train_set = WaveformBatchManager(self.data_dir, self.nb_input, eval_ratio=0.7)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last = True)
        
        model = ANNModel(config)

        model.train_tune_model(train_loader, 2)
        
    def tune_ann(self):
        
        config = self.get_ann_parameters_tuner()
    
        reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
        scheduler = ASHAScheduler(metric="loss", mode="min")
    
        analysis = tune.run(
            self.train_ann,
            resources_per_trial={"cpu": 1},  # Ajustez selon vos ressources disponibles
            config=config,
            num_samples=10,
            scheduler=scheduler,
            progress_reporter=reporter
        )
    
        best_trial = analysis.get_best_trial("loss", "min", "last")
        print("Meilleurs hyperparamètres trouvés étaient: ", best_trial.config)

if __name__ == "__main__":
    tuner = HyperparameterTuner()
    tuner.tune_ann()

