import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import random
 
class WaveformBatchManager(Dataset):
    """Custom Dataset for loading wave signal data"""

    def __init__(self, data_dirs, nb_inputs, eval_ratio=0.1, transform=None):
        """
        Args:
            data_dirs: List of directories containing the stored data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_list = []
        self.file_path = []
        for data_dir in data_dirs:
            # Ensure the directory exists
            if not os.path.isdir(data_dir):
                continue

            # Add files from this directory to the list
            self.file_list += [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                               if os.path.isfile(os.path.join(data_dir, f))]

        # Keep data for evaluation
        random.shuffle(self.file_list)
        split_index = int(len(self.file_list)*(1- eval_ratio))
        self.train_files = self.file_list[:split_index]
        self.eval_files = self.file_list[split_index:]
        
        self.nb_inputs = nb_inputs
        self.transform = transform

    def training_mode(self):
        self.mode = 'train'
        self.file_list = self.train_files

    def eval_mode(self):
        self.mode = 'eval'
        self.file_list = self.eval_files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        #file_path = os.path.join(self.data_dir, file_name)
        
        # Load only the first 128 data points from the CSV file
        signal = pd.read_csv(file_name, usecols=[0], nrows=self.nb_inputs, skiprows=1).values.flatten()

        # Convert label 'PAM' to 0 and 'PWM' to 1
        label = 0 if 'PAM' in file_name else 1

        if self.transform:
            signal = self.transform(signal)

        return torch.from_numpy(signal).float(), label




# # Example usage
# transform = transforms.Compose([
#     # Add any necessary transformations here
# ])
# dataset = WaveformBatchManager(data_dir="./data/raw", transform=transform)
