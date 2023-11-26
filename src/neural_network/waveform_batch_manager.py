import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
 
class WaveformBatchManager(Dataset):
    """Custom Dataset for loading wave signal data"""

    def __init__(self, data_dirs, nb_inputs, max_first_row_=1, max_stride_rows_=1, eval_ratio=0.1, transform=None):
        """
        Args:
            data_dirs: List of directories containing the stored data.
            nb_inputs: the nb of inputs to be read in the file
            max_first_row: The starting row that will be read in the file. It is a random number between 1 and this value
            max_stride_rows: the number of rows to skip between each value read. It is a random number between 0 and this value
            eval_ratio: the ratio of data kept for evaluation
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_list = []
        self.file_path = []
        self.max_first_row = max_first_row_
        self.max_stride_rows = max_stride_rows_
        
        for data_dir in data_dirs:
            # Ensure the directory exists
            if not os.path.isdir(data_dir):
                print(f'not a dir {data_dir}')
                continue

            # Add files from this directory to the list
            self.file_list += [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                               if os.path.isfile(os.path.join(data_dir, f))]

        # Keep data for evaluation
        print(f"\n\n\n\n\n\n\nLenght file list : {len(self.file_list)}\n\n\n\n\n")
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
        # Skip rows before the first index and every divisible by the nb_skip_rows_values
        # these values are random
        #self.first_row = random.randint(1, self.max_first_row)
        #self.stride_row = random.randint(1, self.max_stride_rows)
        signal = pd.read_csv(file_name, usecols=[0], nrows=self.nb_inputs, skiprows=1).values.flatten() # call the skip_rows function
        # print(signal)

        # Convert label 'PAM' to 0 and 'PWM' to 1
        label = 0 if 'PAM' in file_name else 1

        if self.transform:
            signal = self.transform(signal)
        return torch.from_numpy(signal).float(), label


    def skip_rows(self, index):
        if self.first_row < index:
            if (index + self.first_row) % self.stride_row == 0 : # Skip every row except those divisible by nb_skip_rows
                #print(f"index : {index}")
                return False # False = do not skip this row
        return True

import h5py
class WaveformBatchManagerHDF5(Dataset):
    def __init__(self, file, modulation_classes, nb_input, eval_ratio=0.1, transform=None):
        self.frames_per_modulation = 4096
        self.points_per_frames = 1024
        self.snrs = list(range(-2, 20, 2))

        self.file = file
        self.modulation_classes = modulation_classes
        self.nb_input = nb_input
        self.eval_ratio = eval_ratio
        self.transform = transform

        # Create one array of the index of the items for evaluation and one for training
        self.eval_indices = np.array([], dtype=int)
        self.train_indices = np.array([], dtype=int)
        nb_eval_per_snr = int(self.frames_per_modulation * eval_ratio)
        
        for i in range(len(self.snrs) * len(modulation_classes)):
            # Keep an equal ratio for each snrs indices
            indices_array = np.arange(self.frames_per_modulation * i, self.frames_per_modulation * (i + 1))
            # eval indices
            selected_indices = np.random.choice(indices_array, nb_eval_per_snr, replace=False)
            self.eval_indices = np.concatenate((self.eval_indices, selected_indices))

            # training indices
            not_selected_indices = np.setdiff1d(indices_array, selected_indices)
            self.train_indices = np.concatenate((self.train_indices, not_selected_indices))

        #with h5py.File(self.file, 'r') as f:
        #    self.data_shape = list(f['X'].shape) #(a, b, c)

        self.mode = 'train'
        self.current_indices = self.train_indices

    def training_mode(self):
        self.mode = 'train'
        self.current_indices = self.train_indices

    def eval_mode(self):
        self.mode = 'eval'
        self.current_indices = self.eval_indices
    
    def __len__(self):
        # data shape = (frames, point per frame, channels)
        return len(self.current_indices)
    
    def __getitem__(self, idx):

        idx = self.current_indices[idx]

        signal_1d = np.empty(self.nb_input)
        with h5py.File(self.file, 'r') as f:
            signal_2d = f['X'][idx]
            # Get a random sequence of nb_input(128) in the points_in_frame array
            start_index = random.randint(0, self.points_per_frames-self.nb_input-1)
            
            # Get first column of the IQ signal. 
            # Get a sequence (128) from start index
            signal_1d = [row[0] for row in signal_2d[start_index:start_index+self.nb_input]]

        # 0 for the first modulation, 1 for the second...
        label = int(np.floor(idx/(self.frames_per_modulation*len(self.snrs))))

        if self.transform:
            self.transform(signal_1d)

        #print(torch.tensor(signal_1d))
        return torch.tensor(signal_1d), label

