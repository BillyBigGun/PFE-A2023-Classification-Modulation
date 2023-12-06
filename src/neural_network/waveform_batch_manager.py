import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import itertools
 
class WaveformBatchManager(Dataset):
    """Custom Dataset for loading wave signal data"""

    def __init__(self, data_dirs, nb_inputs, eval_ratio=0.1, transform=None):
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
        self.eval_ratio = eval_ratio
        
        for data_dir in data_dirs:
            # Ensure the directory exists
            if not os.path.isdir(data_dir):
                print(f'not a dir {data_dir}')
                continue

            # Add files from this directory to the list
            self.file_list += [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                               if os.path.isfile(os.path.join(data_dir, f))]

        self.snr_file_list, self.snr_index = self.organise_files_by_snr()
        # Keep data for evaluation
        print(f"\n\nLenght file list : {len(self.file_list)}\n\n")
        self.train_files, self.eval_files = self.create_train_eval_list()

        self.nb_inputs = nb_inputs
        self.transform = transform


    def organise_files_by_snr(self):
         # Initialize a dictionary to hold files sorted by SNR
        snr_dict = {}
        snr_index = {}
        index = 0

        # Iterate over the file list
        for file in self.file_list:
            # Extract the SNR value from the file name
            # Assuming file name format is '../../folderName_SNRValue_index'
            # Keep the last part of the hole file name
            file_name = file.split('/')[-1]

            parts = file_name.split('_')
            snr_value = int(parts[1])

            # Append the file to the correct SNR list in the dictionary
            if snr_value not in snr_dict:
                print(f'Adding SNR value: {snr_value}')
                snr_dict[snr_value] = []
                snr_index[snr_value] = index
                index += 1
            snr_dict[snr_value].append(file)

        # Convert the dictionary to a 2D list
        snr_file_list = list(snr_dict.values())

        return snr_file_list, snr_index
    
    def create_train_eval_list(self):
        train_list = []
        eval_list = []

        for snr_subset in self.snr_file_list:
            #print(len(snr_subset))
            random.shuffle(snr_subset)
            split_index = int(len(snr_subset) * self.eval_ratio)
            train_list.append(snr_subset[split_index:])
            eval_list.append(snr_subset[:split_index])

        return train_list, eval_list
    
    def training_mode(self):
        self.mode = 'train'

        # flatten the list
        self.file_list = list(itertools.chain(*self.train_files))

    def eval_mode(self):
        self.mode = 'eval'
        self.file_list = list(itertools.chain(*self.eval_files))

    def eval_mode_snr(self, snr_value):
        # get the the correct index for the specified snr_value
        if snr_value in self.snr_index:
            self.mode = 'eval'
            self.file_list = self.eval_files[self.snr_index[snr_value]]
        else:
            print(self.snr_index)
            print("SNR value does not exist")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        #print(file_name)
        #file_path = os.path.join(self.data_dir, file_name)
        
        # Load only the first 128 data points from the CSV file
        signal = pd.read_csv(file_name, usecols=[0], nrows=self.nb_inputs, skiprows=0).values.flatten() # call the skip_rows function
  
        # Convert label 'PAM' to 0 and 'PWM' to 1
        if 'PAM' in file_name:
            label = 0
        elif 'PWM' in file_name:
            label = 1
        elif 'QPSK' in file_name:
            label = 2
        else:
            label = 3

        if self.transform:
            signal = self.transform(signal)
        signal =  torch.from_numpy(signal).float() 
        return signal, label


# ==================================================================
#               HDF5 SECTION
#===================================================================
import h5py
class WaveformBatchManagerHDF5(Dataset):
    def __init__(self, file, modulation_classes, nb_input, eval_ratio=0.1, transform=None):
        self.frames_per_modulation = 4096
        self.points_per_frames = 1024
        self.snrs = list(range(-2, 22, 2))

        self.file = file
        self.modulation_classes = modulation_classes
        self.nb_input = nb_input
        self.eval_ratio = eval_ratio
        self.transform = transform

        # Create one array of the index of the items for evaluation and one for training
        self.eval_indices = [] #np.array([], dtype=int)
        self.train_indices = np.array([], dtype=int)
        nb_eval_per_snr = int(self.frames_per_modulation * eval_ratio)
        
        for i in range(len(self.snrs) * len(modulation_classes)):
            # Keep an equal ratio for each snrs indices
            indices_array = np.arange(self.frames_per_modulation * i, self.frames_per_modulation * (i + 1))
            # eval indices
            selected_indices = np.random.choice(indices_array, nb_eval_per_snr, replace=False)
            self.eval_indices.append(selected_indices)

            # training indices
            not_selected_indices = np.setdiff1d(indices_array, selected_indices)
            self.train_indices = np.concatenate((self.train_indices, not_selected_indices))

        # convert to numpy
        self.eval_indices = np.array(self.eval_indices)
        #with h5py.File(self.file, 'r') as f:
        #    self.data_shape = list(f['X'].shape) #(a, b, c)

        self.mode = 'train'
        self.current_indices = self.train_indices

    def training_mode(self):
        self.mode = 'train'
        self.current_indices = self.train_indices


    def eval_mode(self):
        self.mode = 'eval'

        self.current_indices = self.eval_indices.flatten()
        print(len(self.current_indices))

    
    def eval_mode_snr(self, snr_value):
        """
        Get the evaluation array for a specific snr_value
        """
        try:
            snr_index = self.snrs.index(snr_value)

            # snr_index exist
            self.mode ='eval'        
            snr_array = np.array([], dtype=int)

            # Get the evaluation data of all modulation at this snr
            for i in range(len(self.modulation_classes)):
                current_index = snr_index + i * len(self.snrs)
                snr_array = np.concatenate((snr_array, self.eval_indices[current_index]))
            
            self.current_indices = snr_array

        except ValueError:
            print(f"No snr at this value : {snr_value}")
            self.current_indices = None

    
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

        signal_1d=np.array(signal_1d)
        if self.transform:
            signal_1d = self.transform(signal_1d)

        #print(torch.tensor(signal_1d))
        return torch.from_numpy(signal_1d).float(), label

