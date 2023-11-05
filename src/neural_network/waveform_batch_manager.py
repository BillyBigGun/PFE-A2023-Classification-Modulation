import os
from torch.utils.data import Dataset
import numpy as np

class WaveformBatchManager(Dataset):
    """Custom Dataset for loading wave signal data"""

    def __init__(self, data_dir="./data/raw", transform=None):
        """
        Args:
            data_dir: The location of the stored data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir

        # Get all the file names in the directory
        self.file_list = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the file path
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        
        # Load signal data 
        all_data = np.loadtxt(file_path, dtype=str)
        label = all_data[0]
        signal = all_data[1:].astype(float) 

        if self.transform:
            signal = self.transform(signal)

        return signal, label