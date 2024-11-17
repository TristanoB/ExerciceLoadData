import torch
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import os
import numpy as np
import json

class NetCDFDataset(Dataset):
    def __init__(self, file_paths, variable_name):
        """
        Args:
            file_paths (list): List of paths to .nc files.
            variable_name (str): Name of the variable to extract from .nc files.
        """
        self.file_paths = file_paths
        self.variable_name = variable_name

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the file to load.
        
        Returns:
            torch.Tensor: Data corresponding to the selected variable.
        """
        file_path = self.file_paths[idx]
        with nc.Dataset(file_path, 'r') as dataset:
            data = np.array(dataset[self.variable_name][:])
        return torch.tensor(data, dtype=torch.float32)

# Example usage
def create_dataloader(nc_dir, variable_name, batch_size, shuffle=True, num_workers=0):
    """
    Create a PyTorch DataLoader for .nc files.
    
    Args:
        nc_dir (str): Directory containing .nc files.
        variable_name (str): Variable to load from .nc files.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker processes to use.
    
    Returns:
        DataLoader: PyTorch DataLoader for the .nc files.
    """
    file_paths = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')]
    dataset = NetCDFDataset(file_paths, variable_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Example parameters
if __name__ == "__main__":
    
    parameters = json.load(os.path('\Users\trist\Documents\dev\Exercice_stage\parameters.json'))

    nc_directory = "C:\Users\trist\Documents\dev\Exercice_stage\data"
    
    variable_name = "temperature"
    
    batch_size = parameters['batch_size']


    dataloader = create_dataloader(nc_directory, variable_name, batch_size)


    for batch_idx, batch_data in enumerate(dataloader):
        print(f"Batch {batch_idx}: {batch_data.shape}")













