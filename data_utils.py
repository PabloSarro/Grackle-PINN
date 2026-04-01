import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class GrackleDataset(Dataset):
    def __init__(self, file_path):
        """
        Reads Grackle .dat files and provides normalized tensors for PINN training.
        """
        # Read file, dropping the '#' header lines
        data = pd.read_csv(file_path, comment='#', delim_whitespace=True, header=None)
        
        # Rigorous column mapping via integer indices
        time = data.iloc[:, 1].values.astype(np.float32)
        temp = data.iloc[:, 3].values.astype(np.float32)
        hi   = data.iloc[:, 6].values.astype(np.float32)
        hii  = data.iloc[:, 7].values.astype(np.float32)

        # Preprocessing: Strict logarithmic scaling with hard floors
        # Utilizing np.maximum to prevent log10(0)
        self.t_norm = np.log10(np.maximum(time, 1e-20)) 
        
        # Stack targets into a single [N, 3] matrix for vectorized PINN loss
        y_raw = np.stack([temp, hi, hii], axis=1)
        self.y_norm = np.log10(np.maximum(y_raw, 1e-20))

    def __len__(self):
        return len(self.t_norm)

    def __getitem__(self, idx):
        """
        Returns (x, y) where:
        x: Time tensor [1]
        y: State tensor [Temp, HI, HII] [3]
        All data is kept on CPU here to allow num_workers > 0 in DataLoader.
        """
        t_tensor = torch.tensor([self.t_norm[idx]], dtype=torch.float32)
        y_tensor = torch.tensor(self.y_norm[idx], dtype=torch.float32)
        
        return t_tensor, y_tensor