import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
import os

class GrackleDataset(Dataset):
    def __init__(self, folder_path, pattern="output_*.dat"):
        """
        Load multiple Grackle simulations (will be our inputs).
        PINN's input  (x): [log10(t), log10(T0), log10(HI0), log10(HII0)]
        PINN's output (y): [log10(T), log10(HI), log10(HII)]
        """
        self.all_inputs = []
        self.all_targets = []
        
        # Search for all files that match the pattern
        files = glob.glob(os.path.join(folder_path, pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching the pattern {pattern} in {folder_path}")

        for f in files:
            # Read data (using the modern syntax to avoid the FutureWarning)
            data = pd.read_csv(f, comment='#', sep='\s+', header=None)
            
            # 1. Extract raw state vectors
            time = data.iloc[:, 1].values.astype(np.float32)
            temp = data.iloc[:, 3].values.astype(np.float32)
            hi   = data.iloc[:, 6].values.astype(np.float32)
            hii  = data.iloc[:, 7].values.astype(np.float32)

            # 2. Identify the Initial Condition (IC)
            t0, hi0, hii0 = temp[0], hi[0], hii[0]

            # 3. Logarithmic Normalization (with safety floor to avoid log(0))
            log_t = np.log10(np.maximum(time, 1e-20))
            log_temp = np.log10(np.maximum(temp, 1e-20))
            log_hi = np.log10(np.maximum(hi, 1e-20))
            log_hii = np.log10(np.maximum(hii, 1e-20))
            
            # Calculate the scalar values first
            val_t0 = np.log10(np.maximum(t0, 1e-20))
            val_hi0 = np.log10(np.maximum(hi0, 1e-20))
            val_hii0 = np.log10(np.maximum(hii0, 1e-20))

            # Turn them into arrays of the same length as log_t
            log_t0 = np.full_like(log_t, val_t0)
            log_hi0 = np.full_like(log_t, val_hi0)
            log_hii0 = np.full_like(log_t, val_hii0)

            # Stack into (N, 4) and (N, 3)
            file_inputs = np.stack([log_t, log_t0, log_hi0, log_hii0], axis=1)
            file_targets = np.stack([log_temp, log_hi, log_hii], axis=1)

            self.all_inputs.append(file_inputs)
            self.all_targets.append(file_targets)

        self.all_inputs = torch.from_numpy(np.concatenate(self.all_inputs, axis=0)).float()
        self.all_targets = torch.from_numpy(np.concatenate(self.all_targets, axis=0)).float()

        self.x_min = self.all_inputs.min(axis=0)[0]
        self.x_max = self.all_inputs.max(axis=0)[0]

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, idx):
        x = (self.all_inputs[idx] - self.x_min) / (self.x_max - self.x_min + 1e-8)
        return x, self.all_targets[idx]




# Test block
if __name__ == "__main__":
    # Make sure you have output_1.dat and output_2.dat in the current folder before running this test
    try:
        ds = GrackleDataset(folder_path=".")
        print(f"✓ Dataset multi-file loaded: {len(ds)} total data points.")
        
        x, y = ds[0]
        print(f"✓ Example of input (t, T0, HI0, HII0): {x.numpy()}")
        print(f"✓ Example of output  (T, HI, HII): {y.numpy()}")
        
        if x.shape[0] != 4:
            print("✗ Error: The input dimension should be 4.")
    except Exception as e:
        print(f"✗ Error: {e}")