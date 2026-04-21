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

        # For each file
        for f in files:
            # Read its data
            data = pd.read_csv(f, comment='#', sep='\s+', header=None)
            
            # Extract its state vectors
            t = data.iloc[:, 1].values.astype(np.float32)
            T = data.iloc[:, 3].values.astype(np.float32)
            nHI   = data.iloc[:, 6].values.astype(np.float32)
            nHII  = data.iloc[:, 7].values.astype(np.float32)

            # Extract the Initial Conditions (first row of the file)
            T0, nHI0, nHII0 = T[0], nHI[0], nHII[0]

            # Logarithmic Normalisation (with safety floor to avoid log(0))

            # // ---- Inputs [t, T0, HI0, HII0] for every timestep t ---- \\
            # Log of the ICs.
            val_T0 = np.log10(np.maximum(T0, 1e-20))
            val_nHI0 = np.log10(np.maximum(nHI0, 1e-20))
            val_nHII0 = np.log10(np.maximum(nHII0, 1e-20))

            # Turn the ICs into possible inputs of the PINN --> arrays of size #timesteps
            t_lin = t.astype(np.float32)
            log_T0 = np.full_like(t_lin, val_T0)
            log_nHI0 = np.full_like(t_lin, val_nHI0)
            log_nHII0 = np.full_like(t_lin, val_nHII0)

            # Stack into (N, 4)
            file_inputs = np.stack([t_lin, log_T0, log_nHI0, log_nHII0], axis=1)
            # [log_t0, log_T0, log_HI0, log_HII0] (input at time t0), 
            # [log_t1, log_T0, log_HI0, log_HII0] (input at time t1), 
            # [log_t2, log_T0, log_HI0, log_HII0] (input at time t2), ...

            # // ---- Targets [T, nHI, nHII] for every timestep t ---- \\
            log_T = np.log10(np.maximum(T, 1e-20))
            log_nHI = np.log10(np.maximum(nHI, 1e-20))
            log_nHII = np.log10(np.maximum(nHII, 1e-20))

            # Stack into (N, 3)
            file_targets = np.stack([log_T, log_nHI, log_nHII], axis=1)
            # [log_T0, log_HI0, log_HII0] (output at time t0), 
            # [log_T1, log_HI1, log_HII1] (output at time t1), 
            # [log_T2, log_HI2, log_HII2] (output at time t2), ...

            # maybe no need to include outputs at t0, since they are the same as the ICs #

            self.all_inputs.append(file_inputs)
            self.all_targets.append(file_targets)

        self.all_inputs = torch.from_numpy(np.concatenate(self.all_inputs, axis=0)).float()
        self.all_targets = torch.from_numpy(np.concatenate(self.all_targets, axis=0)).float()
        
        print("\n--- TIME DEBUG: BEFORE NORM ---")
        raw_time = self.all_inputs[:, 0]
        print(f"Global Raw log_t Min: {raw_time.min().item():.4f}")
        print(f"Global Raw log_t Max: {raw_time.max().item():.4f}")
        print(f"First 10 raw log_t values:\n{raw_time[:10].numpy()}")

        print("Pre-normalising inputs and targets...")
        self.x_min = self.all_inputs.min(axis=0)[0]
        self.x_max = self.all_inputs.max(axis=0)[0]
        self.y_min = self.all_targets.min(axis=0)[0]
        self.y_max = self.all_targets.max(axis=0)[0]

        # Note that inputs contain [log_t, log_T0, log_HI0, log_HII0], and outputs [log_T, log_HI, log_HII].
        # Force the inputs at indices 1, 2, 3 to use the same scale as the targets (same physical quantities)
        # In this way, the PINN learns to predict values that are on the same scale as the targets.
        self.x_min[1:] = self.y_min
        self.x_max[1:] = self.y_max

        print("\n--- SCALING DEBUG ---")
        print(f"x_min: {self.x_min.numpy()}")
        print(f"x_max: {self.x_max.numpy()}")
        print(f"y_min: {self.y_min.numpy()}")
        print(f"y_max: {self.y_max.numpy()}")
        print("--- END SCALING DEBUG ---\n")

        self.all_inputs = (self.all_inputs - self.x_min) / (self.x_max - self.x_min + 1e-8)
        self.all_targets = (self.all_targets - self.y_min) / (self.y_max - self.y_min + 1e-8)

        print("\n--- TIME DEBUG: AFTER NORM ---")
        norm_time = self.all_inputs[:, 0]
        print(f"Global Norm log_t Min: {norm_time.min().item():.4f}")
        print(f"Global Norm log_t Max: {norm_time.max().item():.4f}")
        print(f"First 10 norm log_t values:\n{norm_time[:10].numpy()}\n")
        
        print("Dataset ready.")


    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, idx):
        return self.all_inputs[idx], self.all_targets[idx]




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