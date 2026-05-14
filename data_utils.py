import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
import os

m_H = 1.6726219e-24  # Hydrogen Mass

class GrackleDataset(Dataset):
    def __init__(self, folder_path, pattern="output_*.dat"):
        """
        Load multiple Grackle simulations (will be our inputs).
        PINN's input  (x): [dt, log(y(t))] (and standardised), for y = T, nHI, nHII
        PINN's output (y): [log(y(t+dt)/y(t))] (and standardised), for y = T, nHI, nHII
        """
        self.all_inputs = []
        self.all_targets = []
        
        # Search for all files that match the pattern
        files = glob.glob(os.path.join(folder_path, pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching the pattern {pattern} in {folder_path}")

        # For each file
        for f in files:
            # Extract the units used
            with open(f, 'r') as header_file:
                # Read only the first 4 lines
                head = [header_file.readline() for _ in range(3)]
            
            mass_units = float(head[1].split(':')[1].split('[')[0]) # should be 1.98841e43
            length_units = float(head[2].split(':')[1].split('[')[0]) # should be 3.08567758e24
            density_units = mass_units / (length_units**3)
            
            # Extract the data
            data = pd.read_csv(f, comment='#', sep='\s+', header=None)

            # // ---- Inputs [dt, log(T(t)), log(nHI(t)), log(nHII(t))] for every timestep ---- \\
            # // ---- Targets [log(T(t+dt)/T(t)), log(nHI(t+dt)/nHI(t)), log(nHII(t+dt)/nHII(t))] for every timestep ---- \\
            
            # Extract its state vectors, and convert to physical units
            dt = data.iloc[1:, 2].values.astype(np.float64) # Vector of dt's (ignoring the initial dt)
            T = data.iloc[:, 3].values.astype(np.float64)
            nHI = data.iloc[:, 6].values.astype(np.float64) * density_units / m_H
            nHII = data.iloc[:, 7].values.astype(np.float64) * density_units / m_H
                        
            T_curr = T[:-1]            # T(t), for all t (except last one)
            T_next = T[1:]             # T(t+dt), for all t (besides first)
            T_targ = np.log(T_next)-np.log(T_curr)     # log(T(t+dt)/T(t))
            
            nHI_curr = nHI[:-1]        # nHI(t)
            nHI_next = nHI[1:]         # nHI(t+dt)
            nHI_targ = np.log(nHI_next)-np.log(nHI_curr)  # log(nHI(t+dt)/nHI(t))
            
            nHII_curr = nHII[:-1]      # nHII(t)
            nHII_next = nHII[1:]       # nHII(t+dt)
            nHII_targ = np.log(nHII_next)-np.log(nHII_curr)  # log(nHII(t+dt)/nHII(t))

            # Stack into (N, 4) and (N, 3), respectively.
            inputs = np.stack([dt, np.log(T_curr), np.log(nHI_curr), np.log(nHII_curr)], axis=1)
            targets = np.stack([T_targ, nHI_targ, nHII_targ], axis=1)

            self.all_inputs.append(inputs)
            self.all_targets.append(targets)

        self.all_inputs = torch.from_numpy(np.concatenate(self.all_inputs, axis=0)).float()
        self.all_targets = torch.from_numpy(np.concatenate(self.all_targets, axis=0)).float()

        print("Pre-normalising inputs and targets...")        
        # No need to normalise time.
        #   self.dt_min = dt.min()
        #   self.dt_max = dt.max()
        self.in_mean = self.all_inputs.mean(dim=0)
        self.in_std = self.all_inputs.std(dim=0)

        self.tg_mean = self.all_targets.mean(dim=0)
        self.tg_std = self.all_targets.std(dim=0)

        print("\n--- SCALING (MINS/MAXS) ---")
        print("inputs_mean:", self.in_mean)
        print("inputs_std:", self.in_std)
        print("targets_mean:", self.tg_mean)
        print("targets_std:", self.tg_std)
        print("--- END SCALING ---\n")

        self.all_inputs = (self.all_inputs - self.in_mean) / (self.in_std + 1e-8)
        self.all_targets = (self.all_targets - self.tg_mean) / (self.tg_std + 1e-8)

        print("Dataset ready.")


    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, idx):
        return self.all_inputs[idx], self.all_targets[idx]


def DEBUG_helper(name, t, min_val=1e-10, max_val=1e10):
    if torch.isnan(t).any():
        print(f"[!] FATAL: '{name}' contains NaNs.")
    elif torch.isinf(t).any():
        print(f"[!] FATAL: '{name}' contains Inf.")
    # elif t.max() > max_val:
    #     print(f"[!] WARNING: '{name}' is too large! Max: {t.max():.2e}")
    # elif t.min() < min_val:
    #     print(f"[!] WARNING: '{name}' is too small! Min: {t.min():.2e}")


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