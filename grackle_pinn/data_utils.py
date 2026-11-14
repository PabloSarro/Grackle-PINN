import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
import os

m_H = 1.6726219e-24  # Hydrogen Mass

class GrackleDataset(Dataset):
    def __init__(self, folder_path, pattern="output_*.dat", in_mean=None, in_std=None, tg_mean=None, tg_std=None,
                 param=None):
        """
        Load multiple Grackle simulations (will be our inputs).
        With the default `report` parameterisation:
        PINN's input  (x): [log(y(t))] (and standardised), for y = T, nHI, nHII
        PINN's output (y): [log(y(t+dt)/y(t))] (and standardised), for y = T, nHI, nHII
        Other parameterisations (grackle_pinn.param) define their own features and targets from the same
        log-states. `folder_path` is a folder of text files or a compact .npz (see scripts/datasets).
        """
        from grackle_pinn.param import Report
        param = param or Report()
        self.param = param
        self.logstate = None   # (n_traj, n_rows, 3) when loaded from .npz; used for rollout-based selection

        if str(folder_path).endswith(".npz"):
            # Compact format (A3): logstate (n_traj, n_rows, 3) = ln [T, nHI, nHII]; adjacent rows form one step.
            self.logstate = np.load(folder_path)["logstate"]
            trajs = list(self.logstate)
        else:
            files = glob.glob(os.path.join(folder_path, pattern))
            if not files:
                raise FileNotFoundError(f"No files found matching the pattern {pattern} in {folder_path}")
            trajs = []
            for f in files:
                data = pd.read_csv(f, comment='#', sep=r'\s+', header=None)
                trajs.append(np.log(data.iloc[:, [3, 6, 7]].values.astype(np.float64)))

        inputs, targets = [], []
        for ls in trajs:
            cur = torch.from_numpy(ls[:-1])
            nxt = torch.from_numpy(ls[1:])
            inputs.append(param.features(cur))
            targets.append(param.targets(cur, nxt))
        self.all_inputs = torch.cat(inputs).to(torch.get_default_dtype())
        self.all_targets = torch.cat(targets).to(torch.get_default_dtype())

        self.target_absmax = self.all_targets.abs().max(dim=0).values

        print("Pre-normalising inputs and targets...")
        # If parameters are not provided, compute them (done for the Training set)
        if in_mean is None:
            self.in_mean = self.all_inputs.mean(dim=0)
            self.in_std = self.all_inputs.std(dim=0)

            self.tg_mean = self.all_targets.mean(dim=0)
            self.tg_std = self.all_targets.std(dim=0)
        # Otherwise, use them directly (done for the Validation set)
        else:
            self.in_mean = in_mean
            self.in_std = in_std
            self.tg_mean = tg_mean
            self.tg_std = tg_std

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