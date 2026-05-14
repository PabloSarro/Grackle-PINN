import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import PINN

m_H = 1.6726219e-24  # Hydrogen Mass

# Prepare model container
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)

# Load Standardization parameters
scaling = torch.load(f"scaling_params_PINN.pth", map_location=device)
in_mean = scaling['in_mean'].cpu().numpy() # [dt, log_T, log_HI, log_HII]
in_std  = scaling['in_std'].cpu().numpy()
tg_mean = scaling['tg_mean'].cpu().numpy() # [dlog_T, dlog_HI, dlog_HII]
tg_std  = scaling['tg_std'].cpu().numpy()

# Define the model, parameter file and validation file.
MODEL_NAMES = ["PINN.pth", "PINN_BEST.pth"]
for MODEL_NAME in MODEL_NAMES:

    VALID_FILES = [f"Test_100yr/output_{i}.dat" for i in range(1,2)]
    PLOT_NAMES = [f"pinn_performance_{MODEL_NAME}_{i}.png" for i in range(1,2)]

    # 1. Setup and Load Model
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
    model.eval()

    for VALID_FILE, PLOT_NAME in zip(VALID_FILES, PLOT_NAMES):
        with open(VALID_FILE, 'r') as header_file:
            head = [header_file.readline() for _ in range(3)]

        mass_units = float(head[1].split(':')[1].split('[')[0])
        length_units = float(head[2].split(':')[1].split('[')[0])
        density_units = mass_units / (length_units**3)

        # 2. Load and Process the validation file
        df = pd.read_csv(VALID_FILE, comment='#', sep='\s+', header=None)

        # 2. Extract Ground Truth (Ignoring Row 0 as decided)
        # Col 1: Time, Col 2: dt, Col 3: T, Col 6: HI, Col 7: HII
        t_yr = df.iloc[1:, 1].values
        dt_yr = df.iloc[1:, 2].values
        T_gt = df.iloc[1:, 3].values
        HI_gt = df.iloc[1:, 6].values * density_units / m_H
        HII_gt = df.iloc[1:, 7].values * density_units / m_H

        # 3. Iterative PINN Prediction
        # Initialize predictions with the first valid GT point
        T_pred, HI_pred, HII_pred = [T_gt[0]], [HI_gt[0]], [HII_gt[0]]
        
        curr_T, curr_HI, curr_HII = T_gt[0], HI_gt[0], HII_gt[0]

        for i in range(len(dt_yr) - 1):
            # Prepare Input: [dt, log(T), log(nHI), log(nHII)]
            # Use np.log() (natural log) to match your training code
            raw_in = np.array([
                dt_yr[i], 
                np.log(curr_T), 
                np.log(curr_HI), 
                np.log(curr_HII)
            ])
            
            # Standardize using the mean/std from training
            norm_in = (raw_in - in_mean) / (in_std + 1e-8)
            input_tensor = torch.from_numpy(norm_in).float().to(device).unsqueeze(0)

            with torch.no_grad():
                pred_norm = model(input_tensor).cpu().numpy()[0]

            # Destandardize prediction: this is dlog_y = log(y_next/y_curr)
            dlog_y = pred_norm * (tg_std + 1e-8) + tg_mean
            
            # Update current state for the next iteration
            curr_T   *= np.exp(dlog_y[0])
            curr_HI  *= np.exp(dlog_y[1])
            curr_HII *= np.exp(dlog_y[2])

            T_pred.append(curr_T)
            HI_pred.append(curr_HI)
            HII_pred.append(curr_HII)

        # 4. Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        plots = [
            (T_gt, T_pred, 'orange', 'Temperature', 'T [K]'),
            (HI_gt, HI_pred, 'blue', 'HI Density', 'n [cm^-3]'),
            (HII_gt, HII_pred, 'red', 'HII Density', 'n [cm^-3]')
        ]

        for ax, (gt, pred, col, title, ylabel) in zip(axes, plots):
            ax.plot(t_yr, gt, 'k--', alpha=0.5, label='Grackle GT')
            ax.plot(t_yr, pred, color=col, label='PINN Predicted')
            ax.set_yscale('log')
            ax.set_title(title)
            ax.set_xlabel("Time [yr]")
            ax.set_ylabel(ylabel)
            ax.legend()

        plt.tight_layout()
        plt.savefig(PLOT_NAME)
        print(f"Visualisation saved as {PLOT_NAME}")