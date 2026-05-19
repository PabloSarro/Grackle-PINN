import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import PINN

m_H = 1.6726219e-24  # Hydrogen Mass

def visualise_preds(VALID_FILE):
    if not os.path.exists(VALID_FILE):
        print(f"[!] Error: File '{VALID_FILE}' not found.")
        return

    # Prepare model container
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(input_dim=4, output_dim=3, hidden_dim=256, hidden_layers=6).to(device)

    # Load Standardization parameters
    scaling = torch.load(f"scaling_params.pth", map_location=device, weights_only=True)
    in_mean = scaling['in_mean'].cpu().numpy() # [dt, log_T, log_HI, log_HII]
    in_std  = scaling['in_std'].cpu().numpy()
    tg_mean = scaling['tg_mean'].cpu().numpy() # [dlog_T, dlog_HI, dlog_HII]
    tg_std  = scaling['tg_std'].cpu().numpy()

    # Define the model, parameter file and validation file.
    MODEL_NAMES = ["PINN.pth", "PINN_BEST.pth"]
    for MODEL_NAME in MODEL_NAMES:

        if not os.path.exists(MODEL_NAME):
            print(f"Skipping {MODEL_NAME}, file not found.")
            continue

        # 1. Setup and Load Model
        model.load_state_dict(torch.load(MODEL_NAME, map_location=device, weights_only=True)) 
        model.eval()

        # 2. Load and Process the validation file
        df = pd.read_csv(VALID_FILE, comment='#', sep='\s+', header=None)

        # Extract Ground Truth (Ignoring Row 0 as decided)
        t_yr = df.iloc[1:, 1].values
        dt_yr = df.iloc[1:, 2].values
        T_gt = df.iloc[1:, 3].values
        HI_gt = df.iloc[1:, 6].values
        HII_gt = df.iloc[1:, 7].values

        # 3. Compute True Log-Increments (Ground Truth Targets)
        dlog_T_gt = np.log(T_gt[1:]) - np.log(T_gt[:-1])
        dlog_HI_gt = np.log(HI_gt[1:]) - np.log(HI_gt[:-1])
        dlog_HII_gt = np.log(HII_gt[1:]) - np.log(HII_gt[:-1])

        # 4. Compute PINN Predictions (Vectorized Teacher Forcing for Log-Increments)
        raw_in = np.stack([
            dt_yr[:-1], 
            np.log(T_gt[:-1]), 
            np.log(HI_gt[:-1]), 
            np.log(HII_gt[:-1])
        ], axis=1)

        norm_in = (raw_in - in_mean) / (in_std + 1e-8)
        input_tensor = torch.from_numpy(norm_in).float().to(device)

        with torch.no_grad():
            pred_norm = model(input_tensor).cpu().numpy()

        preds_dlog = pred_norm * (tg_std + 1e-8) + tg_mean
        dlog_T_pred = preds_dlog[:, 0]
        dlog_HI_pred = preds_dlog[:, 1]
        dlog_HII_pred = preds_dlog[:, 2]

        # 5. Compute Autoregressive Rollout (Pure PyTorch GPU Loop)
        num_steps = len(dt_yr)
        
        # Keep scaling parameters natively on GPU as tensors
        t_in_mean = scaling['in_mean'].float().to(device)
        t_in_std = scaling['in_std'].float().to(device)
        t_tg_mean = scaling['tg_mean'].float().to(device)
        t_tg_std = scaling['tg_std'].float().to(device)
        
        # Pre-load all timesteps to GPU
        dt_tensor = torch.tensor(dt_yr, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Pre-allocate the entire trajectory on the GPU in log-space
        # Shape: (num_steps, 3) representing [log(T), log(HI), log(HII)]
        trajectory_log = torch.zeros((num_steps, 3), dtype=torch.float32, device=device)
        trajectory_log[0, 0] = np.log(T_gt[0])
        trajectory_log[0, 1] = np.log(HI_gt[0])
        trajectory_log[0, 2] = np.log(HII_gt[0])
        
        model.eval() # Ensure eval mode
        
        with torch.no_grad():
            for i in range(num_steps - 1):
                # Pack input: [dt, log_T, log_HI, log_HII] natively on GPU
                curr_state = trajectory_log[i:i+1, :]  # Shape (1, 3)
                curr_dt = dt_tensor[i:i+1, :]          # Shape (1, 1)
                raw_in_ar = torch.cat([curr_dt, curr_state], dim=1)
                
                # Normalize, Forward Pass, Denormalize
                norm_in_ar = (raw_in_ar - t_in_mean) / (t_in_std + 1e-8)
                pred_norm_ar = model(norm_in_ar)
                dlog_y_ar = pred_norm_ar * (t_tg_std + 1e-8) + t_tg_mean
                
                # Autoregressive Update: Next state = Current state + log increment
                trajectory_log[i+1:i+2, :] = curr_state + dlog_y_ar

        # ONE single transfer back to CPU and exp() conversion at the very end
        trajectory_phys = torch.exp(trajectory_log).cpu().numpy()
        
        T_pred_ar = trajectory_phys[:, 0]
        HI_pred_ar = trajectory_phys[:, 1]
        HII_pred_ar = trajectory_phys[:, 2]


        # 6. Plotting 3x3 Grid
        fig, axes = plt.subplots(3, 3, figsize=(18, 15  ))

        # --- ROW 1: Log-Increments (Teacher Forcing) ---
        t_plot = t_yr[:-1]
        plots_log = [
            (dlog_T_gt, dlog_T_pred, 'orange', 'Temperature Log-Increment', 'log(T_next / T_curr)'),
            (dlog_HI_gt, dlog_HI_pred, 'blue', 'HI Density Log-Increment', 'log(HI_next / HI_curr)'),
            (dlog_HII_gt, dlog_HII_pred, 'red', 'HII Density Log-Increment', 'log(HII_next / HII_curr)')
        ]

        for ax, (gt, pred, col, title, ylabel) in zip(axes[0], plots_log):
            ax.plot(t_plot, gt, 'k--', alpha=0.5, label='Grackle GT')
            ax.plot(t_plot, pred, color=col, alpha=0.9, label='PINN 1-Step Pred')
            ax.set_title(title)
            ax.set_xlabel("Time [yr]")
            ax.set_ylabel(ylabel)
            ax.legend()

        # --- ROW 2: Absolute Quantities (Autoregressive Rollout) ---
        plots_phys = [
            (T_gt, T_pred_ar, 'orange', 'Temperature Evolution', 'T [K]'),
            (HI_gt, HI_pred_ar, 'blue', 'Neutral Hydrogen ($n_{HI}$)', 'n [cm^-3]'),
            (HII_gt, HII_pred_ar, 'red', 'Ionized Hydrogen ($n_{HII}$)', 'n [cm^-3]')
        ]

        for ax, (gt, pred, col, title, ylabel) in zip(axes[1], plots_phys):
            ax.plot(t_yr, gt, 'k--', alpha=0.5, label='Grackle GT')
            ax.plot(t_yr, pred, color=col, alpha=0.9, label='PINN AR Rollout')
            ax.set_yscale('log')
            ax.set_title(title)
            ax.set_xlabel("Time [yr]")
            ax.set_ylabel(ylabel)
            ax.legend()

        # --- ROW 3: True vs Predicted Scatter (x=y Plot) ---
        plots_scatter = [
            (T_gt, T_pred_ar, 'orange', 'Temperature: True vs Pred', 'True T [K]', 'Pred T [K]'),
            (HI_gt, HI_pred_ar, 'blue', 'Neutral Hydrogen: True vs Pred', 'True $n_{HI}$ [cm^-3]', 'Pred $n_{HI}$ [cm^-3]'),
            (HII_gt, HII_pred_ar, 'red', 'Ionized Hydrogen: True vs Pred', 'True $n_{HII}$ [cm^-3]', 'Pred $n_{HII}$ [cm^-3]')
        ]

        for ax, (gt, pred, col, title, xlabel, ylabel) in zip(axes[2], plots_scatter):
            ax.scatter(gt, pred, color=col, alpha=0.4, s=15, label='AR Predictions')
            
            # Draw the y=x line
            min_val = min(np.min(gt), np.min(pred))
            max_val = max(np.max(gt), np.max(pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='y = x')
            
            # Log scale is crucial since the variables span many orders of magnitude
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()

        plt.tight_layout()
        
        # Save mechanism
        if 'BEST' not in MODEL_NAME:
            out_name = 'test_fig.png'
        else:
            out_name = 'test_fig_BEST.png'
            
        plt.savefig(out_name, dpi=150)
        print(f"Visualisation saved as '{out_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot T, nHI, and nHII from a Grackle .dat file")
    parser.add_argument("filepath", type=str, help="Path to the .dat file (e.g., Test_100yr/output_151.dat)")
    args = parser.parse_args()
    
    visualise_preds(args.filepath)