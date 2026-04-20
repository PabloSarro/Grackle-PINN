import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import PINN

# Define the model, parameter file and validation file.
MODEL_NAME = "PINN.pth"
VALID_FILES = ["Old_Grackle_GTs/output_2.dat", "Old_Grackle_GTs/output_3.dat", "Old_Grackle_GTs/output_4.dat"]
PLOT_NAMES = ["pinn_performance_2.png", "pinn_performance_3.png", "pinn_performance_4.png"]

# 1. Setup and Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
model.eval()

for VALID_FILE, PLOT_NAME in zip(VALID_FILES, PLOT_NAMES):
    # 2. Load and Process the validation file
    df = pd.read_csv(VALID_FILE, comment='#', sep='\s+', header=None)

    # Extract Ground Truth (GT)
    time_lin = df.iloc[:, 1].values.astype(np.float32)
    T_lin    = df.iloc[:, 3].values.astype(np.float32)
    HI_lin   = df.iloc[:, 6].values.astype(np.float32)
    HII_lin  = df.iloc[:, 7].values.astype(np.float32)

    # Prepare PINN Inputs: [log_t, log_T0, log_HI0, log_HII0]
    log_t = np.log10(np.maximum(time_lin, 1e-20))
    log_T0   = np.full_like(log_t, np.log10(np.maximum(T_lin[0], 1e-20)))
    log_HI0  = np.full_like(log_t, np.log10(np.maximum(HI_lin[0], 1e-20)))
    log_HII0 = np.full_like(log_t, np.log10(np.maximum(HII_lin[0], 1e-20)))

    inputs_log = np.stack([log_t, log_T0, log_HI0, log_HII0], axis=1)

    # Normalisation
    scaling = torch.load(f"scaling_params_{MODEL_NAME}", map_location=device)
    x_min_train = scaling['x_min'].cpu().numpy()
    x_max_train = scaling['x_max'].cpu().numpy()
    y_min_train = scaling['y_min'].cpu().numpy()
    y_max_train = scaling['y_max'].cpu().numpy()
    inputs_norm = (inputs_log - x_min_train) / (x_max_train - x_min_train + 1e-8)

    print(f"Input Max values: {x_max_train}; Min values: {x_min_train}")
    print(f"Target Max values: {y_max_train}; Min values: {y_min_train}")

    # 3. Model Prediction
    inputs_tensor = torch.from_numpy(inputs_norm).float().to(device)
    with torch.no_grad():
        print(f"Normalized Time Range: {inputs_norm[:, 0].min():.4f} to {inputs_norm[:, 0].max():.4f}")
        preds_norm = model(inputs_tensor).cpu().numpy()

    preds_log = preds_norm * (y_max_train - y_min_train + 1e-8) + y_min_train

    # Convert back to linear scale
    T_pred   = 10**preds_log[:, 0]
    HI_pred  = 10**preds_log[:, 1]
    HII_pred = 10**preds_log[:, 2]

    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Temperature
    axes[0].plot(time_lin, T_lin, 'k--', label='Grackle GT')
    axes[0].scatter(time_lin, T_pred, color='orange', s=10, label='PINN Prediction')
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].set_title("Temperature (T)")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("T [K]")
    axes[0].legend()

    # Plot 2: Neutral Hydrogen
    axes[1].plot(time_lin, HI_lin, 'k--', label='Grackle GT')
    axes[1].scatter(time_lin, HI_pred, color='blue', s=10, label='PINN Prediction')
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_title("Neutral Hydrogen (nHI)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"Density [cm$^{-3}$]")
    axes[1].legend()

    # Plot 3: Ionized Hydrogen
    axes[2].plot(time_lin, HII_lin, 'k--', label='Grackle GT')
    axes[2].scatter(time_lin, HII_pred, color='red', s=10, label='PINN Prediction')
    axes[2].set_yscale('log')
    axes[2].set_xscale('log')
    axes[2].set_title("Ionized Hydrogen (nHII)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel(r"Density [cm$^{-3}$]")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(PLOT_NAME)
    print(f"Visualisation saved as {PLOT_NAME}")
    plt.show()