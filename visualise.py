import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import PINN

# 1. Setup and Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN(input_dim=4, output_dim=3).to(device)
model.load_state_dict(torch.load("PINN.pth", map_location=device))
model.eval()

# 2. Load and Process output_100.dat
df = pd.read_csv("output_100.dat", comment='#', sep='\s+', header=None)

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
scaling = torch.load("scaling_params.pth", map_location=device)
x_min_train = scaling['x_min'].cpu().numpy()
x_max_train = scaling['x_max'].cpu().numpy()
inputs_norm = (inputs_log - x_min_train) / (x_max_train - x_min_train + 1e-8)

# 3. Model Prediction
inputs_tensor = torch.from_numpy(inputs_norm).float().to(device)
with torch.no_grad():
    preds_log = model(inputs_tensor).cpu().numpy()

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
axes[1].set_ylabel("Density [cm^-3]")
axes[1].legend()

# Plot 3: Ionized Hydrogen
axes[2].plot(time_lin, HII_lin, 'k--', label='Grackle GT')
axes[2].scatter(time_lin, HII_pred, color='red', s=10, label='PINN Prediction')
axes[2].set_yscale('log')
axes[2].set_xscale('log')
axes[2].set_title("Ionized Hydrogen (nHII)")
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Density [cm^-3]")
axes[2].legend()

plt.tight_layout()
plt.savefig("pinn_performance.png")
print("Visualisation saved as pinn_performance.png")
plt.show()