import torch
import numpy as np
import pandas as pd

# 1. Load the specific file and scaling parameters
file_path = "Grackle_GTs/output_10.dat" # Change to your target file
params = torch.load("scaling_params_PINN.pth")
x_min = params['x_min'][0].item() # Time min
x_max = params['x_max'][0].item() # Time max

# 2. Extract and Transform
data = pd.read_csv(file_path, comment='#', sep='\s+', header=None)
t_lin = data.iloc[:, 1].values.astype(np.float32)

# Apply the Dynamic Time Floor logic
if t_lin[0] == 0.0 and len(t_lin) > 1:
    t_lin[0] = t_lin[1] / 10.0

# Log-space transformation
t_log = np.log10(t_lin)

# Normalization using the GLOBAL training bounds
t_norm = (t_log - x_min) / (x_max - x_min + 1e-8)

# 3. Print Analysis
print(f"--- Time Distribution for {file_path} ---")
print(f"Total points: {len(t_norm)}")
print(f"First 5 normalized times: {t_norm[:5]}")
print(f"Last 5 normalized times:  {t_norm[-5:]}")

# Check density: How many points are in the last 20% of the time domain?
high_val_count = np.sum(t_norm > 0.9)
percentage = (high_val_count / len(t_norm)) * 100

print(f"\nPoints with t_norm > 0.9: {high_val_count} ({percentage:.2f}%)")

if percentage > 80:
    print("\n[!] CONFIRMED: Your points are heavily concentrated at the end of the simulation.")
    print("The model is likely suffering from 'Steady-State Bias'.")