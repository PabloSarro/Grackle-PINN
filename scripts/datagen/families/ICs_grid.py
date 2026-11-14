import numpy as np
import os

# Physical bounds (log-space for T and n)
T_min, T_max = 2.0, 8.0
n_min, n_max = -4.0, 2.0

# 0.5 log-steps (e.g., 10^2, 10^2.5, 10^3...)
step = 0.25

# Build grids in log-space
T_grid = np.arange(T_min, T_max + step, step)
n_grid = np.arange(n_min, n_max + step, step)

# Expanded x_grid: Capturing the violent extremes (near 0 and near 1) + the middle
x_grid = [1e-4, 1e-2, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9999]

# Define the Scratch output directory
import sys
out_dir = sys.argv[1] if len(sys.argv) > 1 else "configs/ics"
os.makedirs(out_dir, exist_ok=True)
param_file = os.path.join(out_dir, "ICs_grid.txt")

count = 0
with open(param_file, "w") as f:
    for logT in T_grid:
        for logn in n_grid:
            for xHII in x_grid:
                
                T0 = 10 ** logT
                nH = 10 ** logn

                f.write(f"{T0:.6e} {nH:.6e} {xHII:.6f}\n")
                count += 1

print(f"[✓] Generated {count} parameter combinations.")
print(f"[✓] Saved to: {param_file}")