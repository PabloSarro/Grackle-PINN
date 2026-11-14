import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
source_dir = Path("/scratch/izar/sarro/Full_Grackle_Data/")
dest_dir = Path("/scratch/izar/sarro/Active_Grackle_Data/")
dest_dir.mkdir(parents=True, exist_ok=True)

# List all relevant files
all_files = list(source_dir.glob("output_*.dat"))

def get_activity_score(filepath):
    """
    Computes activity score: sum of absolute log-increments
    across Temperature (col 3), nHI (col 6), and nHII (col 7).
    """
    try:
        data = pd.read_csv(filepath, comment='#', sep='\s+', header=None)
        # Extract variables
        T = data.iloc[:, 3].values.astype(np.float64)
        nHI = data.iloc[:, 6].values.astype(np.float64)
        nHII = data.iloc[:, 7].values.astype(np.float64)
        
        # Compute absolute sum of log-increments
        score = (np.sum(np.abs(np.diff(np.log(T)))) + 
                 np.sum(np.abs(np.diff(np.log(nHI)))) + 
                 np.sum(np.abs(np.diff(np.log(nHII)))))
        return score
    except Exception:
        return -1.0 # Penalize failed files

print(f"Calculating activity scores for {len(all_files)} files...")
scores = []
for f in all_files:
    scores.append((f, get_activity_score(f)))

# Sort descending by score
scores.sort(key=lambda x: x[1], reverse=True)

# Select top 3000
top_3000 = scores[:3000]

print(f"Moving top 3000 active files to {dest_dir}...")
for f, score in top_3000:
    shutil.copy(f, dest_dir / f.name)

print("Process complete.")