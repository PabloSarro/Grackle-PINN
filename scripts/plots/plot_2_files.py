import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

def plot_comparison_vertical(file_top, file_bottom):
    files = [file_top, file_bottom]
    
    # Check if both files exist
    for f in files:
        if not os.path.exists(f):
            print(f"[!] Error: File '{f}' not found.")
            return

    # Create a 6x1 vertical grid for portrait document insertion
    fig, axes = plt.subplots(6, 1, figsize=(8, 20), sharex=True)
    
    MIN_DECADE_SPAN = 2.0 

    for file_idx, filepath in enumerate(files):
        print(f"Loading data from {filepath}...")
        data = pd.read_csv(filepath, comment='#', sep='\s+', header=None)
        
        # Extract variables
        t_yr = data.iloc[:, 1].values
        T = data.iloc[:, 3].values
        nHI = data.iloc[:, 6].values
        nHII = data.iloc[:, 7].values

        file_label = os.path.basename(filepath)
        
        # Calculate the starting row index for the current file (0 for Top, 3 for Bottom)
        row_offset = file_idx * 3

        # 1. Plot Temperature
        axes[row_offset].plot(t_yr, T, color='orange', linewidth=2)
        axes[row_offset].set_ylabel('T [K]', fontsize=12)
        axes[row_offset].set_title(f'Temperature Evolution ({file_label})', fontsize=13, pad=8)
        
        # 2. Plot nHI
        axes[row_offset + 1].plot(t_yr, nHI, color='blue', linewidth=2)
        axes[row_offset + 1].set_ylabel('$n_{HI}$ [$cm^{-3}$]', fontsize=12)
        axes[row_offset + 1].set_title(f'Neutral Hydrogen ({file_label})', fontsize=13, pad=8)

        # 3. Plot nHII
        axes[row_offset + 2].plot(t_yr, nHII, color='red', linewidth=2)
        axes[row_offset + 2].set_ylabel('$n_{HII}$ [$cm^{-3}$]', fontsize=12)
        axes[row_offset + 2].set_title(f'Ionized Hydrogen ({file_label})', fontsize=13, pad=8)

        # Apply consistent styling and mathematically enforce flatness limits
        for local_idx, data_array in enumerate([T, nHI, nHII]):
            ax = axes[row_offset + local_idx]
            ax.set_yscale('log')
            
            # Calculate base-10 log spread
            log_min, log_max = np.log10(np.min(data_array)), np.log10(np.max(data_array))
            
            # Enforce minimum y-axis span to flatten equilibrium plots visually
            if (log_max - log_min) < MIN_DECADE_SPAN:
                mid = (log_max + log_min) / 2.0
                ax.set_ylim(10**(mid - (MIN_DECADE_SPAN/2.0)), 10**(mid + (MIN_DECADE_SPAN/2.0)))
            
            # Force major grid lines exactly 1 order of magnitude apart
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            
            ax.grid(True, which="major", linestyle="-", alpha=0.7)
            ax.grid(True, which="minor", linestyle="--", alpha=0.3)
            ax.tick_params(labelsize=10)

    # Only add the X-axis label to the very bottom plot
    axes[5].set_xlabel('Time [yr]', fontsize=14)

    # Adjust vertical spacing so titles and axes do not overlap
    plt.subplots_adjust(hspace=0.4)
    
    out_name = "comparison_profile_vertical.png"
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    print(f"[✓] Plot successfully saved to: {out_name}")

if __name__ == "__main__":
    # Hardcoded to exactly the two files you requested
    FILE_TOP = "Active_Files/train/output_1.dat"
    FILE_BOTTOM = "/scratch/izar/sarro/FewFiles_Train_Data/output_140.dat"
    
    plot_comparison_vertical(FILE_TOP, FILE_BOTTOM)