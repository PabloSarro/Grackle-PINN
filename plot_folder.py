import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_grackle_file(filepath, out_dir):
    print(f"Loading data from {filepath}...")
    
    # Load the data, ignoring the # header rows
    try:
        data = pd.read_csv(filepath, comment='#', sep='\s+', header=None)
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")
        return
    
    # Extract variables based on the Grackle output format
    # Col 1: Time, Col 3: T, Col 6: nHI, Col 7: nHII
    t_yr = data.iloc[:, 1].values
    T = data.iloc[:, 3].values
    nHI = data.iloc[:, 6].values
    nHII = data.iloc[:, 7].values

    try:
        T0 = data.iloc[0, 3]
        nHI0 = data.iloc[0, 6]
        nHII0 = data.iloc[0, 7]
    except Exception as e:
        print(f"[!] Could not extract ICs: {e}")
        T0 = nHI0 = nHII0 = np.nan

    # Create a clean 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Plot Temperature
    axes[0].plot(t_yr, T, color='orange', linewidth=2)
    axes[0].set_title('Temperature Evolution', fontsize=14, pad=10)
    axes[0].set_xlabel('Time [yr]', fontsize=12)
    axes[0].set_ylabel('T [K]', fontsize=12)
    
    # 2. Plot nHI
    axes[1].plot(t_yr, nHI, color='blue', linewidth=2)
    axes[1].set_title('Neutral Hydrogen ($n_{HI}$)', fontsize=14, pad=10)
    axes[1].set_xlabel('Time [yr]', fontsize=12)
    axes[1].set_ylabel('$n_{HI}$ [$cm^{-3}$]', fontsize=12)

    # 3. Plot nHII
    axes[2].plot(t_yr, nHII, color='red', linewidth=2)
    axes[2].set_title('Ionized Hydrogen ($n_{HII}$)', fontsize=14, pad=10)
    axes[2].set_xlabel('Time [yr]', fontsize=12)
    axes[2].set_ylabel('$n_{HII}$ [$cm^{-3}$]', fontsize=12)

    # Apply consistent styling across all axes
    for ax in axes:
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=10)

    fig.suptitle(
        f"T₀ = {T0:.3e} K | nHI₀ = {nHI0:.3e} cm⁻³ | nHII₀ = {nHII0:.3e} cm⁻³",
        fontsize=14
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the plot dynamically based on the input file name
    out_name = os.path.splitext(os.path.basename(filepath))[0] + "_profile.png"
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    
    # Clear the figure from memory to prevent RAM bloat during batch processing
    plt.close(fig) 
    
    print(f"[✓] Plot successfully saved to: {out_path}")

def process_folder(folderpath):
    if not os.path.isdir(folderpath):
        print(f"[!] Error: Directory '{folderpath}' not found.")
        return

    # Define and create the Plots_Subfolder inside the target directory
    out_dir = os.path.join(folderpath, "Plots_Subfolder")
    os.makedirs(out_dir, exist_ok=True)

    # Find all .dat files in the folder
    dat_files = glob.glob(os.path.join(folderpath, "*.dat"))
    
    if not dat_files:
        print(f"[!] No .dat files found in '{folderpath}'.")
        return

    print(f"Found {len(dat_files)} .dat files. Processing...")
    
    for filepath in sorted(dat_files):
        plot_grackle_file(filepath, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch plot T, nHI, and nHII from Grackle .dat files")
    parser.add_argument("folderpath", type=str, help="Path to the folder containing the .dat files")
    args = parser.parse_args()
    
    process_folder(args.folderpath)