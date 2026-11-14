import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def map_initial_conditions(csv_path):
    if not os.path.exists(csv_path):
        print(f"[!] Error: '{csv_path}' not found.")
        return

    # Load the clustered data
    df = pd.read_csv(csv_path)
    
    # Convert the log values back to physical units for readable output
    df['T_0'] = 10 ** df['T_0(log)']
    df['nHI_0'] = 10 ** df['nHI_0(log)']
    df['nHII_0'] = 10 ** df['nHII_0(log)']
    
    families = sorted(df['Family_ID'].unique())
    
    print("========================================================")
    print("       INITIAL CONDITION BOUNDARIES PER FAMILY          ")
    print("========================================================")
    
    # 1. Print the mathematical bounding boxes
    for fid in families:
        subset = df[df['Family_ID'] == fid]
        count = len(subset)
        
        print(f"\n[Family {fid}] - {count} trajectories")
        print(f"  T_0    Range : {subset['T_0'].min():.2e} K  to  {subset['T_0'].max():.2e} K")
        print(f"  nHI_0  Range : {subset['nHI_0'].min():.2f} cm^-3   to  {subset['nHI_0'].max():.2f} cm^-3")
        print(f"  nHII_0 Range : {subset['nHII_0'].min():.2f} cm^-3   to  {subset['nHII_0'].max():.2f} cm^-3")

    # 2. Generate the visual scatter plots
    print("\nGenerating phase space scatter plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Styling for the scatter points
    scatter_kws = {'alpha': 0.7, 's': 35, 'edgecolors': 'w', 'linewidth': 0.5}

    for fid in families:
        subset = df[df['Family_ID'] == fid]
        
        axes[0].scatter(subset['T_0'], subset['nHI_0'], label=f'Family {fid}', **scatter_kws)
        axes[1].scatter(subset['T_0'], subset['nHII_0'], label=f'Family {fid}', **scatter_kws)
        axes[2].scatter(subset['nHI_0'], subset['nHII_0'], label=f'Family {fid}', **scatter_kws)

    # Plot 1: T_0 vs nHI_0
    axes[0].set_xlabel('Initial Temperature $T_0$ [K]', fontsize=12)
    axes[0].set_ylabel('Initial Neutral H $n_{HI,0}$ [cm$^{-3}$]', fontsize=12)
    axes[0].set_title('Temperature vs Neutral Hydrogen', fontsize=14)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    # Plot 2: T_0 vs nHII_0
    axes[1].set_xlabel('Initial Temperature $T_0$ [K]', fontsize=12)
    axes[1].set_ylabel('Initial Ionized H $n_{HII,0}$ [cm$^{-3}$]', fontsize=12)
    axes[1].set_title('Temperature vs Ionized Hydrogen', fontsize=14)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')

    # Plot 3: nHI_0 vs nHII_0
    axes[2].set_xlabel('Initial Neutral H $n_{HI,0}$ [cm$^{-3}$]', fontsize=12)
    axes[2].set_ylabel('Initial Ionized H $n_{HII,0}$ [cm$^{-3}$]', fontsize=12)
    axes[2].set_title('Neutral vs Ionized Hydrogen', fontsize=14)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')

    # Formatting
    for ax in axes:
        ax.legend()
        ax.grid(True, which="major", linestyle="-", alpha=0.5)
        ax.grid(True, which="minor", linestyle="--", alpha=0.2)

    plt.tight_layout()
    out_name = "IC_Family_Map.png"
    plt.savefig(out_name, dpi=150)
    print(f"[✓] Scatter plot saved to {out_name}")

if __name__ == "__main__":
    CSV_FILE = "Trajectory_Families.csv"
    map_initial_conditions(CSV_FILE)