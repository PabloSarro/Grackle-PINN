import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_grackle_file(filepath):
    if not os.path.exists(filepath):
        print(f"[!] Error: File '{filepath}' not found.")
        return

    print(f"Loading data from {filepath}...")
    
    # Load the data, ignoring the # header rows
    data = pd.read_csv(filepath, comment='#', sep='\s+', header=None)
    
    # Extract variables based on the Grackle output format
    # Col 1: Time, Col 3: T, Col 6: nHI, Col 7: nHII
    t_yr = data.iloc[:, 1].values
    T = data.iloc[:, 3].values
    nHI = data.iloc[:, 6].values
    nHII = data.iloc[:, 7].values

    # Create a clean 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

    plt.tight_layout()
    
    # Save the plot dynamically based on the input file name
    out_name = os.path.splitext(os.path.basename(filepath))[0] + "_profile.png"
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    print(f"[✓] Plot successfully saved to: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot T, nHI, and nHII from a Grackle .dat file")
    parser.add_argument("filepath", type=str, help="Path to the .dat file (e.g., Test_100yr/output_659.dat)")
    args = parser.parse_args()
    
    plot_grackle_file(args.filepath)