import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_family_overlays(csv_path, data_folder, max_lines_per_plot=20):
    if not os.path.exists(csv_path):
        print(f"[!] Error: {csv_path} not found.")
        return

    # Load the clustering results
    df = pd.read_csv(csv_path)
    families = df['Family_ID'].unique()
    
    # Create an output directory for the visualisations
    out_dir = "Family_Plots"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Found {len(families)} distinct families. Generating overlay plots...")

    for fid in sorted(families):
        # Isolate files belonging to this family
        family_df = df[df['Family_ID'] == fid]
        total_in_family = len(family_df)
        
        # Sample files to avoid plotting hundreds of overlapping lines
        sample_size = min(total_in_family, max_lines_per_plot)
        sampled_files = family_df['Filename'].sample(sample_size, random_state=7000).tolist()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        if (fid==0):
            fig.suptitle(f"Family {fid}: Hot and Ionized Equilibrium", fontsize=20)
        elif (fid==1):
            fig.suptitle(f"Family {fid}: Low-Density Static Equilibrium", fontsize=20)
        elif (fid==2):
            fig.suptitle(f"Family {fid}: High-Density Static Equilibrium", fontsize=20)
        else:
            fig.suptitle(f"Family {fid}: Rapid Recombination and Cooling", fontsize=20)

        footnote = f"(Showing {sample_size} of {total_in_family} trajectories)"
        fig.text(0.5, -0.04, footnote, ha='center', fontsize=14, style='italic')

        # Plot each sampled trajectory
        for filename in sampled_files:
            filepath = os.path.join(data_folder, filename)
            
            if not os.path.exists(filepath):
                continue
                
            data = np.loadtxt(filepath, comments='#')
            if len(data) < 2:
                continue
                
            t_yr = data[1:, 1]
            T = data[1:, 3]
            nHI = data[1:, 6]
            nHII = data[1:, 7]

            # Use alpha=0.3 to create a density map effect where lines overlap
            axes[0].plot(t_yr, T, color='orange', alpha=0.3, linewidth=1.5)
            axes[1].plot(t_yr, nHI, color='blue', alpha=0.3, linewidth=1.5)
            axes[2].plot(t_yr, nHII, color='red', alpha=0.3, linewidth=1.5)

        # Formatting
        titles = ['Temperature [K]', 'Neutral Hydrogen [$cm^{-3}$]', 'Ionized Hydrogen [$cm^{-3}$]']
        for i, ax in enumerate(axes):
            ax.set_yscale('log')
            ax.set_title(titles[i], fontsize=14)
            ax.set_xlabel('Time [yr]', fontsize=12)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"Family_{fid}_envelope.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[✓] Saved {out_path}")

if __name__ == "__main__":
    CSV_FILE = "Trajectory_Families.csv"
    DATA_FOLDER = "/scratch/izar/sarro/Grid_Analysis/Grid_Files/" # Adjust if your .dat files are elsewhere
    
    plot_family_overlays(CSV_FILE, DATA_FOLDER)