import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global Configuration
LAMBDA_MAX = 1e7

source_dir = Path("ResultsToDownload")
loss_files = list(source_dir.glob("losses_*.txt"))

if not loss_files:
    print(f"No files matching 'losses_*.txt' found in {source_dir}")
else:
    print(f"Found {len(loss_files)} files. Starting adaptive plotting routine...")

for file_path in loss_files:
    try:
        # Read the file
        df = pd.read_csv(file_path, sep=r"\s+")
        
        # Dynamically retrieve total epochs from the final row of the Epoch column
        total_epochs = df["Epoch"].max()
        
        epochs = df["Epoch"].values
        data_loss = df["AvgDataLoss"].values
        mass_loss = df["AvgMassLoss"].values
        
        # Reconstruct the exact linear weight annealing schedule matching this specific run length
        lambda_phys = LAMBDA_MAX * (epochs / total_epochs)
        
        # Compute the weighted physics loss component
        weighted_mass_loss = lambda_phys * mass_loss
        
        # Calculate the percentage contribution of the physics loss to the total active loss
        physics_percentage = (weighted_mass_loss / (data_loss + weighted_mass_loss)) * 100
        
        # Initialize the 2-panel presentation layout
        # 1. Add facecolor to the main figure (the outer canvas)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5.5), dpi=200, facecolor='#f2f2f2')

        # 2. Set the background color for the individual plot areas
        ax1.set_facecolor('#f2f2f2')
        ax2.set_facecolor('#f2f2f2')
        
        # --- Panel 1: Order of Magnitude Alignment (Log Scale) ---
        ax1.plot(epochs, data_loss, color="darkorange", linewidth=2, label=r"Data Loss ($\mathcal{L}_{data}$)")
        ax1.plot(epochs, weighted_mass_loss, color="royalblue", linewidth=2, linestyle="--", 
                 label=r"Weighted Physics Loss ($\lambda_{phys}\cdot\mathcal{L}_{mass}$)")
        ax1.set_yscale("log")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Effective Loss Value (Log Scale)", fontsize=12)
        ax1.set_title(r"1. Loss Contributions: $\mathcal{L}_{data} \ vs. \ \lambda_{phys}\cdot\mathcal{L}_{mass}$)", fontsize=18, pad=10)
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)
        ax1.legend(loc="best", frameon=True, fontsize=14)
        ax1.fill_between(epochs, data_loss, weighted_mass_loss, color="purple", alpha=0.05)
        
        # --- Panel 2: Percentage Contribution to Total Loss (Linear Scale) ---
        ax2.plot(epochs, physics_percentage, color="crimson", linewidth=2.5, label="Physics Contribution %")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Share of Total Loss Vector (%)", fontsize=12)
        ax2.set_ylim(-5, 105)
        ax2.set_title(r"2. Relative Mass Loss Contribution: $\mathcal{L}_{mass}$ vs $\mathcal{L}_{total}$", 
                     fontsize=18, pad=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.axhline(0, color="black", linewidth=0.8, alpha=0.7)
        ax2.axhline(50, color="gray", linestyle=":", alpha=0.6, label="50% Equal Balance")
        ax2.legend(loc="best", fontsize=14)
        
        plt.suptitle("Physical Regularization Scaling", 
                     fontsize=20, y=1)
        plt.tight_layout()
        
        # Save output graphic: losses_{...}.txt -> justification_{...}.png
        output_name = file_path.name.replace("losses_", "justification_").replace(".txt", ".png")
        plt.savefig(source_dir / output_name, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated dynamic justification plot: {output_name} (Detected Epochs: {total_epochs})")
        
    except Exception as e:
        print(f"Could not process {file_path.name}: {e}")

print("All plots updated successfully.")