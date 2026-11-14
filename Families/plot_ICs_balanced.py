import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def validate_colored_ics(csv_path, txt_path):
    if not os.path.exists(csv_path):
        print(f"[!] Error: '{csv_path}' not found.")
        return
    if not os.path.exists(txt_path):
        print(f"[!] Error: '{txt_path}' not found.")
        return

    # ========================================================
    # 1. Load Original Data & Train the Oracle
    # ========================================================
    df_orig = pd.read_csv(csv_path)
    
    X_train = df_orig[['T_0(log)', 'nHI_0(log)', 'nHII_0(log)']].values
    y_train = df_orig['Family_ID'].values
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Convert back to physical units for background plotting
    df_orig['T_0'] = 10 ** df_orig['T_0(log)']
    df_orig['nHI_0'] = 10 ** df_orig['nHI_0(log)']
    df_orig['nHII_0'] = 10 ** df_orig['nHII_0(log)']
    families = sorted(df_orig['Family_ID'].unique())

    # ========================================================
    # 2. Load New Data & Predict Families
    # ========================================================
    new_data = np.loadtxt(txt_path)
    T0_new = new_data[:, 0]
    nH_new = new_data[:, 1]
    xHII_new = new_data[:, 2]
    
    nHI_new = nH_new * (1.0 - xHII_new)
    nHII_new = nH_new * xHII_new

    # Convert new points to log-space to ask the Oracle
    log_T0_new = np.log10(T0_new + 1e-16)
    log_nHI_new = np.log10(nHI_new + 1e-16)
    log_nHII_new = np.log10(nHII_new + 1e-16)
    
    X_new = np.c_[log_T0_new, log_nHI_new, log_nHII_new]
    y_new_predicted = knn.predict(X_new)

    # ========================================================
    # 3. Plotting with Matched Colors
    # ========================================================
    print(f"Overlaying {len(new_data)} matched-color stars onto {len(df_orig)} background points...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Standard exact colors for Fam 0, 1, 2, 3
    colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}
    bg_kws = {'alpha': 0.15, 's': 20, 'edgecolors': 'none'}
    star_kws = {'alpha': 1.0, 's': 60, 'marker': '*', 'edgecolors': 'black', 'linewidth': 0.4}

    for fid in families:
        # Plot Background Dots
        bg_subset = df_orig[df_orig['Family_ID'] == fid]
        axes[0].scatter(bg_subset['T_0'], bg_subset['nHI_0'], color=colors[fid], **bg_kws)
        axes[1].scatter(bg_subset['T_0'], bg_subset['nHII_0'], color=colors[fid], **bg_kws)
        axes[2].scatter(bg_subset['nHI_0'], bg_subset['nHII_0'], color=colors[fid], label=f'Fam {fid} Area', **bg_kws)

        # Plot New Stars matching this family
        mask = (y_new_predicted == fid)
        axes[0].scatter(T0_new[mask], nHI_new[mask], color=colors[fid], **star_kws)
        axes[1].scatter(T0_new[mask], nHII_new[mask], color=colors[fid], **star_kws)
        axes[2].scatter(nHI_new[mask], nHII_new[mask], color=colors[fid], label=f'New Fam {fid} ICs', **star_kws)

    # Formatting 
    titles = ['Temperature vs Neutral Hydrogen', 'Temperature vs Ionized Hydrogen', 'Neutral vs Ionized Hydrogen']
    x_labels = ['Initial Temperature $T_0$ [K]', 'Initial Temperature $T_0$ [K]', 'Initial Neutral H $n_{HI,0}$ [cm$^{-3}$]']
    y_labels = ['Initial Neutral H $n_{HI,0}$ [cm$^{-3}$]', 'Initial Ionized H $n_{HII,0}$ [cm$^{-3}$]', 'Initial Ionized H $n_{HII,0}$ [cm$^{-3}$]']

    for i, ax in enumerate(axes):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(x_labels[i], fontsize=12)
        ax.set_ylabel(y_labels[i], fontsize=12)
        ax.set_title(titles[i], fontsize=14)
        ax.grid(True, which="major", linestyle="-", alpha=0.5)
        ax.grid(True, which="minor", linestyle="--", alpha=0.2)
        
        if i == 2: # Put the legend only on the 3rd plot to save space
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    out_name = "Validation_Colored_Overlay.png"
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    print(f"[✓] Validation overlay saved to {out_name}")

if __name__ == "__main__":
    CSV_FILE = "Trajectory_Families.csv"
    TXT_FILE = "ICs_balanced.txt" 
    
    validate_colored_ics(CSV_FILE, TXT_FILE)