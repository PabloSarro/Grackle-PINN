import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_global_stats(file_list):
    """Computes global v_min and v_max for T, nHI, nHII."""
    global_mins = {3: np.inf, 6: np.inf, 7: np.inf}
    global_maxs = {3: -np.inf, 6: -np.inf, 7: -np.inf}
    
    for f in file_list:
        data = pd.read_csv(f, comment='#', sep='\s+', header=None)
        for col in [3, 6, 7]:
            vals = data.iloc[:, col].values
            vals = vals[vals > 0]
            if len(vals) > 0:
                global_mins[col] = min(global_mins[col], np.min(vals))
                global_maxs[col] = max(global_maxs[col], np.max(vals))
    return global_mins, global_maxs

def compute_Af(data, mins, maxs):
    Af = 0
    for col in [3, 6, 7]:
        v = np.maximum(data.iloc[:, col].values, 1e-30)
        delta_max = np.log(maxs[col] / mins[col])
        inner_sum = np.sum(np.abs(np.diff(np.log(v))))
        Af += (1 / delta_max) * inner_sum
    return Af

def process_trajectories(file_paths):
    mins, maxs = calculate_global_stats(file_paths)
    titles = ['Temperature [K]', 'Neutral Hydrogen [$cm^{-3}$]', 'Ionized Hydrogen [$cm^{-3}$]']
    
    for filepath in file_paths:
        data = pd.read_csv(filepath, comment='#', sep='\s+', header=None)
        t_yr, T, nHI, nHII = data.iloc[:, 1], data.iloc[:, 3], data.iloc[:, 6], data.iloc[:, 7]
        Af = compute_Af(data, mins, maxs)
        
        # Determine title
        traj_name = "Equilibrium Trajectory" if "eq_traj" in filepath else "Active Trajectory"
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plot_info = [(T, 'orange', 'T [K]', 3), (nHI, 'blue', '$n_{HI}$ [$cm^{-3}$]', 6), (nHII, 'red', '$n_{HII}$ [$cm^{-3}$]', 7)]
        
        for i, ax in enumerate(axes):
            col_idx = plot_info[i][3]
            ax.plot(t_yr, plot_info[i][0], color=plot_info[i][1], linewidth=2)
            
            if 'eq_traj' in filepath:
                ax.set_ylim(mins[col_idx] * 0.5, maxs[col_idx] * 2.0)
            
            ax.set_yscale('log')
            ax.set_title(titles[i], fontsize=14)
            ax.set_ylabel(plot_info[i][2])
            ax.set_xlabel('Time [yr]', fontsize=12) # Added X-axis label
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            
        plt.suptitle(f"{traj_name} | $A_f$ = {Af:.4f}", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{os.path.splitext(os.path.basename(filepath))[0]}_analysis.png")
        plt.close()

# Execution
files = ['eq_traj.dat', 'ac_traj.dat']
process_trajectories(files)