import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def generate_balanced_ics(csv_path, target_per_family=250):
    if not os.path.exists(csv_path):
        print(f"[!] Error: '{csv_path}' not found.")
        return

    print("Training the KNN Oracle on your phase space...")
    df = pd.read_csv(csv_path)
    
    # Train the Oracle on the log-space initial conditions
    X_train = df[['T_0(log)', 'nHI_0(log)', 'nHII_0(log)']].values
    y_train = df['Family_ID'].values
    
    # k=3 tightly hugs the boundaries learned by your Grid analysis
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Physical bounds based on your previous grid
    T_min, T_max = 2.0, 8.0     # log10(Kelvin)
    n_min, n_max = -4.0, 2.0    # log10(cm^-3)
    x_min, x_max = 1e-4, 0.9999 # Ionization fraction
    
    # Trackers
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    valid_params = []
    
    print(f"Sampling phase space to find exactly {target_per_family} ICs per family...")

    total_attempts = 0
    while any(c < target_per_family for c in counts.values()):
        total_attempts += 1
        
        # 1. Randomly sample the physical parameters
        logT = np.random.uniform(T_min, T_max)
        lognH = np.random.uniform(n_min, n_max)
        xHII = np.random.uniform(x_min, x_max)
        
        # 2. Convert to the features the Oracle expects (T_0, nHI_0, nHII_0 in log-space)
        T0 = 10 ** logT
        nH = 10 ** lognH
        nHI = nH * (1.0 - xHII)
        nHII = nH * xHII
        
        log_T0 = np.log10(T0 + 1e-16)
        log_nHI = np.log10(nHI + 1e-16)
        log_nHII = np.log10(nHII + 1e-16)
        
        # 3. Ask the Oracle
        predicted_family = knn.predict([[log_T0, log_nHI, log_nHII]])[0]
        
        # 4. If we still need this family, keep it!
        if counts[predicted_family] < target_per_family:
            counts[predicted_family] += 1
            valid_params.append(f"{T0:.6e} {nH:.6e} {xHII:.6f}\n")
            
            # Print an update every 100 successful finds
            total_found = sum(counts.values())
            if total_found % 100 == 0:
                print(f"  -> Found {total_found}/{target_per_family * 4} ... {counts}")

    # Write the perfectly balanced dataset to a new parameter file
    out_file = "ICs_balanced.txt"
    with open(out_file, "w") as f:
        f.writelines(valid_params)

    print("\n========================================================")
    print(f"[✓] Successfully generated 1,000 perfectly balanced parameters!")
    print(f"    Total random attempts required: {total_attempts:,}")
    print(f"    Saved to: {out_file}")
    print("========================================================")

if __name__ == "__main__":
    CSV_FILE = "Trajectory_Families.csv"
    generate_balanced_ics(CSV_FILE, target_per_family=250)