import os
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def stratify_dataset(csv_path, txt_path):
    # 1. Train the Oracle
    df = pd.read_csv(csv_path)
    X_train = df[['T_0(log)', 'nHI_0(log)', 'nHII_0(log)']].values
    y_train = df['Family_ID'].values
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # 2. Read the 1,000 generated parameters
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Dictionaries to hold the raw text lines grouped by family
    family_lines = {0: [], 1: [], 2: [], 3: []}

    print("Classifying and sorting the 1,000 parameters...")
    for line in lines:
        t0, n0, x0 = map(float, line.split())
        
        # Derive species and log space for the Oracle
        nHI = n0 * (1.0 - x0)
        nHII = n0 * x0
        
        log_T0 = np.log10(t0 + 1e-16)
        log_nHI = np.log10(nHI + 1e-16)
        log_nHII = np.log10(nHII + 1e-16)
        
        fid = knn.predict([[log_T0, log_nHI, log_nHII]])[0]
        family_lines[fid].append(line)

    # 3. Stratify and Split
    train_lines = []
    val_lines = []

    for fid, f_lines in family_lines.items():
        # Shuffle within the family to ensure random selection
        random.seed(42)
        random.shuffle(f_lines)
        
        # Exactly 200 for Train, 50 for Val
        train_lines.extend(f_lines[:200])
        val_lines.extend(f_lines[200:250])

    # 4. Global Shuffle (CRITICAL FOR NEURAL NETWORKS)
    # This destroys the family grouping so your mini-batches are diverse
    random.shuffle(train_lines)
    random.shuffle(val_lines)

    # 5. Write out the final files
    with open("ICs_train.txt", "w") as f:
        f.write("\n".join(train_lines) + "\n")
        
    with open("ICs_valid.txt", "w") as f:
        f.write("\n".join(val_lines) + "\n")

    print("\n========================================================")
    print("[✓] Stratification Complete!")
    print(f"    -> ICs_train.txt : {len(train_lines)} lines (Exactly 200 per family)")
    print(f"    -> ICs_valid.txt   : {len(val_lines)} lines (Exactly 50 per family)")
    print("========================================================")

if __name__ == "__main__":
    CSV_FILE = "Trajectory_Families.csv"
    TXT_FILE = "ICs_balanced.txt" 
    
    stratify_dataset(CSV_FILE, TXT_FILE)