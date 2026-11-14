import os
import shutil
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# --- Configuration ---
TRAIN_DIR = "/scratch/izar/sarro/Balanced_Dataset/train/"
VALID_DIR = "/scratch/izar/sarro/Balanced_Dataset/valid/"
CSV_PATH = "Families/Trajectory_Families.csv"
PLOT_DIR = "Plot_Files"

# Update these indices based on where T, nHI, and nHII are located in your Grackle .dat files!
T_COL = 3
NHI_COL = 6
NHII_COL = 7

def build_oracle(csv_path):
    print(f"Reading ground truth from '{csv_path}' and training KNN Oracle...")
    df = pd.read_csv(csv_path)
    X_train = df[['T_0(log)', 'nHI_0(log)', 'nHII_0(log)']].values
    y_train = df['Family_ID'].values
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn

def extract_samples(source_dir, prefix, knn_oracle):
    if not os.path.exists(source_dir):
        print(f"[!] Warning: Source directory '{source_dir}' not found.")
        return

    found_families = set()
    files_to_check = [f for f in os.listdir(source_dir) if f.endswith(".dat")]
    
    print(f"\nScanning '{source_dir}' for representative family files...")
    
    for filename in files_to_check:
        if len(found_families) == 4:
            break # We found one of each family (0, 1, 2, 3), we can stop checking
            
        filepath = os.path.join(source_dir, filename)
        
        try:
            # Read only the first row of the file to get the Initial Conditions
            # The 'sep='\s+'' handles multiple spaces between columns robustly
            first_row = pd.read_csv(filepath, sep='\s+', nrows=1, header=None, comment='#').iloc[0].values
            
            T0 = first_row[T_COL]
            nHI0 = first_row[NHI_COL]
            nHII0 = first_row[NHII_COL]
            
            # Convert to log-space for the Oracle
            log_T0 = np.log10(T0 + 1e-16)
            log_nHI = np.log10(nHI0 + 1e-16)
            log_nHII = np.log10(nHII0 + 1e-16)
            
            # Ask the Oracle what family this IC belongs to
            fid = knn_oracle.predict([[log_T0, log_nHI, log_nHII]])[0]
            
            # If we haven't found a file for this family yet, copy it to the Plot folder
            if fid not in found_families:
                dest_name = f"{prefix}_Fam_{fid}.dat"
                dest_path = os.path.join(PLOT_DIR, dest_name)
                shutil.copy2(filepath, dest_path)
                found_families.add(fid)
                print(f"  [✓] Found Family {fid} -> Saved as {dest_name}")
                
        except Exception as e:
            # Skip files that are empty, malformed, or currently being written to
            continue 

    if len(found_families) < 4:
        print(f"  [!] Only found families {found_families} in {prefix}. Dataset might be imbalanced!")

if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_PATH):
        print(f"[!] CRITICAL ERROR: Could not find '{CSV_PATH}'.")
        print("Please ensure you are running this in the same directory as your CSV file.")
    else:
        # 1. Train the Oracle
        oracle = build_oracle(CSV_PATH)
        
        # 2. Extract Training Samples
        extract_samples(TRAIN_DIR, "Train", oracle)
        
        # 3. Extract Validation Samples
        extract_samples(VALID_DIR, "Valid", oracle)
        
        print(f"\n[✓] Extraction complete. 8 sample files are ready in '{PLOT_DIR}/'.")