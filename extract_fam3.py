import os
import shutil
import glob
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# --- Configuration ---
SOURCE_TRAIN = "/scratch/izar/sarro/Balanced_Dataset/train"
SOURCE_VALID = "/scratch/izar/sarro/Balanced_Dataset/valid"
DEST_TRAIN = "/scratch/izar/sarro/Fam3_Dataset/train"
DEST_VALID = "/scratch/izar/sarro/Fam3_Dataset/valid"

CSV_PATH = "Families/Trajectory_Families.csv"

# Grackle Data Columns (from your previous scripts)
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

def extract_target_family(source_dir, dest_dir, tag, knn_oracle, target_family=3):
    if not os.path.exists(source_dir):
        print(f"[!] Warning: Source directory '{source_dir}' not found.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    files_to_check = glob.glob(os.path.join(source_dir, "*.dat"))
    
    print(f"\nScanning '{source_dir}' for Family {target_family} files...")
    
    copied_count = 0
    
    for filepath in files_to_check:
        try:
            # Read only the first row of the file to get the Initial Conditions
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
            
            # If it's the target family, copy the ENTIRE file over
            if fid == target_family:
                filename = os.path.basename(filepath)
                dest_path = os.path.join(dest_dir, filename)
                shutil.copy2(filepath, dest_path)
                copied_count += 1
                
                # Print progress every 50 files
                if copied_count % 50 == 0:
                    print(f"  -> Found and copied {copied_count} Family {target_family} files...")
                
        except Exception as e:
            # Skip files that are malformed
            continue 

    print(f"[✓] {tag} Extraction Complete: Copied {copied_count} files to {dest_dir}")

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"[!] CRITICAL ERROR: Could not find '{CSV_PATH}'.")
        print("Please ensure you are running this in the same directory as your CSV file.")
    else:
        # 1. Train the Oracle
        oracle = build_oracle(CSV_PATH)
        
        # 2. Extract Family 3 Training Samples (Should be exactly 250)
        extract_target_family(SOURCE_TRAIN, DEST_TRAIN, "Training", oracle, target_family=3)
        
        # 3. Extract Family 3 Validation Samples (Should be exactly 50)
        extract_target_family(SOURCE_VALID, DEST_VALID, "Validation", oracle, target_family=3)
        
        print("\n[✓] All Family 3 files successfully isolated!")