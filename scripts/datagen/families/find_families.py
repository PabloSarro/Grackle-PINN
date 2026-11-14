import os
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_fingerprint(filepath):
    """Extracts a physical fingerprint from a Grackle trajectory."""
    data = np.loadtxt(filepath, comments='#')
    if len(data) < 2:
        return None
        
    T = data[:, 3]
    nHI = data[:, 6]
    nHII = data[:, 7]
    
    # 1. Initial Conditions (Where does it start?)
    T_0 = np.log10(T[0] + 1e-16)
    nHI_0 = np.log10(nHI[0] + 1e-16)
    nHII_0 = np.log10(nHII[0] + 1e-16)
    
    # 2. Total Variance (How active is it?)
    var_T = np.var(np.log10(T + 1e-16))
    var_HI = np.var(np.log10(nHI + 1e-16))
    var_HII = np.var(np.log10(nHII + 1e-16))
    
    # 3. Final State (Where does it equilibrate?)
    T_end = np.log10(T[-1] + 1e-16)
    
    # The "Fingerprint" array representing the trajectory's behavior
    return [T_0, nHI_0, nHII_0, var_T, var_HI, var_HII, T_end]

def analyze_families(folderpath, num_families=4):
    print(f"Scanning '{folderpath}' and building physical fingerprints...")
    
    dat_files = glob.glob(os.path.join(folderpath, "output*.dat"))
    
    fingerprints = []
    valid_files = []
    
    for f in dat_files:
        fp = extract_fingerprint(f)
        if fp is not None:
            fingerprints.append(fp)
            valid_files.append(f)
            
    X = np.array(fingerprints)
    
    # Standardize the features so variance doesn't overpower initial conditions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Clustering {len(valid_files)} trajectories into {num_families} behavioral families...")
    
    # Run K-Means Clustering
    kmeans = KMeans(n_clusters=num_families, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Combine results into a DataFrame
    df = pd.DataFrame(X, columns=['T_0(log)', 'nHI_0(log)', 'nHII_0(log)', 'Var_T', 'Var_HI', 'Var_HII', 'T_end(log)'])
    df['Filename'] = [os.path.basename(f) for f in valid_files]
    df['Family_ID'] = labels
    
    # Print the exact mathematical center of each family
    print("\n--- K-Means Centroids (Log Scale) ---")
    print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=df.columns[:-2]))

    # Print the absolute Min/Max boundaries for the initial conditions of each family
    print("\n--- Family Boundaries (Min/Max Log Values) ---")
    print(df.groupby('Family_ID')[['T_0(log)', 'nHI_0(log)', 'nHII_0(log)']].agg(['min', 'max']))
    
    
    # Analyze the distinct families
    print("\n========================================================")
    print("                 TRAJECTORY FAMILIES                    ")
    print("========================================================")
    
    for family_id in range(num_families):
        family_df = df[df['Family_ID'] == family_id]
        count = len(family_df)
        
        # Calculate the median fingerprint for this family
        med_T0 = 10 ** family_df['T_0(log)'].median()
        med_VarT = family_df['Var_T'].median()
        
        print(f"\n[Family {family_id}] - {count} files")
        print(f"  -> Median Initial Temp : {med_T0:.2e} K")
        print(f"  -> Median Temp Variance: {med_VarT:.4f} (Log scale)")
        
        if med_VarT > 0.5:
            print("  -> Behavior: Highly Active / Rapid Cooling")
        elif med_VarT > 0.01:
            print("  -> Behavior: Moderate Transition")
        else:
            print("  -> Behavior: Stable / Equilibrium")
            
        # Show 3 random example files from this family so you can go plot them!
        examples = family_df['Filename'].sample(min(3, count)).tolist()
        print(f"  -> Example files: {examples}")

    # Save to CSV so you can easily sort and filter them later
    csv_name = "Trajectory_Families.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n[✓] Full mapping saved to {csv_name}")

if __name__ == "__main__":
    TARGET_FOLDER = "/scratch/izar/sarro/Grid_Analysis/Grid_Files/" # Change to wherever your files are
    analyze_families(TARGET_FOLDER, num_families=4)