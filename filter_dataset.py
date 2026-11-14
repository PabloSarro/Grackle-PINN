import os
import glob
import numpy as np

# --- Configuration ---
# Your original balanced dataset (Read-Only)
SOURCE_TRAIN = "/scratch/izar/sarro/Balanced_Dataset/train"
SOURCE_VALID = "/scratch/izar/sarro/Balanced_Dataset/valid"

# The new highly-condensed dataset
DEST_TRAIN = "/scratch/izar/sarro/Filtered_Data/train"
DEST_VALID = "/scratch/izar/sarro/Filtered_Data/valid"

# Hyperparameters for the temporal slicer
THRESHOLD = 1e-5          # Minimum log-derivative to be considered an "active shock"
BUFFER_STEPS = 10         # How many steps to keep BEFORE and AFTER the shock
FLATLINE_KEEP_RATE = 100  # Keep 1 out of every X points during flatlines

def dynamic_temporal_downsample(filepath, out_filepath):
    """Reads a Grackle .dat file, filters out dead time, and saves it."""
    
    # 1. Extract the Grackle header to preserve compatibility
    header_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                # Strip the leading '#' because np.savetxt adds it automatically
                header_lines.append(line.lstrip('#').strip())
            else:
                break
    header_str = "\n".join(header_lines)

    # 2. Load the data (ignoring the header)
    try:
        data = np.loadtxt(filepath, comments='#')
    except Exception as e:
        print(f"  [!] Failed to read {filepath}: {e}")
        return False

    if len(data) < 2:
        return False

    # 3. Extract Temperature (Assuming column 3 based on your previous scripts)
    T = data[:, 3]
    
    # 4. Calculate the Log-Gradient (Derivative)
    dlog_T = np.zeros_like(T)
    dlog_T[1:] = np.abs(np.log10(T[1:] + 1e-16) - np.log10(T[:-1] + 1e-16))
    
    # 5. Create the "Active Physics" mask
    active_mask = dlog_T > THRESHOLD
    
    # Pad the mask using a simple 1D convolution to keep a buffer around the shock
    kernel = np.ones(BUFFER_STEPS * 2 + 1)
    padded_mask = np.convolve(active_mask, kernel, mode='same') > 0
    
    # 6. Create the "Sparse Flatline" mask
    downsample_mask = np.zeros(len(T), dtype=bool)
    downsample_mask[::FLATLINE_KEEP_RATE] = True 
    
    # 7. Combine the masks
    final_mask = padded_mask | downsample_mask
    
    # Always forcefully keep the very first and very last boundary conditions
    final_mask[0] = True
    final_mask[-1] = True
    
    # 8. Apply the mask and save
    filtered_data = data[final_mask]
    
    np.savetxt(out_filepath, filtered_data, header=header_str, comments='# ', fmt='%.6e')
    
    # Return some quick stats to print
    return len(data), len(filtered_data)

def process_directory(source_dir, dest_dir, tag):
    if not os.path.exists(source_dir):
        print(f"[!] Critical: Source directory '{source_dir}' not found.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    files = glob.glob(os.path.join(source_dir, "*.dat"))
    
    print(f"\n--- Processing {tag} Set ({len(files)} files) ---")
    
    total_original = 0
    total_filtered = 0
    
    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        out_filepath = os.path.join(dest_dir, filename)
        
        stats = dynamic_temporal_downsample(filepath, out_filepath)
        
        if stats:
            orig_len, filt_len = stats
            total_original += orig_len
            total_filtered += filt_len
            
            if i % 100 == 0:
                print(f"  [{i}/{len(files)}] {filename}: {orig_len} pts -> {filt_len} pts")

    if total_original > 0:
        reduction = (1 - (total_filtered / total_original)) * 100
        print(f"[✓] {tag} Complete. Reduced {total_original:,} points to {total_filtered:,} points ({reduction:.1f}% reduction).")

if __name__ == "__main__":
    process_directory(SOURCE_TRAIN, DEST_TRAIN, "Training")
    process_directory(SOURCE_VALID, DEST_VALID, "Validation")
    print("\n[✓] All data successfully filtered and saved to 'Filtered_Data/'.")