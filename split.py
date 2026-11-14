import os
import glob
import random
import shutil

def split_dataset(src_dir, train_prop=0.7):
    if not os.path.isdir(src_dir):
        print(f"[!] Error: Source directory '{src_dir}' not found.")
        return

    # Find all trajectory files
    files = glob.glob(os.path.join(src_dir, "output*.dat"))
    total_files = len(files)
    
    if total_files == 0:
        print(f"[!] No 'output*.dat' files found in '{src_dir}'.")
        return

    print(f"Found {total_files} files in '{src_dir}'. Shuffling and splitting...")

    # Shuffle randomly to ensure an unbiased distribution
    random.seed(42)  # Set seed for reproducibility
    random.shuffle(files)

    # Calculate split index
    split_idx = int(total_files * train_prop)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    # Define target directories
    train_dir = os.path.join(src_dir, "train")
    val_dir = os.path.join(src_dir, "val")

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Move files to train
    for f in train_files:
        shutil.move(f, os.path.join(train_dir, os.path.basename(f)))

    # Move files to val
    for f in val_files:
        shutil.move(f, os.path.join(val_dir, os.path.basename(f)))

    print(f"[✓] Split complete:")
    print(f"    - Moved {len(train_files)} files to {train_dir}")
    print(f"    - Moved {len(val_files)} files to {val_dir}")

if __name__ == "__main__":
    # Point this to your new folder containing the 1,000 files
    TARGET_FOLDER = "Active_Files/" 
    split_dataset(TARGET_FOLDER)