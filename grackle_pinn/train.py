import os
import json
import shutil
import time
import torch
import argparse
from grackle_pinn.model import PINN
from grackle_pinn.metrics import ComputeMSE
from grackle_pinn.data_utils import GrackleDataset
from grackle_pinn.physics import PhysicsLossManager
from torch.utils.data import DataLoader


# Run configuration: one JSON file per experiment (see configs/).
CONFIG_KEYS = dict(
    run_name=None, train_path=None, valid_path=None, out_dir="results/runs",
    max_runtime_hours=23.75, batch_size=8192, learning_rate=1.0e-4, epochs=5000,
    check_every=10, weight_factor=10.0,
)
starting_time = time.time()

# Argument Parsing (if refining a pre-trained model)
parser = argparse.ArgumentParser(description="Train Grackle PINN")
parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration of the run')
parser.add_argument('--model', type=str, default=None, 
                    help='Path to pre-trained model .pth file (leaves it as None to train from scratch)')
args = parser.parse_args()

cfg = dict(CONFIG_KEYS)
with open(args.config) as f:
    cfg.update(json.load(f))
os.makedirs(os.path.join(cfg["out_dir"], cfg["run_name"]), exist_ok=True)
_out = lambda name: os.path.join(cfg["out_dir"], cfg["run_name"], name)
shutil.copy(args.config, _out("config.json"))

MODEL_NAME = _out(f"PINN_{cfg['run_name']}.pth")
BEST_NAME = _out(f"PINN_BEST_{cfg['run_name']}.pth")
SCALING_NAME = _out(f"scaling_params_{cfg['run_name']}.pth")
LOSSES_NAME = _out(f"losses_{cfg['run_name']}.txt")
TRAIN_PATH, VALID_PATH = cfg["train_path"], cfg["valid_path"]
MAX_RUNTIME_SECONDS = cfg["max_runtime_hours"] * 3600

# Global Precision Setup: 64-bit arithmetic
torch.set_default_dtype(torch.float64)

# Initial Configuration and Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Training Precision: {torch.get_default_dtype()}")

# Training parameters.
BATCH_SIZE = cfg["batch_size"]
LEARNING_RATE = cfg["learning_rate"]
EPOCHS = cfg["epochs"]
CHECK_EVERY = cfg["check_every"] # Epoch frequency for which validation loss is computed.

# Training Data Loading
dataset = GrackleDataset(folder_path=TRAIN_PATH)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
torch.save({
    'in_mean': dataset.in_mean,
    'in_std': dataset.in_std,
    'tg_mean': dataset.tg_mean,
    'tg_std': dataset.tg_std
}, SCALING_NAME)
print(f"Scaling parameters saved to {SCALING_NAME}")

# Validation Data Loading
dataset_valid = GrackleDataset(
    folder_path=VALID_PATH,
    in_mean=dataset.in_mean,   # Force validation data to use Training means and stds
    in_std=dataset.in_std,
    tg_mean=dataset.tg_mean,
    tg_std=dataset.tg_std
)
dataloader_valid = DataLoader(
    dataset_valid, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True # DEBUG?
)

# Model, Optimiser, and Loss
model = PINN(input_dim=3, output_dim=3, hidden_dim=512, hidden_layers=12).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, 
    mode='min',        # We want to minimize the loss
    factor=0.5,        # Cut the learning rate in half when triggered
    patience=5,        # If the loss doesn't reach a new low for 5*CHECK_EVERY = 50 epochs, trigger the cut
    min_lr=1e-6        # Never drop the LR below this absolute floor
)

# Pre-trained model mechanism (or training from scratch, if nothing was specified) ---
if args.model is not None:
    if os.path.exists(args.model):
        pre_trained = torch.load(args.model, map_location=device, weights_only=True)
        # Load the weights natively onto your active device (CPU or GPU)
        if 'model' in pre_trained:
            model.load_state_dict(pre_trained['model'])
            optimiser.load_state_dict(pre_trained["optimizer"])
            scheduler.load_state_dict(pre_trained["scheduler"])
            best_loss = pre_trained["best_loss"]
        else:
            model.load_state_dict(pre_trained)
            best_loss = float('inf')
        print(f"[✓] Successfully loaded pre-trained weights from: {args.model}")
    else:
        print(f"[!] CRITICAL: Specified model file '{args.model}' not found!")
        import sys; sys.exit(1)
else:
    print("[#] No pre-trained model specified. Initialising and training from scratch.")
    best_loss = float('inf')


phys_manager = PhysicsLossManager(
    model, 
    in_mean=dataset.in_mean.to(device), 
    in_std=dataset.in_std.to(device),
    tg_mean=dataset.tg_mean.to(device),
    tg_std=dataset.tg_std.to(device)
)

# Header write to losses file.
with open(LOSSES_NAME, "w") as f:
    f.write(f"Epoch  AvgDataLoss  AvgMassLoss AvgTotalLoss ValidationLoss\n")


print(f"\nStarting training on {len(dataset)} points...")

# 5. Training Loop
for epoch in range(EPOCHS):
    start = time.time()
    # print(f"Epoch {epoch+1}")
    model.train()
    total_loss = 0
    total_data_loss = 0
    total_mass_loss = 0
    valid_loss_str = "nan"
    # batch_num = 0

    for batch_x, batch_y in dataloader:
        # print(f"Batch {batch_num+1}")
        # batch_num += 1
        # Move data to GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimiser.zero_grad()
        
        # Forward pass and losses
        predictions = model(batch_x)

        # Weighted Data Loss (punishes mistakes in spikes, keeps regular MSE in equilibrium)
        weights = 1.0 + cfg["weight_factor"] * torch.abs(batch_y)
        
        # Calculate element-wise squared error, multiply by weights, then mean
        loss_data = torch.mean(weights * (predictions - batch_y)**2)
        loss_mass = phys_manager.mass_conserv(batch_x, predictions)

        LAMBDA_MASS = 1e7*(epoch)/(EPOCHS-1)  
        
        # Total combined loss with weightings
        loss = loss_data + LAMBDA_MASS * loss_mass

        # ==== DEBUG BLOCK! ====
        if torch.isnan(loss):
            print(f"\n[!] NAN DETECTED at Epoch {epoch+1}")#, Batch {batch_num}")
            print(f"Data Loss: {loss_data.item():.4e}")
            print(f"Mass Loss: {loss_mass.item():.4e}")
            print(f"Preds Max/Min: {predictions.max().item():.2f} / {predictions.min().item():.2f}")
            print(f"Inputs Max/Min: {batch_x.max().item():.2f} / {batch_x.min().item():.2f}")
            
            # Check if the Ground Truth itself has NaNs
            if torch.isnan(batch_y).any():
                print("CRITICAL: NaNs found in batch_y (Ground Truth Data!)")
            
            # Stop the execution so you don't waste cluster credits
            import sys; sys.exit(1)
        # =======================

        # Backward pass
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0) # max_norm usually set to 0.5 to 10 times the average gradient norm.
        optimiser.step()

        total_loss += loss.item()
        total_data_loss += loss_data.item()
        total_mass_loss += loss_mass.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_data_loss = total_data_loss / len(dataloader)
    avg_mass_loss = total_mass_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss = {avg_loss:.6e} --> Data: {avg_data_loss:.6e}, Mass: {avg_mass_loss:.6e}")

    # Once every 'epoch_freq_valid' epochs,
    if epoch % CHECK_EVERY == 0:
        # Save the best model so far.
        valid_loss = ComputeMSE(model, dataloader_valid, device)
        valid_loss_str = f"{valid_loss:.6e}"
        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
            }, BEST_NAME)
            print(f"[✓] New best model saved as: {BEST_NAME} ---> MSE_loss={valid_loss_str})")
            

    if epoch % 100 == 0:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss
        }, MODEL_NAME)
        print(f"[✓] PERIODIC STORE: Current model saved as: {MODEL_NAME}")

    # Log the training and validation losses in a file
    with open(LOSSES_NAME, "a") as f:
        f.write(f"{epoch+1} {avg_data_loss:.6e} {avg_mass_loss:.6e} {avg_loss:.6e} {valid_loss_str}\n")


    # Store last model before job dies!
    elapsed_time = time.time() - starting_time
    if elapsed_time > MAX_RUNTIME_SECONDS:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss
        }, MODEL_NAME)
        print(f"[!] CLOSE TO REACHING TIME LIMIT ({elapsed_time/3600:.2f} hours). Last model saved as: {MODEL_NAME}.")
        break

    print(f"Epoch {epoch+1}: {time.time() - start:.2e} seconds.")


print(f"Training complete in {(time.time() - starting_time)/3600:.2f} hours.")
torch.save({
    "model": model.state_dict(),
    "optimizer": optimiser.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_loss": best_loss
}, MODEL_NAME)
print(f"[✓] Final model saved as: {MODEL_NAME} ---> MSE_loss={ComputeMSE(model, dataloader_valid, device):.4e})")