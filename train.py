import torch
import torch.nn as nn
import argparse
from model import PINN
from ComputeMSE import ComputeMSE
from torch.utils.data import DataLoader
from data_utils import GrackleDataset
from physics import GrackleRates, PhysicsLossManager
import time

# --- ARGUMENT PARSER ---
parser = argparse.ArgumentParser(description="Train Grackle PINN")
parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float64'], 
                    help='Set training precision')
args = parser.parse_args()

# --- GLOBAL PRECISION SETUP ---
if args.precision == 'float64':
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)

# Initial Configuration and Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Training Precision: {torch.get_default_dtype()}")

MODEL_NAME = f"PINN.pth"#{args.precision}.pth"
BEST_NAME = f"PINN_BEST.pth"#{args.precision}.pth"
SCALING_NAME = f"scaling_params.pth"#_PINN_{args.precision}.pth"
DATA_PATH = "Train_Stratified/" # Path where the output_*.dat are located

BATCH_SIZE = 4096 # 16384 # 8192 # 2048
LEARNING_RATE = 1.0e-3
EPOCHS = 1000
LAMBDA_COOL = 0.0

# Data Loading
dataset = GrackleDataset(folder_path=DATA_PATH)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
torch.save({
    'in_mean': dataset.in_mean,
    'in_std': dataset.in_std,
    'tg_mean': dataset.tg_mean,
    'tg_std': dataset.tg_std
}, SCALING_NAME)
print(f"Scaling parameters saved to {SCALING_NAME}")

# Model, Optimiser, and Loss
model = PINN(input_dim=4, output_dim=3, hidden_dim=256, hidden_layers=6).to(device)
grackle_phys = GrackleRates()
phys_manager = PhysicsLossManager(
    model, 
    grackle_phys, 
    in_mean=dataset.in_mean.to(device), 
    in_std=dataset.in_std.to(device),
    tg_mean=dataset.tg_mean.to(device),
    tg_std=dataset.tg_std.to(device)
)
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # DEBUG: Try Ethan's suggestion!
# criterion = nn.MSELoss()

print(f"Starting training on {len(dataset)} points...")

# 5. Training Loop
best_loss = float('inf')
times_best_saved = 0

start_time = time.time()
for epoch in range(EPOCHS):
    # print(f"Epoch {epoch+1}")
    model.train()
    total_data_loss = 0
    total_mass_loss = 0
    total_cool_loss = 0
    batch_num = 0
    for batch_x, batch_y in dataloader:
        # print(f"Batch {batch_num+1}")
        batch_num += 1
        # Move data to GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # batch_x.requires_grad = True ---> NOT USED FOR NOW, ONLY IF I DID: torch.autograd.grad(...) FOR DERIVATIVES IN 'physics.py'
        optimiser.zero_grad()
        
        # Forward pass and losses
        predictions = model(batch_x)
        # --- THE WEIGHTED DATA LOSS ---
        # If batch_y is close to 0, weight is ~1.0
        # If batch_y is large (a shock), weight becomes massive.
        # The factor 20.0 is an aggressive multiplier to force it to fit the spikes.
        weights = 1.0 + 20.0 * torch.abs(batch_y) 
        
        # Calculate element-wise squared error, multiply by weights, then mean
        loss_data = torch.mean(weights * (predictions - batch_y)**2)
        # loss_data = criterion(predictions, batch_y)
        loss_mass, loss_cool = phys_manager.get_residuals(batch_x, predictions)

        LAMBDA_MASS = 1e4*(epoch)/(EPOCHS-1)  
        
        # Total combined loss with weightings
        loss = loss_data + LAMBDA_MASS * loss_mass # + LAMBDA_COOL * loss_cool

        # --- DEBUG BLOCK ---
        if torch.isnan(loss):
            print(f"\n[!] NAN DETECTED at Epoch {epoch+1}")#, Batch {batch_num}")
            print(f"Data Loss: {loss_data.item():.4e}")
            print(f"Mass Loss: {loss_mass.item():.4e}")
            print(f"Cool Loss: {loss_cool.item():.4e}")
            print(f"Preds Max/Min: {predictions.max().item():.2f} / {predictions.min().item():.2f}")
            print(f"Inputs Max/Min: {batch_x.max().item():.2f} / {batch_x.min().item():.2f}")
            
            # Check if the Ground Truth itself has NaNs
            if torch.isnan(batch_y).any():
                print("CRITICAL: NaNs found in batch_y (Ground Truth Data!)")
            
            # Stop the execution so you don't waste cluster credits
            import sys; sys.exit(1)
        # -----------------------------

        # Backward pass
        loss.backward()
        
        # Gradient Clipping
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0) # max_norm usually set to 0.5 to 10 times the average gradient norm.
        if (batch_num < 10) and (epoch % 100) == 0:
            print(f"[Diagnostics] True Gradient Norm: {total_norm.item():.4f}")

        optimiser.step()
        
        total_data_loss += loss_data.item()
        total_mass_loss += loss_mass.item()
        total_cool_loss += loss_cool.item()
    
    avg_data_loss = total_data_loss / len(dataloader)
    avg_mass_loss = total_mass_loss / len(dataloader)
    avg_cool_loss = total_cool_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Data Loss: {avg_data_loss:.6e}, Avg Mass Loss: {avg_mass_loss:.6e}, Avg Cool Loss: {avg_cool_loss:.6e}")

    # Save the best model so far.
    avg_loss = avg_data_loss # + LAMBDA_MASS * loss_mass + LAMBDA_COOL * loss_cool
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), BEST_NAME)

        if (times_best_saved % 20) == 0:
            print(f"[✓] New best model saved ---> MSE_loss={ComputeMSE(model, dataloader, device, BATCH_SIZE):.4e})")
        else:
            print(f"[✓] New best model saved ---> AVG_loss={avg_loss:.4e})")
        times_best_saved += 1


end_time = time.time()
print(f"Training complete in {end_time - start_time:.2f} seconds.")
torch.save(model.state_dict(), MODEL_NAME)
print(f"[✓] Final model saved ---> MSE_loss={ComputeMSE(model, dataloader, device, BATCH_SIZE):.4e})")