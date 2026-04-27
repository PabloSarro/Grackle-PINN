import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import GrackleDataset
from model import PINN
from physics import GrackleRates, PhysicsLossManager
import time

# Initial Configuration and Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_NAME = "PINN.pth"
BEST_NAME = "PINN_BEST.pth"
DATA_PATH = "Grackle_GTs/"    # Path where output_*.dat are located

BATCH_SIZE = 8192 # 16384
LEARNING_RATE = 1.0e-4
EPOCHS = 250
LAMBDA_PHYS = 1.0

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
    'x_min': dataset.x_min,
    'x_max': dataset.x_max,
    'y_min': dataset.y_min,
    'y_max': dataset.y_max
}, f"scaling_params_{MODEL_NAME}")
print(f"Scaling parameters saved to scaling_params_{MODEL_NAME}")

# Model, Optimiser, and Loss
model = PINN(input_dim=4, output_dim=3, hidden_dim=128, hidden_layers=8).to(device)
grackle_phys = GrackleRates()
phys_manager = PhysicsLossManager(
    model, 
    grackle_phys, 
    x_min=dataset.x_min.to(device), 
    x_max=dataset.x_max.to(device),
    y_min=dataset.y_min.to(device),
    y_max=dataset.y_max.to(device)
)
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # DEBUG: Try Ethan's suggestion!
criterion = nn.MSELoss()

print(f"Starting training on {len(dataset)} points...")

# 5. Training Loop
best_loss = float('inf')

start_time = time.time()
for epoch in range(EPOCHS):
    # print(f"Epoch {epoch+1}")
    model.train()
    total_data_loss = 0
    total_phys_loss = 0
    # batch_num = 0
    for batch_x, batch_y in dataloader:
        # print(f"Batch {batch_num+1}")
        # batch_num += 1
        # Move data to GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        batch_x.requires_grad = True
        optimiser.zero_grad()
        
        # Forward pass and losses
        predictions = model(batch_x)
        loss_data = criterion(predictions, batch_y)
        loss_phys = phys_manager.get_residuals(batch_x, predictions)
        
        # Total combined loss with weighting
        loss = loss_data + LAMBDA_PHYS * loss_phys

        # --- DEBUG BLOCK ---
        if torch.isnan(loss):
            print(f"\n[!] NAN DETECTED at Epoch {epoch+1}")#, Batch {batch_num}")
            print(f"Data Loss: {loss_data.item():.4e}")
            print(f"Phys Loss: {loss_phys.item():.4e}")
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

        # # --- DETAILED GRADIENT INSPECTION ---
        # print("\n--- Gradient Breakdown ---")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_max = param.grad.data.abs().max().item()
        #         grad_mean = param.grad.data.abs().mean().item()
        #         print(f"Layer: {name:10} | Max Grad: {grad_max:.2e} | Mean Grad: {grad_mean:.2e}")
        #     else:
        #         print(f"Layer: {name:10} | No Gradient")

        # # --- GRADIENT MONITORING BLOCK ---
        # total_norm = 0.0
        # max_grad = 0.0
        # for p in model.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2).item()
        #         total_norm += param_norm ** 2
        #         max_grad = max(max_grad, p.grad.data.abs().max().item())
        # total_norm = total_norm ** 0.5

        # if torch.isnan(torch.tensor(total_norm)) or total_norm > 1e6:
        #     print(f"\n[!] EXPLODING GRADIENT DETECTED")
        #     print(f"    Total Gradient Norm: {total_norm:.4e}")
        #     print(f"    Max Absolute Gradient: {max_grad:.4e}")
        #     print(f"    Current Data Loss: {loss_data.item():.4e}")
        #     print(f"    Current Phys Loss: {loss_phys.item():.4e}")
            
            # This identifies which loss is driving the explosion
            # If phys_loss is 1e20, it's the physics constraints

        # Temporarily calculate gradients for one component at a time to debug
        # (Do this only if the main loss is NaN)
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm usually set to 0.5 to 10 times the average gradient norm. Consider increasing to 10 or 20 if PINN clips gradients too aggressively.
        optimiser.step()
        
        total_data_loss += loss_data.item()
        total_phys_loss += loss_phys.item()
    
    avg_data_loss = total_data_loss / len(dataloader)
    avg_phys_loss = total_phys_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Data Loss: {avg_data_loss:.6e}, Avg Physics Loss: {avg_phys_loss:.6e}")

    # Save the best model so far.
    avg_loss = avg_data_loss + LAMBDA_PHYS * avg_phys_loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), BEST_NAME)
        print(f"[✓] New best model saved (avg_loss={best_loss:.4e})")


end_time = time.time()
print(f"Training complete in {end_time - start_time:.2f} seconds.")
torch.save(model.state_dict(), MODEL_NAME)
print(f"[✓] Final model saved (avg_loss={avg_loss:.4e})")