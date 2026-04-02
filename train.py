import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import GrackleDataset
from model import PINN
from physics import GrackleRates, PhysicsLossManager
import time

# Initial Configuration and Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_PATH = "Grackle_GTs/"    # Path where output_*.dat are located
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
EPOCHS = 100
LAMBDA_PHYS = 0.1  # Weight used for the physics loss

# Data Loading
dataset = GrackleDataset(folder_path=DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Optimiser, and Loss
model = PINN(input_dim=4, output_dim=3).to(device)
grackle_phys = GrackleRates()
phys_manager = PhysicsLossManager(
    model, 
    grackle_phys, 
    x_min=dataset.x_min.to(device), 
    x_max=dataset.x_max.to(device)
)
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

print(f"Starting training on {len(dataset)} points...")

# 5. Training Loop
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_data_loss = 0
    total_phys_loss = 0
    
    for batch_x, batch_y in dataloader:
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
        
        # Backward pass
        loss.backward()
        optimiser.step()
        
        total_data_loss += loss_data.item()
        total_phys_loss += loss_phys.item()
    
    if (epoch + 1) % 10 == 0:
        avg_data_loss = total_data_loss / len(dataloader)
        avg_phys_loss = total_phys_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Data Loss: {avg_data_loss:.6e}, Avg Physics Loss: {avg_phys_loss:.6e}")


end_time = time.time()
print(f"Training complete in {end_time - start_time:.2f} seconds.")
torch.save(model.state_dict(), "PINN.pth")
print("Training complete. Model saved as PINN.pth")