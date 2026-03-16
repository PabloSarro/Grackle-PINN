import torch
from model import CoolingNetwork
from physics import physics_loss

# 1. Setup
model = CoolingNetwork().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# 2. Training Loop (Simplified)
for epoch in range(10):
    # Assume 'inputs' and 'targets' come from your output_1e7.dat
    optimizer.zero_grad()
    
    # Loss 1: Data Loss (The standard NN part)
    predictions = model(inputs)
    loss_data = criterion(predictions, targets)
    
    # Loss 2: Physics Loss (The PINN part)
    loss_physics = physics_loss(model, inputs)
    
    # Total combined loss
    total_loss = loss_data + 0.1 * loss_physics # 0.1 is the 'physics weight'
    
    total_loss.backward()
    optimizer.step()
    
    print(f"Data Loss: {loss_data.item()}, Physics Loss: {loss_physics.item()}")