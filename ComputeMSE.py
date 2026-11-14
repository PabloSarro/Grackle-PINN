import time
import torch
import torch.nn as nn

# from data_utils import GrackleDataset
# from torch.utils.data import DataLoader

# dataset = GrackleDataset(folder_path=data_path)
# dataloader = DataLoader(
#     dataset, 
#     batch_size=BATCH_SIZE, 
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True
# )

def ComputeMSE(model, dataloader, device):
    """
    Computes the Mean Squared Error (MSE) across an entire dataset.
    
    Args:
        model: Your trained PyTorch PINN model.
        dataset: The GrackleDataset instance.
        device: 'cuda' or 'cpu'.
        batch_size: Keep this large for fast evaluation.
        
    Returns:
        float: The exact average MSE across all points.
    """
    # 1. Create a non-shuffled dataloader for evaluation
    criterion = nn.MSELoss()
    
    total_squared_error = 0.0
    total_samples = 0
    
    # 2. Lock the model into evaluation mode (disables dropout/batchnorm updates)
    model.eval()
    
    # 3. CRITICAL: Disable gradient tracking to prevent RAM explosion
    # start = time.time()
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x)
            
            # nn.MSELoss returns the MEAN of the batch.
            batch_mse = criterion(predictions, batch_y).item()
            
            # Multiply back by batch size to get the absolute sum of errors
            current_batch_size = batch_x.size(0)
            total_squared_error += batch_mse * current_batch_size
            total_samples += current_batch_size
            # print(f"Count: {count}")
            
    # 4. Divide by total exact samples for the true global MSE
    global_mse = total_squared_error / total_samples
    # end = time.time()
    # print(f"Time taken to loop: {end-start} seconds.")
    return global_mse

from data_utils import GrackleDataset
from torch.utils.data import DataLoader
from model import PINN

if __name__ == "__main__":

    dataset = GrackleDataset(folder_path='Valid_Stratified/')
    dataloader = DataLoader(
        dataset, 
        batch_size=4096, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(input_dim=3, output_dim=3, hidden_dim=256, hidden_layers=8).to(device)
    model.load_state_dict(torch.load('ResultsToDownload/PINN_noClamp_Long.pth', map_location=device, weights_only=True)) 
    print(f"MSE: {ComputeMSE(model, dataloader, device)}")