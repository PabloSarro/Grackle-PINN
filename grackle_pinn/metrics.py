import torch
import torch.nn as nn


def ComputeMSE(model, dataloader, device):
    """Exact mean squared error of the model over a whole dataloader (standardised units).

    Teacher-forced one-step error: a diagnostic only, not a rollout metric (see TASKS.md, section 0.2).
    """
    criterion = nn.MSELoss()
    total_squared_error = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_x)
            n = batch_x.size(0)
            total_squared_error += criterion(predictions, batch_y).item() * n
            total_samples += n
    return total_squared_error / total_samples
