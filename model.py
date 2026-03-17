import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super().__init__()
        # 4 inputs: T_init, HI, HII, dt
        # 1 output: T_final (or Delta T)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)