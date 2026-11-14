import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=128, hidden_layers=8):
        """
        Optimised PINN to solve coupled ODEs.
        
        Args:
            input_dim (int): Dimension of the input tensor. Set to 3: [log(y(t))], for y = T, nHI, nHII. 
            output_dim (int): Dimension of the output tensor. Set to 3: [log(y(t+dt)/y(t))], for y = T, nHI, nHII.
            hidden_dim (int): Number of neurons in each hidden layer.
            hidden_layers (int): Number of hidden layers in the network.
        """
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)  # restore gain to 1.0
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)