import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=4, output_dim=3, hidden_dim=128, hidden_layers=8):
        """
        Optimised PINN to solve coupled ODEs.
        
        Args:
            input_dim (int): Dimension of the input tensor. Set to 4: [t, T0, HI0, HII0] 
            output_dim (int): Dimension of the output tensor. Set to 3: [T, HI, HII]
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
        # To enforce the initial conditions, we format the PINN as follows:
        #   y(t) = y0 + t·PINN(t,y), 
        # where y0 is the initial condition and PINN(t,y) is the output of the network.
        y0 = x[:, 1:]
        t = x[:, 0:1]
        PINN = self.net(x)
        # print("y0=",y0)
        # print("t=",t)
        # print("PINN=",PINN)
        return y0 + t*PINN

# ---------------------------------------------------------
# Testing block
# ---------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if CUDA is globally available
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    
    test_model = PINN().to(device)
    test_input = torch.randn(5, 4).to(device)
    test_output = test_model(test_input)
    
    print(f"✓ Model instantiated successfully.")
    print(f"  Output shape: {test_output.shape} # Expected: [5, 3]")