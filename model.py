import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=4, output_dim=3, hidden_dim=64, num_layers=4):
        """
        Optimised PINN to solve coupled ODEs.
        
        Args:
            input_dim (int): Dimension of the input tensor. Set to 4: [t, T0, HI0, HII0] 
            output_dim (int): Dimension of the output tensor. Set to 3: [T, HI, HII]
        """
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
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