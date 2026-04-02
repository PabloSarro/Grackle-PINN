import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=4, output_dim=3):
        """
        Optimised PINN to solve coupled ODEs.
        
        Args:
            input_dim (int): Dimension of the input tensor. Set to 4: [t, T0, HI0, HII0] 
            output_dim (int): Dimension of the output tensor. Set to 3: [T, HI, HII]
        """
        super(PINN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, output_dim)
            
        # Tanh activation for smooth higher-order derivatives
        self.activation = nn.Tanh()
        
        # Manual weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_normal_(m.weight, gain=0.1)  # Xavier initialization with reduced gain for stability
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the network.
        """
        x = self.fc1(x)
        x = self.activation(x)
        
        # Output layer (no activation, allows full log-scale range)
        x = self.fc2(x)
        return x

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