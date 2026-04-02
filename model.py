import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=4, output_dim=3, hidden_layers=[64, 64]):
        """
        Optimised PINN to solve coupled ODEs.
        
        Args:
            input_dim (int): Dimension of the input tensor. Set to 4: [t, T0, HI0, HII0] 
            output_dim (int): Dimension of the output tensor. Set to 3: [T, HI, HII]
            hidden_layers (list): List of integers representing the number of neurons in each hidden layer.
        """
        super(PINN, self).__init__()
        
        dims = [input_dim] + hidden_layers + [output_dim]
        self.linears = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.linears.append(nn.Linear(dims[i], dims[i+1]))
            
        # Tanh activation for smooth higher-order derivatives
        self.activation = nn.Tanh()
        
        # Xavier/Glorot weight initialization (standard for Tanh activation)
        self._init_weights()

    def _init_weights(self):
        for m in self.linears:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the network.
        """
        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
            
        # The last layer has NO activation. 
        # The network must be able to freely predict any value in the log10 scale.        
        return self.linears[-1](x) 
    

# ---------------------------------------------------------
# Testing block
# ---------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if CUDA is globally available
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    
    test_model = PINN().to(device)
    test_input = torch.randn(64, 4).to(device)
    test_output = test_model(test_input)
    
    print(f"✓ Model instantiated successfully.")
    print(f"  Input shape: {test_input.shape} # Expected: [64, 4]")
    print(f"  Output shape: {test_output.shape} # Expected: [64, 3]")