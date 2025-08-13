"""
Neural Operator Layer for PINO Model

This module implements the core neural operator component of the PINO architecture,
learning the mapping between PDE space and latent solution space.
"""

import torch
import torch.nn as nn
from typing import List


class NeuralOperator(nn.Module):
    """
    Neural Operator Network for PINO Model.
    
    This is the core component of the PINO model, designed to learn and approximate
    the underlying PDE operator and the underlying physics of the problem.
    
    The network consists of fully connected layers with GELU activation functions,
    processing complex-valued tensors by separating real and imaginary parts.
    """
    
    def __init__(self, input_size: int = 64, hidden_dims: List[int] = [128, 256, 128]):
        """
        Initialize the Neural Operator.
        
        Args:
            input_size: Size of the input grid (assumed square)
            hidden_dims: List of hidden layer dimensions
        """
        super(NeuralOperator, self).__init__()
        
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        
        # Calculate input and output dimensions
        # Complex numbers have 2 components (real, imaginary)
        input_dim = input_size * input_size * 2
        output_dim = input_size * input_size * 2
        
        # Build the network architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU()
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural operator.
        
        Args:
            x: Complex input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Complex output tensor of same shape as input
            
        Raises:
            ValueError: If input tensor is not 4D
        """
        # Validate input dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D tensor")
        
        batch_size = x.shape[0]
        
        # Convert complex tensor to real-valued tensor with two channels
        # (real and imaginary parts)
        x_real = torch.view_as_real(x)
        
        # Flatten the input tensor, keeping the batch dimension
        x_flat = torch.flatten(x_real, start_dim=1)
        
        # Pass through the neural network
        x_out = self.network(x_flat)
        
        # Reshape the output tensor back to the original shape
        # with two channels (real and imaginary parts)
        x_out = x_out.view(batch_size, self.input_size, self.input_size, 2)
        
        # Convert the real-valued tensor back to a complex tensor
        x_complex = torch.view_as_complex(x_out)
        
        return x_complex
    
    def get_layer_info(self) -> dict:
        """
        Get information about the neural operator architecture.
        
        Returns:
            Dictionary containing layer information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "input_size": self.input_size,
            "hidden_dims": self.hidden_dims,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "activation": "GELU"
        }
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return f"NeuralOperator(input_size={self.input_size}, hidden_dims={self.hidden_dims})"
