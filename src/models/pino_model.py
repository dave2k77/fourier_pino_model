"""
Physics-Informed Neural Operator (PINO) for 2D Heat Equation

This module implements the complete PINO architecture combining Fourier transform
layers with neural operators for solving partial differential equations.
"""

import torch
import torch.nn as nn
from ..layers import FourierTransformLayer, NeuralOperator, InverseFourierTransformLayer


class PINO_2D_Heat_Equation(nn.Module):
    """
    Physics-Informed Neural Operator for 2D Heat Equation.
    
    This model combines Fourier analysis with neural operators to solve
    the 2D heat equation efficiently.
    
    Architecture:
        - Encoder: Fourier Transform Layer
        - Neural Operator: Multi-layer perceptron with GELU activation
        - Decoder: Inverse Fourier Transform Layer
    """
    
    def __init__(self, input_size: int = 64, hidden_dims: list = [128, 256, 128]):
        """
        Initialize the PINO model.
        
        Args:
            input_size: Size of the input grid (assumed square)
            hidden_dims: List of hidden layer dimensions for the neural operator
        """
        super(PINO_2D_Heat_Equation, self).__init__()
        
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        
        # Initialize the three main components
        self.encoder = FourierTransformLayer()
        self.neural_operator = NeuralOperator(
            input_size=input_size,
            hidden_dims=hidden_dims
        )
        self.decoder = InverseFourierTransformLayer()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PINO model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Predicted solution tensor of same shape as input
        """
        # Validate input
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        
        # Pass through encoder (spatial to frequency domain)
        x = self.encoder(x)
        
        # Pass through neural operator (learn PDE operator)
        x = self.neural_operator(x)
        
        # Pass through decoder (frequency to spatial domain)
        x = self.decoder(x)
        
        return x
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "PINO_2D_Heat_Equation",
            "input_size": self.input_size,
            "hidden_dims": self.hidden_dims,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "encoder": "FourierTransformLayer",
            "neural_operator": f"NeuralOperator({self.hidden_dims})",
            "decoder": "InverseFourierTransformLayer"
        }
