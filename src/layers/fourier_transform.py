"""
Fourier Transform Layer for PINO Model

This module implements the encoder component of the PINO architecture,
transforming spatial domain data to the frequency domain using 2D FFT.
"""

import torch
import torch.nn as nn


class FourierTransformLayer(nn.Module):
    """
    Fourier Transform Layer for PINO Model.
    
    This layer applies 2D Fast Fourier Transform (FFT) to transform
    input data from the spatial domain to the frequency domain.
    This transformation simplifies the computational requirements
    by reducing the complexity of the problem.
    """
    
    def __init__(self):
        """Initialize the Fourier Transform Layer."""
        super(FourierTransformLayer, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D Fourier Transform to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Complex tensor of same shape as input in frequency domain
            
        Raises:
            ValueError: If input tensor is not 4D
        """
        # Validate input dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D tensor")
        
        # Apply 2D FFT to each sample in the batch
        # torch.fft.fft2 applies FFT to the last two dimensions
        x_fft = torch.fft.fft2(x)
        
        return x_fft
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return "FourierTransformLayer()"
