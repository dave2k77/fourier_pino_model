"""
Inverse Fourier Transform Layer for PINO Model

This module implements the decoder component of the PINO architecture,
transforming frequency domain data back to the spatial domain using 2D IFFT.
"""

import torch
import torch.nn as nn


class InverseFourierTransformLayer(nn.Module):
    """
    Inverse Fourier Transform Layer for PINO Model.
    
    This layer applies 2D Inverse Fast Fourier Transform (IFFT) to transform
    output data from the frequency domain back to the spatial domain.
    This is the final step in the PINO architecture, producing the
    predicted solution to the PDE.
    """
    
    def __init__(self):
        """Initialize the Inverse Fourier Transform Layer."""
        super(InverseFourierTransformLayer, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D Inverse Fourier Transform to input tensor.
        
        Args:
            x: Complex input tensor of shape (batch_size, channels, height, width)
               in frequency domain
               
        Returns:
            Complex tensor of same shape as input in spatial domain
            
        Raises:
            ValueError: If input tensor is not 4D
        """
        # Validate input dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D tensor")
        
        # Apply 2D IFFT to each sample in the batch
        # torch.fft.ifft2 applies IFFT to the last two dimensions
        x_ifft = torch.fft.ifft2(x)
        
        return x_ifft
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return "InverseFourierTransformLayer()"
