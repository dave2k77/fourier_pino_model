import torch
import torch.nn as nn
import numpy as np

class FourierTransformLayer(nn.Module):
    """
    This class is a simple encoder neural network that takes the spatial and temporal coordinates 
    as input and maps them to a higher-dimensional feature space using the 2d fourier transform. 
    """

    def __init__(self):
        
        super(FourierTransformLayer, self).__init__()

 
    def forward(self, x):

        # Perform 2D Fourier Transform
        x_fft = torch.fft.fft2(x)

        return x_fft
    
