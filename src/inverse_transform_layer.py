import torch
import torch.nn as nn

class InverseFourierTransformLayer(nn.Module):
    """
    This class implements a simple decoder neural network that returns 
    the output from operator network from the frequency domain back to the 
    the spatial domain using the 2d inverse fourier transform.
    """

    def __init__(self):
        
        super(InverseFourierTransformLayer, self).__init__()


    def forward(self, x):

        # Perform 2D Inverse Fourier Transform
        x_ifft = torch.fft.ifft2(x)

        return x_ifft