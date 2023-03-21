import torch
import torch.nn as nn

class NeuralOperator(nn.Module):
    """
    A neural network with four fully connected layers and GELU activation functions.
    """

    def __init__(self):
        super(NeuralOperator, self).__init__()

        # GELU activation function
        self.gelu = nn.GELU()

        # First fully connected layer
        self.fc1 = nn.Linear(64, 128)

        # Second fully connected layer
        self.fc2 = nn.Linear(128, 256)

        # Third fully connected layer
        self.fc3 = nn.Linear(256, 128)

        # Fourth fully connected layer and output
        self.fc4 = nn.Linear(128, 64)
   

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        x = self.fc4(x)  

        return x