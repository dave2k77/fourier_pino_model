import torch
import torch.nn as nn

class NeuralOperator(nn.Module):
    """
    A neural network with four complex linear layers and GELU activation functions.
    """

    def __init__(self):
        super(NeuralOperator, self).__init__()

        # GELU activation function
        self.gelu = nn.GELU()

        # First linear layer
        self.fc1 = nn.Linear(64 * 64 * 2, 128)

        # Second linear layer
        self.fc2 = nn.Linear(128, 256)

        # Third linear layer
        self.fc3 = nn.Linear(256, 128)

        # Fourth linear layer and output
        self.fc4 = nn.Linear(128, 64 * 64 * 2)

    def forward(self, x):
        # Convert complex tensor to real-valued tensor with two channels: real and imaginary parts
        x = torch.view_as_real(x)

        # Flatten the input tensor
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor, keeping the batch dimension

        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        x = self.fc4(x)

        # Reshape the output tensor back to the original shape with two channels (real and imaginary parts)
        x = x.view(-1, 64, 64, 2)

        # Convert the real-valued tensor back to a complex tensor
        x = torch.view_as_complex(x)

        return x
