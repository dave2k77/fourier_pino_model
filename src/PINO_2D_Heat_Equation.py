import torch.nn as nn

from fourier_transform_layer import FourierTransformLayer
from inverse_transform_layer import InverseFourierTransformLayer
from neural_operator_layer import NeuralOperator
from preprocess import *

class PINO_2D_Heat_Equation(nn.Module):
    def __init__(self):
        super(PINO_2D_Heat_Equation, self).__init__()
        self.encoder = FourierTransformLayer()
        self.neural_operator = NeuralOperator()
        self.decoder = InverseFourierTransformLayer()

    def forward(self, x):
        x = self.encoder(x)
        x = self.neural_operator(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":

    model = PINO_2D_Heat_Equation()
    data = None # Load your data here
    loss_history = train(model, data, epochs=100, learning_rate=1e-3, physics_loss_coefficient=1e-4)
    plot_loss(loss_history)