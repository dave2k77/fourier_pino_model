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

    # Initilise PINO 2D Heat Equation Model
    model = PINO_2D_Heat_Equation()

    # Load training and test data
    data = HeatmapPDEDataset(heatmap_folder, pde_solution_folder)
    train_dataset, test_dataset = split_data(dataset)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Set hyperparameters
    lr = 0.001 # options: 0.001, 0.005, 0.01, 0.05, 0.1
    num_epochs = 100 # options: 100, 250, 500
    physics_loss_coefficient = 0.001 # options: 0.01, 0.1, 1.0, 2.0
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = loss_function


    loss_history = train(model, loss_fn, optimizer, train_loader, test_loader, num_epochs, physics_loss_coefficient=physics_loss_coefficient)
    plot_loss(loss_history)