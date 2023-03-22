import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_builder import HeatmapPDEDataset, split_data
from preprocess import *
from fourier_transform_layer import FourierTransformLayer
from inverse_transform_layer import InverseFourierTransformLayer
from neural_operator_layer import NeuralOperator


class PINO_2D_Heat_Equation(nn.Module):
    def __init__(self):
        super(PINO_2D_Heat_Equation, self).__init__()
        self.encoder = FourierTransformLayer()
        self.neural_operator = NeuralOperator()
        self.decoder = InverseFourierTransformLayer()

    def forward(self, x):
        x = self.encoder(x)  # returns x: complex float
        x = self.neural_operator(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    # Replace these paths with the correct ones for your dataset
    heatmap_folder = r"images\heatmaps"
    pde_solution_folder = r"images\pde_solutions"

    # Initilise PINO 2D Heat Equation Model
    model = PINO_2D_Heat_Equation()

    # Load training and test data
    data = HeatmapPDEDataset(heatmap_folder, pde_solution_folder)
    train_dataset, test_dataset = split_data(data)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    # Set hyperparameters
    num_epochs = 100  # options: 100, 250, 500
    physics_loss_coefficient = 5  # options: 0.01, 0.1, 1.0, 3.0, 5.0
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = loss_function

    train_loss_history, test_loss_history = train(model=model, loss_fn=loss_fn, optimizer=optimizer,
                                                  train_loader=train_loader, test_loader=test_loader,
                                                  num_epochs=num_epochs, physics_loss_coefficient=physics_loss_coefficient)
    plot_loss(train_loss_history, test_loss_history, save=True, save_path=r'graphs\train_test-phyloss-01.png')

