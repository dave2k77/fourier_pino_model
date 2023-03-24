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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Replace these paths with the correct ones for your dataset
    heatmap_folder = r"images\heatmaps"
    pde_solution_folder = r"images\pde_solutions"

    # Initilise PINO 2D Heat Equation Model
    model = PINO_2D_Heat_Equation()
    model.to(device)

    # Load training and test data
    data = HeatmapPDEDataset(heatmap_folder, pde_solution_folder)
    train_dataset, test_dataset = split_data(data)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    # Set hyperparameters
    num_epochs = 100  # options: 100, 250, 500

    # physics loss coefficients
    physics_loss_coefficient_A = 0.001
    physics_loss_coefficient_B = 0.01 
    physics_loss_coefficient_C = 0.1

    # Learning Rates
    lr_1 = 0.001
    lr_2 = 0.005
    lr_3 = 0.01

    # Experiment A Optimizers
    optimizerA_1 = optim.SGD(model.parameters(), lr=lr_1)
    optimizerA_2 = optim.SGD(model.parameters(), lr=lr_2)
    optimizerA_3 = optim.SGD(model.parameters(), lr=lr_3)

    # Experiment B Optimzers
    optimizerB_1 = optim.Adam(model.parameters(), lr=lr_1)
    optimizerB_2 = optim.Adam(model.parameters(), lr=lr_2)
    optimizerB_3 = optim.Adam(model.parameters(), lr=lr_3)
    
    # loss function
    loss_fn = loss_function

    # train the model
    train_loss_history, test_loss_history, mean_r2_score = train(model=model, loss_fn=loss_fn, optimizer=optimizerB_2,
                                                  train_loader=train_loader, test_loader=test_loader,
                                                  num_epochs=num_epochs, physics_loss_coefficient=physics_loss_coefficient_B)
    plot_loss(train_loss_history, test_loss_history, save=True, save_path=r'graphs\train_test-phyloss-01.png')

    time_index = 0  # Choose the time step you want to analyze

    heatmaps_test, labels_test = next(iter(test_loader))
    heatmaps_test = heatmaps_test.to(device)
    labels_test = labels_test.to(device)
    error = torch.abs(heatmaps_test - labels_test)

    # Get the model predictions for the test data
    predictions_test = model(heatmaps_test)

    # Convert tensors to numpy arrays for plotting
    heatmaps_test_np = heatmaps_test.detach().cpu().numpy()
    labels_test_np = labels_test.detach().cpu().numpy()
    predictions_test_np = predictions_test.detach().cpu().numpy()
    error_np = error.detach().cpu().numpy()

    compare_solutions(predictions_test, predictions_test, error_np, time_index)

    physics_loss_coefficients = [0.001, 0.01, 0.1]
    plot_r2_vs_physics_loss_coefficients(physics_loss_coefficients, train_and_evaluate, model, loss_fn, optimizer=optimizerB_2, num_epochs=num_epochs)


