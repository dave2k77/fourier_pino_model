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
    physics_loss_coefficient = 0.001  # options: 0.01, 0.1, 1.0, 3.0, 5.0
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = loss_function

    train_loss_history, test_loss_history, mean_r2_score = train(model=model, loss_fn=loss_fn, optimizer=optimizer,
                                                  train_loader=train_loader, test_loader=test_loader,
                                                  num_epochs=num_epochs, physics_loss_coefficient=physics_loss_coefficient)
    plot_loss(train_loss_history, test_loss_history, save=True, save_path=r'graphs\train_test-phyloss-01.png')

# Assuming predicted_solution and original_solution are NumPy arrays with the shape (num_time_steps, height, width)

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

    physics_loss_coefficients = [0.001, 0.01, 0.1, 1.0, 3.0]
    plot_r2_vs_physics_loss_coefficients(physics_loss_coefficients, train_and_evaluate, model, loss_fn, optimizer, num_epochs)


