import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from dataset_builder import *

def fourier_derivative_2d(input, axis=0):
    assert input.dim() == 2, "The input must be a 2D tensor."
    assert axis in (0, 1), "The axis must be either 0 or 1."

    # Compute the Fourier Transform of the input
    input_fft = torch.fft.fftn(input)

    # Create a tensor of wave numbers (k) corresponding to the input dimensions
    k = torch.fft.fftfreq(input.shape[axis], dtype=input.dtype, device=input.device)

    # Compute the derivative in the Fourier domain
    if axis == 0:
        k = k.view(-1, 1)  # Reshape k to match the input dimensions
        input_fft *= 1j * k  # Multiply the Fourier coefficients by 1j * k
    else:
        k = k.view(1, -1)  # Reshape k to match the input dimensions
        input_fft *= 1j * k  # Multiply the Fourier coefficients by 1j * k

    # Compute the inverse Fourier Transform to get the derivative in the spatial domain
    derivative = torch.fft.ifftn(input_fft).real

    return derivative


def loss_function(output, target, physics_loss_coefficient):
    output_real = output.real  # Take the real part of the output tensor
    operator_loss = nn.MSELoss()(output_real, target)
    physics_loss = energy_conservation_loss(output_real, target)
    loss = operator_loss + physics_loss_coefficient * physics_loss
    return loss



def energy_conservation_loss(output, target, dx=1, dy=1, alpha=0.1):

    dt = dx ** 2 /(4 * alpha)
    
    # Calculate the time derivative error
    time_derivative_error = torch.abs((output[:, 1:] - output[:, :-1]) / dt - (target[:, 1:] - target[:, :-1]) / dt)

    # Calculate the spatial derivative errors
    batch_size = output.size(0)
    x_derivative_error = torch.zeros_like(time_derivative_error)
    y_derivative_error = torch.zeros_like(time_derivative_error)

    for i in range(batch_size):
        x_derivative_error[i] = torch.abs(fourier_derivative_2d(output[i, :-1], axis=0) - fourier_derivative_2d(target[i, :-1], axis=0))
        y_derivative_error[i] = torch.abs(fourier_derivative_2d(output[i, :-1], axis=1) - fourier_derivative_2d(target[i, :-1], axis=1))

    # Implement the loss based on the energy conservation law
    energy_law_error = ((time_derivative_error - alpha * (x_derivative_error + y_derivative_error)) ** 2).mean()

    return energy_law_error





def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.show()


def train(model, loss_fn, optimizer, train_loader, test_loader, num_epochs, physics_loss_coefficient=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_history = []  # Initialize the loss_history list

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (heatmaps, pde_solutions) in enumerate(train_loader):
            heatmaps = heatmaps.to(device)
            pde_solutions = pde_solutions.to(device)

            optimizer.zero_grad()

            outputs = model(heatmaps)
            loss = loss_function(outputs, pde_solutions, physics_loss_coefficient)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            epoch_loss = running_loss / (i + 1)
            loss_history.append(epoch_loss)  # Append the average loss of the epoch to the loss_history list
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

        # Evaluate on the test dataset
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, (heatmaps, pde_solutions) in enumerate(test_loader):
                heatmaps = heatmaps.to(device)
                pde_solutions = pde_solutions.to(device)

                outputs = model(heatmaps)
                loss = loss_function(outputs, pde_solutions, physics_loss_coefficient)

                test_loss += loss.item()

                test_epoch_loss = test_loss / (i + 1)
                print(f"Test Loss: {test_epoch_loss:.4f}")

    return model, loss_history

