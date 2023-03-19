import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def preprocess_data(raw_input_data, device='cpu'):
    """
    This function is a helper function that formats and reshapes the raw input data into a format 
    that can be used by the Fourier Transform Layer.
    """

    # Convert input data to a PyTorch tensor
    formatted_input_data_tensor = torch.tensor(raw_input_data, dtype=torch.float32, device=device)

    # Reshape
    formatted_input_data_tensor = formatted_input_data_tensor.unsqueeze(0).unsqueeze(-1)

    return formatted_input_data_tensor


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


def loss_function(output, target, initial_condition, physics_loss_coefficient):
    operator_loss = nn.MSELoss()(output, target)
    initial_condition_loss = torch.norm(initial_condition, p="2")
    physics_loss = energy_conservation_loss(output, target)
    loss = operator_loss + initial_condition_loss + physics_loss_coefficient * physics_loss
    return loss


def energy_conservation_loss(output, target, dx=1, dy=1, alpha=0.1):

    dt = dx ** 2 /(4 * alpha)
    
    # Calculate the time derivative error
    time_derivative_error = torch.abs((output[:, 1:] - output[:, :-1]) / dt - (target[:, 1:] - target[:, :-1]) / dt)

    # Calculate the spatial derivative errors
    x_derivative_error = torch.abs(fourier_derivative_2d(output, axis=0) - fourier_derivative_2d(target, axis=0))
    y_derivative_error = torch.abs(fourier_derivative_2d(output, axis=1) - fourier_derivative_2d(target, axis=1))

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


def train(model, data, epochs, learning_rate, physics_loss_coefficient):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        input_data, initial_condition, target = preprocess_data(data)
        output = model(input_data)
        loss = loss_function(output, target, initial_condition, physics_loss_coefficient)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
    return loss_history