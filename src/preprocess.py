import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from dataset_preprocess import *

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


def train(model, loss_fn, optimizer, train_loader, test_loader, num_epochs, physics_loss_coefficient=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loss_history = []
    test_loss_history = []
    r2_mean = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (heatmaps, pde_solutions) in enumerate(train_loader):
            heatmaps = heatmaps.to(device)
            pde_solutions = pde_solutions.to(device)

            optimizer.zero_grad()

            outputs = model(heatmaps)
            loss_fn = loss_function(outputs, pde_solutions, physics_loss_coefficient)
            loss_fn.backward()
            optimizer.step()

            running_loss += loss_fn.item()

            epoch_loss = running_loss / (i + 1)
            train_loss_history.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

        # Evaluate on the test dataset
        model.eval()
        test_loss = 0.0
        r2_scores = []
        
        with torch.no_grad():
            for i, (heatmaps, pde_solutions) in enumerate(test_loader):
                heatmaps = heatmaps.to(device)
                pde_solutions = pde_solutions.to(device)

                outputs = model(heatmaps)
                loss = loss_function(outputs, pde_solutions, physics_loss_coefficient)
                r2 = r2_score(pde_solutions, outputs.real)

                test_loss += loss.item()
                r2_scores.append(r2)

                test_epoch_loss = test_loss / (i + 1)
                test_loss_history.append(test_epoch_loss)
                mean_r2_score = sum(r2_scores) / len(r2_scores)
                print(f"Test Loss: {test_epoch_loss:.4f}, R-squared Score: {mean_r2_score:.4f}")
                r2_mean += mean_r2_score
    

    return train_loss_history, test_loss_history, r2_mean


def r2_score(y_true, y_pred):
    ss_total = ((y_true - y_true.mean()) ** 2).sum()
    ss_residual = ((y_true - y_pred) ** 2).sum()
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()



def plot_loss(train_loss_history, test_loss_history, save=False, save_path=r'graphs\loss_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(test_loss_history, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss History')
    plt.legend()

    if save:
        # Create the directory if it does not exist
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(save_path)

    plt.show()


def train_and_evaluate(model, loss_fn, optimizer, num_epochs, physics_loss_coefficient):
    # Create and initialize your model, loss function, and optimizer
    # ...

    # Train the model and get the mean R-squared Score
    _, _, mean_r2_score = train(model, loss_fn, optimizer, train_loader, test_loader, num_epochs, physics_loss_coefficient)

    return mean_r2_score

def plot_r2_vs_physics_loss_coefficients(physics_loss_coefficients, train_and_evaluate, model, loss_fn, optimizer, num_epochs):
    r2_scores = []
    for coeff in physics_loss_coefficients:
        r2 = train_and_evaluate(model, loss_fn, optimizer, num_epochs, physics_loss_coefficient=coeff)
        r2_scores.append(r2)

    plt.plot(physics_loss_coefficients, r2_scores)
    plt.xlabel("Physics Loss Coefficient")
    plt.ylabel("R2 Score")
    plt.title("R2 Score vs. Physics Loss Coefficient")
    plt.show()


def compare_solutions(predicted_solution, original_solution, error, time_index):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Predicted Solution")
    plt.imshow(np.abs(predicted_solution[time_index].detach().cpu().numpy()), cmap='viridis', origin='lower')  # Add .cpu().numpy()

    plt.subplot(1, 3, 2)
    plt.title("Original Solution")
    plt.imshow(np.abs(original_solution[time_index].detach().cpu().numpy()), cmap='viridis', origin='lower')  # Add .cpu().numpy()

    plt.subplot(1, 3, 3)
    plt.title("Error")
    plt.imshow(np.abs(error[time_index][0]), cmap='viridis', origin='lower')  # Add .cpu().numpy()

    plt.show()
