"""
Training Utilities for PINO Model

This module provides training functions and loss computations for the PINO model,
including physics-informed loss functions and Fourier-based derivatives.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional, Callable
import numpy as np
from tqdm import tqdm


def fourier_derivative_2d(input_tensor: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    Compute 2D derivative using Fourier transform.
    
    Args:
        input_tensor: 2D input tensor
        axis: Axis along which to compute derivative (0 or 1)
        
    Returns:
        Derivative tensor
        
    Raises:
        ValueError: If input is not 2D or axis is invalid
    """
    if input_tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {input_tensor.dim()}D")
    if axis not in (0, 1):
        raise ValueError(f"Axis must be 0 or 1, got {axis}")
    
    # Compute the Fourier Transform of the input
    input_fft = torch.fft.fftn(input_tensor)
    
    # Create wave numbers tensor
    k = torch.fft.fftfreq(input_tensor.shape[axis], 
                         dtype=input_tensor.dtype, 
                         device=input_tensor.device)
    
    # Compute derivative in Fourier domain
    if axis == 0:
        k = k.view(-1, 1)
    else:
        k = k.view(1, -1)
    
    input_fft *= 1j * k
    
    # Compute inverse Fourier Transform
    derivative = torch.fft.ifftn(input_fft).real
    
    return derivative


def energy_conservation_loss(output: torch.Tensor, target: torch.Tensor,
                           dx: float = 1.0, dy: float = 1.0, alpha: float = 0.1) -> torch.Tensor:
    """
    Compute simplified energy conservation loss for the heat equation.
    
    Args:
        output: Model output tensor
        target: Target tensor
        dx: Spatial step in x direction
        dy: Spatial step in y direction
        alpha: Thermal diffusivity coefficient
        
    Returns:
        Energy conservation loss
    """
    # Simplified physics loss: just compare the spatial gradients
    # This is a baseline implementation for testing
    
    # Calculate spatial gradients using finite differences
    batch_size = output.size(0)
    total_loss = 0.0
    
    for i in range(batch_size):
        # Get 2D slices (remove channel dimension)
        output_2d = output[i, 0]  # Shape: (H, W)
        target_2d = target[i, 0]  # Shape: (H, W)
        
        # Calculate gradients using finite differences
        output_grad_x = torch.diff(output_2d, dim=0, prepend=output_2d[0:1, :])
        output_grad_y = torch.diff(output_2d, dim=1, prepend=output_2d[:, 0:1])
        target_grad_x = torch.diff(target_2d, dim=0, prepend=target_2d[0:1, :])
        target_grad_y = torch.diff(target_2d, dim=1, prepend=target_2d[:, 0:1])
        
        # Physics loss: gradient consistency
        grad_loss = torch.mean((output_grad_x - target_grad_x) ** 2 + 
                              (output_grad_y - target_grad_y) ** 2)
        
        total_loss += grad_loss
    
    return total_loss / batch_size


def loss_function(output: torch.Tensor, target: torch.Tensor, 
                 physics_loss_coefficient: float = 0.001) -> torch.Tensor:
    """
    Combined loss function for PINO model.
    
    Args:
        output: Model output (complex tensor)
        target: Target values (real tensor)
        physics_loss_coefficient: Weight for physics loss component
        
    Returns:
        Combined loss value
    """
    output_real = output.real
    operator_loss = nn.MSELoss()(output_real, target)
    physics_loss = energy_conservation_loss(output_real, target)
    total_loss = operator_loss + physics_loss_coefficient * physics_loss
    
    return total_loss


def train(model: nn.Module, 
          loss_fn: Callable, 
          optimizer: optim.Optimizer, 
          train_loader: DataLoader, 
          test_loader: DataLoader, 
          num_epochs: int, 
          physics_loss_coefficient: float = 0.001,
          device: Optional[torch.device] = None,
          verbose: bool = True) -> Tuple[List[float], List[float], float]:
    """
    Train the PINO model.
    
    Args:
        model: PINO model to train
        loss_fn: Loss function
        optimizer: Optimizer
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        physics_loss_coefficient: Weight for physics loss
        device: Device to train on (auto-detect if None)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (train_loss_history, test_loss_history, mean_r2_score)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    train_loss_history = []
    test_loss_history = []
    r2_scores = []
    
    if verbose:
        print(f"Training on device: {device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Physics loss coefficient: {physics_loss_coefficient}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") if verbose else train_loader
        
        for i, (heatmaps, pde_solutions) in enumerate(train_pbar):
            heatmaps = heatmaps.to(device)
            pde_solutions = pde_solutions.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(heatmaps)
            loss = loss_fn(outputs, pde_solutions, physics_loss_coefficient)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if verbose and isinstance(train_pbar, tqdm):
                train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        epoch_train_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        epoch_r2_scores = []
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]") if verbose else test_loader
            
            for heatmaps, pde_solutions in test_pbar:
                heatmaps = heatmaps.to(device)
                pde_solutions = pde_solutions.to(device)
                
                outputs = model(heatmaps)
                loss = loss_fn(outputs, pde_solutions, physics_loss_coefficient)
                test_loss += loss.item()
                
                # Calculate R² score
                outputs_real = outputs.real
                r2_score = calculate_r2_score(outputs_real, pde_solutions)
                epoch_r2_scores.append(r2_score)
        
        epoch_test_loss = test_loss / len(test_loader)
        test_loss_history.append(epoch_test_loss)
        
        mean_r2 = np.mean(epoch_r2_scores)
        r2_scores.append(mean_r2)
        
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Test Loss: {epoch_test_loss:.4f}, "
                  f"R² Score: {mean_r2:.4f}")
    
    final_mean_r2 = np.mean(r2_scores)
    
    if verbose:
        print(f"Training completed. Final mean R² score: {final_mean_r2:.4f}")
    
    return train_loss_history, test_loss_history, final_mean_r2


def calculate_r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate R² score for model predictions.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        R² score
    """
    # Convert to numpy for calculation
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    
    # Calculate R²
    ss_res = np.sum((target_np - pred_np) ** 2)
    ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return r2
