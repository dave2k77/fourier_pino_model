"""
Metrics and Evaluation Functions for PINO Model

This module provides various evaluation metrics for assessing the performance
of the PINO model on PDE solving tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


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


def calculate_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        MSE value
    """
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    
    return mean_squared_error(target_np, pred_np)


def calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        MAE value
    """
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    
    return mean_absolute_error(target_np, pred_np)


def calculate_relative_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate relative error.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        Relative error
    """
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    
    # Avoid division by zero
    mask = target_np != 0
    if not np.any(mask):
        return 0.0
    
    relative_error = np.mean(np.abs((pred_np[mask] - target_np[mask]) / target_np[mask]))
    return relative_error


def calculate_physics_accuracy(predictions: torch.Tensor, targets: torch.Tensor, 
                              alpha: float = 0.1, dx: float = 1.0, dy: float = 1.0) -> float:
    """
    Calculate physics accuracy based on heat equation conservation.
    
    Args:
        predictions: Model predictions
        targets: Target values
        alpha: Thermal diffusivity coefficient
        dx: Spatial step in x direction
        dy: Spatial step in y direction
        
    Returns:
        Physics accuracy score
    """
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    dt = dx ** 2 / (4 * alpha)
    
    # Calculate time derivatives
    pred_time_deriv = (pred_np[:, 1:] - pred_np[:, :-1]) / dt
    target_time_deriv = (target_np[:, 1:] - target_np[:, :-1]) / dt
    
    # Calculate spatial derivatives (simplified)
    pred_spatial_deriv = np.gradient(pred_np[:, :-1], axis=(1, 2))
    target_spatial_deriv = np.gradient(target_np[:, :-1], axis=(1, 2))
    
    # Physics error
    physics_error = np.mean((pred_time_deriv - alpha * (pred_spatial_deriv[0] + pred_spatial_deriv[1])) ** 2)
    target_physics_error = np.mean((target_time_deriv - alpha * (target_spatial_deriv[0] + target_spatial_deriv[1])) ** 2)
    
    # Physics accuracy (lower is better)
    physics_accuracy = 1.0 / (1.0 + physics_error / (target_physics_error + 1e-8))
    
    return physics_accuracy


def calculate_all_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                         alpha: float = 0.1) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Target values
        alpha: Thermal diffusivity coefficient
        
    Returns:
        Dictionary containing all metrics
    """
    # Get real parts if predictions are complex
    if torch.is_complex(predictions):
        predictions_real = predictions.real
    else:
        predictions_real = predictions
    
    metrics = {
        'r2_score': calculate_r2_score(predictions_real, targets),
        'mse': calculate_mse(predictions_real, targets),
        'mae': calculate_mae(predictions_real, targets),
        'relative_error': calculate_relative_error(predictions_real, targets),
        'physics_accuracy': calculate_physics_accuracy(predictions_real, targets, alpha)
    }
    
    return metrics


def evaluate_model_performance(model: torch.nn.Module, 
                              test_loader: torch.utils.data.DataLoader,
                              device: torch.device,
                              alpha: float = 0.1) -> Dict[str, float]:
    """
    Evaluate model performance on test dataset.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        alpha: Thermal diffusivity coefficient
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for heatmaps, targets in test_loader:
            heatmaps = heatmaps.to(device)
            targets = targets.to(device)
            
            predictions = model(heatmaps)
            
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_all_metrics(all_predictions, all_targets, alpha)
    
    return metrics


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """
    Print a formatted summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    
    metric_names = {
        'r2_score': 'R² Score',
        'mse': 'Mean Squared Error',
        'mae': 'Mean Absolute Error',
        'relative_error': 'Relative Error',
        'physics_accuracy': 'Physics Accuracy'
    }
    
    for key, value in metrics.items():
        name = metric_names.get(key, key.replace('_', ' ').title())
        if key in ['r2_score', 'physics_accuracy']:
            print(f"{name:20}: {value:.4f}")
        else:
            print(f"{name:20}: {value:.6f}")
    
    print("="*50)


def compare_models_performance(model_results: Dict[str, Dict[str, float]]) -> None:
    """
    Compare performance of multiple models.
    
    Args:
        model_results: Dictionary mapping model names to their metrics
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Get all metric names
    all_metrics = set()
    for metrics in model_results.values():
        all_metrics.update(metrics.keys())
    
    # Print header
    header = f"{'Model':<20}"
    for metric in sorted(all_metrics):
        header += f"{metric:>12}"
    print(header)
    print("-" * 80)
    
    # Print results for each model
    for model_name, metrics in model_results.items():
        row = f"{model_name:<20}"
        for metric in sorted(all_metrics):
            value = metrics.get(metric, float('nan'))
            if metric in ['r2_score', 'physics_accuracy']:
                row += f"{value:>12.4f}"
            else:
                row += f"{value:>12.6f}"
        print(row)
    
    print("="*80)
