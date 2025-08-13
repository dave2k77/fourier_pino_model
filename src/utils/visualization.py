"""
Visualization Utilities for PINO Model

This module provides plotting and visualization functions for the PINO model,
including training curves, solution comparisons, and heatmap visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional, Tuple
import os


def plot_loss(train_loss: List[float], test_loss: List[float], 
              save: bool = True, save_path: str = "outputs/plots/training_loss.png",
              title: str = "Training and Test Loss") -> None:
    """
    Plot training and test loss curves.
    
    Args:
        train_loss: List of training loss values
        test_loss: List of test loss values
        save: Whether to save the plot
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, test_loss, 'r-', label='Test Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add final values as text
    plt.text(0.02, 0.98, f'Final Train Loss: {train_loss[-1]:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.text(0.02, 0.92, f'Final Test Loss: {test_loss[-1]:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {save_path}")
    
    plt.show()


def compare_solutions(predictions: torch.Tensor, targets: torch.Tensor, 
                     errors: torch.Tensor, time_index: int = 0,
                     save: bool = True, save_path: str = "outputs/plots/solution_comparison.png") -> None:
    """
    Compare predicted solutions with ground truth.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        errors: Error between predictions and targets
        time_index: Time step to visualize
        save: Whether to save the plot
        save_path: Path to save the plot
    """
    # Convert to numpy and get real parts
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    error_np = errors.detach().cpu().numpy()
    
    # Select time step
    pred_t = pred_np[time_index, 0]  # Remove channel dimension
    target_t = target_np[time_index, 0]
    error_t = error_np[time_index, 0]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot predictions
    im1 = axes[0].imshow(pred_t, cmap='hot', aspect='auto')
    axes[0].set_title('Model Predictions', fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot ground truth
    im2 = axes[1].imshow(target_t, cmap='hot', aspect='auto')
    axes[1].set_title('Ground Truth', fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot error
    im3 = axes[2].imshow(error_t, cmap='RdBu_r', aspect='auto')
    axes[2].set_title('Absolute Error', fontweight='bold')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Solution comparison saved to: {save_path}")
    
    plt.show()


def plot_r2_vs_physics_loss_coefficients(physics_loss_coefficients: List[float], 
                                        r2_scores: List[float],
                                        save: bool = True, 
                                        save_path: str = "outputs/plots/r2_vs_physics_loss.png") -> None:
    """
    Plot R² scores vs physics loss coefficients.
    
    Args:
        physics_loss_coefficients: List of physics loss coefficients
        r2_scores: Corresponding R² scores
        save: Whether to save the plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(physics_loss_coefficients, r2_scores, 'bo-', linewidth=2, markersize=8)
    
    plt.xlabel('Physics Loss Coefficient', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('R² Score vs Physics Loss Coefficient', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (coeff, r2) in enumerate(zip(physics_loss_coefficients, r2_scores)):
        plt.annotate(f'{r2:.3f}', (coeff, r2), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"R² vs physics loss plot saved to: {save_path}")
    
    plt.show()


def plot_heatmap_evolution(heatmaps: np.ndarray, time_steps: List[int] = None,
                          save: bool = True, save_path: str = "outputs/plots/heatmap_evolution.png") -> None:
    """
    Plot heatmap evolution over time.
    
    Args:
        heatmaps: Array of heatmaps with shape (time_steps, height, width)
        time_steps: List of time step indices to plot
        save: Whether to save the plot
        save_path: Path to save the plot
    """
    if time_steps is None:
        time_steps = [0, len(heatmaps)//4, len(heatmaps)//2, 3*len(heatmaps)//4, len(heatmaps)-1]
    
    n_plots = len(time_steps)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, t in enumerate(time_steps):
        im = axes[i].imshow(heatmaps[t], cmap='hot', aspect='auto')
        axes[i].set_title(f'Time Step {t}', fontweight='bold')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap evolution saved to: {save_path}")
    
    plt.show()


def plot_model_architecture(model_info: dict, save: bool = True, 
                           save_path: str = "outputs/plots/model_architecture.png") -> None:
    """
    Create a visual representation of the model architecture.
    
    Args:
        model_info: Dictionary containing model information
        save: Whether to save the plot
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a simple architecture diagram
    components = ['Input', 'Fourier Transform', 'Neural Operator', 'Inverse Transform', 'Output']
    positions = np.arange(len(components))
    
    # Plot components
    ax.bar(positions, [1]*len(components), color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightblue'])
    
    # Add component labels
    ax.set_xticks(positions)
    ax.set_xticklabels(components, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.2)
    
    # Add model info
    info_text = f"""
    Model Type: {model_info['model_type']}
    Input Size: {model_info['input_size']}x{model_info['input_size']}
    Hidden Dimensions: {model_info['hidden_dims']}
    Total Parameters: {model_info['total_parameters']:,}
    Trainable Parameters: {model_info['trainable_parameters']:,}
    """
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    
    ax.set_title('PINO Model Architecture', fontsize=16, fontweight='bold')
    ax.set_ylabel('Component', fontsize=12)
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model architecture saved to: {save_path}")
    
    plt.show()


def create_training_summary(train_loss: List[float], test_loss: List[float], 
                           r2_score: float, config: dict,
                           save: bool = True, save_path: str = "outputs/plots/training_summary.png") -> None:
    """
    Create a comprehensive training summary plot.
    
    Args:
        train_loss: Training loss history
        test_loss: Test loss history
        r2_score: Final R² score
        config: Training configuration
        save: Whether to save the plot
        save_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Loss curves
    epochs = range(1, len(train_loss) + 1)
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_loss, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² score
    ax2.bar(['R² Score'], [r2_score], color='green', alpha=0.7)
    ax2.set_ylabel('R² Score')
    ax2.set_title(f'Final R² Score: {r2_score:.4f}')
    ax2.set_ylim(0, 1)
    
    # Plot 3: Configuration summary
    config_text = f"""
    Training Configuration:
    - Epochs: {config.get('num_epochs', 'N/A')}
    - Learning Rate: {config.get('learning_rate', 'N/A')}
    - Physics Loss Coeff: {config.get('physics_loss_coefficient', 'N/A')}
    - Batch Size: {config.get('batch_size', 'N/A')}
    - Optimizer: {config.get('optimizer', 'N/A')}
    """
    ax3.text(0.1, 0.5, config_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax3.set_title('Configuration Summary')
    ax3.axis('off')
    
    # Plot 4: Training statistics
    stats_text = f"""
    Training Statistics:
    - Final Train Loss: {train_loss[-1]:.4f}
    - Final Test Loss: {test_loss[-1]:.4f}
    - Best Test Loss: {min(test_loss):.4f}
    - Loss Reduction: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.1f}%
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax4.set_title('Training Statistics')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training summary saved to: {save_path}")
    
    plt.show()
