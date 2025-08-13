#!/usr/bin/env python3
"""
Main training script for PINO Model

This script demonstrates how to train the PINO model using the modular
architecture and configuration system.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import PINO_2D_Heat_Equation
from src.data import HeatmapPDEDataset, split_data
from src.utils import train, loss_function
from config import PINOConfig, DEFAULT_CONFIG


def create_optimizer(model, config):
    """Create optimizer based on configuration."""
    if config.training.optimizer == "SGD":
        return optim.SGD(model.parameters(), lr=config.training.learning_rate)
    elif config.training.optimizer == "Adam":
        return optim.Adam(model.parameters(), lr=config.training.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PINO Model")
    parser.add_argument("--config", type=str, default="default", 
                       help="Configuration to use (default, experiment_a, experiment_b)")
    parser.add_argument("--heatmap_folder", type=str, 
                       help="Path to heatmap folder")
    parser.add_argument("--pde_folder", type=str, 
                       help="Path to PDE solution folder")
    parser.add_argument("--epochs", type=int, 
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, 
                       help="Learning rate")
    parser.add_argument("--physics_coeff", type=float, 
                       help="Physics loss coefficient")
    parser.add_argument("--output_dir", type=str, 
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == "default":
        config = DEFAULT_CONFIG
    elif args.config == "experiment_a":
        config = PINOConfig(
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=0.005,
                physics_loss_coefficient=0.01,
                optimizer="SGD"
            )
        )
    elif args.config == "experiment_b":
        config = PINOConfig(
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=0.005,
                physics_loss_coefficient=0.01,
                optimizer="Adam"
            )
        )
    else:
        raise ValueError(f"Unknown config: {args.config}")
    
    # Override config with command line arguments
    if args.heatmap_folder:
        config.data.heatmap_folder = args.heatmap_folder
    if args.pde_folder:
        config.data.pde_solution_folder = args.pde_folder
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.physics_coeff:
        config.training.physics_loss_coefficient = args.physics_coeff
    if args.output_dir:
        config.data.output_dir = args.output_dir
    
    # Validate and create directories
    try:
        config.validate_paths()
        config.create_output_dirs()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Set device
    if config.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.training.device)
    
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    
    # Initialize model
    model = PINO_2D_Heat_Equation(
        input_size=config.model.input_size,
        hidden_dims=config.model.hidden_dims
    )
    model.to(device)
    
    # Print model information
    model_info = model.get_model_info()
    print(f"Model: {model_info}")
    
    # Load dataset
    try:
        dataset = HeatmapPDEDataset(
            heatmap_folder=config.data.heatmap_folder,
            pde_solution_folder=config.data.pde_solution_folder,
            transform_size=config.data.transform_size
        )
        print(f"Dataset: {dataset.get_dataset_info()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Split dataset
    train_dataset, test_dataset = split_data(
        dataset, 
        train_ratio=config.training.train_ratio,
        random_seed=config.training.random_seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.training.batch_size,
        shuffle=False
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Train model
    print("Starting training...")
    train_loss_history, test_loss_history, final_r2 = train(
        model=model,
        loss_fn=loss_function,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config.training.num_epochs,
        physics_loss_coefficient=config.training.physics_loss_coefficient,
        device=device,
        verbose=args.verbose
    )
    
    # Save model
    model_path = os.path.join(config.data.output_dir, "models", "pino_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_loss_history': train_loss_history,
        'test_loss_history': test_loss_history,
        'final_r2': final_r2
    }, model_path)
    
    print(f"Training completed!")
    print(f"Final R² score: {final_r2:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
