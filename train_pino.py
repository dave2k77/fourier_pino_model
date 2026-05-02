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
import json
from copy import deepcopy

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import PINO_2D_Heat_Equation
from src.data import HeatmapPDEDataset, split_data
from src.utils import train, loss_function
from config import DEFAULT_CONFIG


def create_optimizer(model, config):
    """Create optimizer based on configuration."""
    if config.training.optimizer == "SGD":
        return optim.SGD(model.parameters(), lr=config.training.learning_rate)
    elif config.training.optimizer == "Adam":
        return optim.Adam(model.parameters(), lr=config.training.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def create_data_loaders(config):
    """Create train/test data loaders from the configured heat-equation dataset."""
    dataset = HeatmapPDEDataset(
        heatmap_folder=config.data.heatmap_folder,
        pde_solution_folder=config.data.pde_solution_folder,
        transform_size=config.data.transform_size
    )
    train_dataset, test_dataset = split_data(
        dataset,
        train_ratio=config.training.train_ratio,
        random_seed=config.training.random_seed
    )
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
    return dataset, train_loader, test_loader


def run_training(config, train_loader, test_loader, device, verbose=False):
    """Train one PINO configuration and return the model, optimizer, and history."""
    model = PINO_2D_Heat_Equation(
        input_size=config.model.input_size,
        hidden_dims=config.model.hidden_dims
    )
    model.to(device)
    optimizer = create_optimizer(model, config)
    history = train(
        model=model,
        loss_fn=loss_function,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config.training.num_epochs,
        physics_loss_coefficient=config.training.physics_loss_coefficient,
        device=device,
        verbose=verbose,
        return_history=True
    )
    return model, optimizer, history


def save_training_state(config, model, optimizer, history, suffix="pino_model"):
    """Save a trained model checkpoint and metrics history."""
    model_path = os.path.join(config.data.output_dir, "models", f"{suffix}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "history": history,
    }, model_path)
    return model_path


def save_experiment_summary(config, rows):
    """Write experiment results to JSON and CSV files."""
    results_dir = os.path.join(config.data.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "physics_loss_sweep.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    csv_path = os.path.join(results_dir, "physics_loss_sweep.csv")
    headers = [
        "optimizer",
        "physics_loss_coefficient",
        "final_train_loss",
        "final_test_loss",
        "final_train_data_loss",
        "final_test_data_loss",
        "final_train_physics_loss",
        "final_test_physics_loss",
        "final_r2",
    ]
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(headers) + "\n")
        for row in rows:
            fh.write(",".join(str(row[key]) for key in headers) + "\n")

    return json_path, csv_path


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PINO Model")
    parser.add_argument("--mode", choices=["single", "sweep"], default="single",
                       help="Run one training job or the canonical optimizer/physics-loss sweep")
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
    parser.add_argument("--optimizer", choices=["SGD", "Adam"],
                       help="Optimizer for single-run mode")
    parser.add_argument("--output_dir", type=str, 
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    config = deepcopy(DEFAULT_CONFIG)
    
    # Override config with command line arguments
    if args.heatmap_folder:
        config.data.heatmap_folder = args.heatmap_folder
    if args.pde_folder:
        config.data.pde_solution_folder = args.pde_folder
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.physics_coeff is not None:
        config.training.physics_loss_coefficient = args.physics_coeff
    if args.optimizer:
        config.training.optimizer = args.optimizer
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

    try:
        dataset, train_loader, test_loader = create_data_loaders(config)
        print(f"Dataset: {dataset.get_dataset_info()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if args.mode == "single":
        print("Starting single training run...")
        model, optimizer, history = run_training(
            config, train_loader, test_loader, device, args.verbose
        )
        model_path = save_training_state(config, model, optimizer, history)
        print("Training completed!")
        print(f"Final R² score: {history['final_r2']:.4f}")
        print(f"Model saved to: {model_path}")
        return

    print("Starting canonical physics-loss sweep...")
    rows = []
    for optimizer_name in config.experiment.optimizers:
        for physics_coeff in config.experiment.physics_loss_coefficients:
            run_config = deepcopy(config)
            run_config.training.optimizer = optimizer_name
            run_config.training.physics_loss_coefficient = physics_coeff
            print(f"Running optimizer={optimizer_name}, physics_coeff={physics_coeff}")
            model, optimizer, history = run_training(
                run_config, train_loader, test_loader, device, args.verbose
            )
            suffix = f"pino_{optimizer_name.lower()}_physics_{physics_coeff:g}".replace(".", "p")
            model_path = save_training_state(run_config, model, optimizer, history, suffix=suffix)
            rows.append({
                "optimizer": optimizer_name,
                "physics_loss_coefficient": physics_coeff,
                "final_train_loss": history["train_loss"][-1],
                "final_test_loss": history["test_loss"][-1],
                "final_train_data_loss": history["train_data_loss"][-1],
                "final_test_data_loss": history["test_data_loss"][-1],
                "final_train_physics_loss": history["train_physics_loss"][-1],
                "final_test_physics_loss": history["test_physics_loss"][-1],
                "final_r2": history["final_r2"],
                "model_path": model_path,
            })

    json_path, csv_path = save_experiment_summary(config, rows)
    print("Sweep completed!")
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
