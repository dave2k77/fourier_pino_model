#!/usr/bin/env python3
"""
Basic Usage Example for PINO Model

This script demonstrates the basic usage of the PINO model for solving
the 2D heat equation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.models import PINO_2D_Heat_Equation
from src.data import HeatmapPDEDataset, split_data
from src.utils import train, loss_function, calculate_r2_score


def main():
    """Demonstrate basic PINO usage."""
    print("🚀 Fourier PINO Model - Basic Usage Example")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("\n📦 Initializing PINO model...")
    model = PINO_2D_Heat_Equation(
        input_size=64,
        hidden_dims=[128, 256, 128]
    )
    model.to(device)
    
    # Print model information
    model_info = model.get_model_info()
    print(f"Model parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Load dataset (assuming data exists)
    print("\n📊 Loading dataset...")
    try:
        dataset = HeatmapPDEDataset(
            heatmap_folder="images/heatmaps",
            pde_solution_folder="images/pde_solutions",
            transform_size=(64, 64)
        )
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # Split dataset
        train_dataset, test_dataset = split_data(dataset, train_ratio=0.8)
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
    except FileNotFoundError as e:
        print(f"⚠️  Data not found: {e}")
        print("Please ensure you have the required data files in images/heatmaps and images/pde_solutions")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Setup training
    print("\n⚙️  Setting up training...")
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Train model (short training for demonstration)
    print("\n🎯 Starting training...")
    print("Note: This is a demonstration with limited epochs")
    
    train_loss_history, test_loss_history, final_r2 = train(
        model=model,
        loss_fn=loss_function,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=10,  # Short training for demo
        physics_loss_coefficient=0.01,
        device=device,
        verbose=True
    )
    
    print(f"\n✅ Training completed!")
    print(f"Final R² score: {final_r2:.4f}")
    
    # Demonstrate inference
    print("\n🔮 Demonstrating inference...")
    model.eval()
    with torch.no_grad():
        # Get a sample from test set
        sample_heatmap, sample_solution = test_dataset[0]
        sample_heatmap = sample_heatmap.unsqueeze(0).to(device)  # Add batch dimension
        
        # Make prediction
        prediction = model(sample_heatmap)
        prediction_real = prediction.real.squeeze(0).cpu()
        
        # Calculate R² for this sample
        sample_r2 = calculate_r2_score(prediction_real, sample_solution)
        print(f"Sample R² score: {sample_r2:.4f}")
    
    # Plot training history
    print("\n📈 Plotting training history...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.plot(test_loss_history, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training history saved to outputs/plots/training_history.png")
    
    # Save model
    print("\n💾 Saving model...")
    os.makedirs('outputs/models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': train_loss_history,
        'test_loss_history': test_loss_history,
        'final_r2': final_r2
    }, 'outputs/models/pino_model_example.pth')
    print("Model saved to outputs/models/pino_model_example.pth")
    
    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("1. Check outputs/plots/training_history.png for training curves")
    print("2. Load the saved model for inference")
    print("3. Experiment with different hyperparameters")
    print("4. Try different physics loss coefficients")


if __name__ == "__main__":
    main()
