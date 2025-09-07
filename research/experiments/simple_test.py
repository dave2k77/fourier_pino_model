#!/usr/bin/env python3
"""
Simple Test Script for FractionalPINO
Test basic functionality before running full experiments
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.working_fractional_pino import WorkingFractionalPINO
from src.data.fractional_data_generator_fixed import generate_fractional_heat_data

def test_basic_functionality():
    """Test basic FractionalPINO functionality"""
    print("Testing basic FractionalPINO functionality...")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate simple test data
    print("Generating test data...")
    train_data, test_data = generate_fractional_heat_data(
        alpha=0.5,
        domain_size=16,  # Smaller for testing
        time_steps=10,
        batch_size=4     # Small batch for testing
    )
    
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    print(f"Train data shapes: {x_train.shape}, {y_train.shape}")
    print(f"Test data shapes: {x_test.shape}, {y_test.shape}")
    
    # Create model
    print("Creating model...")
    model = WorkingFractionalPINO(
        modes=8,
        width=32,
        alpha=0.5,
        fractional_method='caputo'
    ).to(device)
    
    # Move data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train)
        print(f"Prediction shape: {y_pred.shape}")
        print(f"Target shape: {y_train.shape}")
        
        # Compute basic metrics
        mse = torch.nn.MSELoss()(y_pred, y_train)
        l2_error = torch.norm(y_pred - y_train) / torch.norm(y_train)
        
        print(f"MSE: {mse.item():.6f}")
        print(f"L2 Error: {l2_error.item():.6f}")
    
    print("✅ Basic functionality test passed!")
    return True

def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate test data
    train_data, _ = generate_fractional_heat_data(
        alpha=0.5,
        domain_size=16,
        time_steps=10,
        batch_size=4
    )
    
    x_train, y_train = train_data
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    # Create model
    model = WorkingFractionalPINO(
        modes=8,
        width=32,
        alpha=0.5,
        fractional_method='caputo'
    ).to(device)
    
    # Training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    optimizer.zero_grad()
    
    y_pred = model(x_train)
    loss = torch.nn.MSELoss()(y_pred, y_train)
    
    print(f"Initial loss: {loss.item():.6f}")
    
    loss.backward()
    optimizer.step()
    
    # Test again
    with torch.no_grad():
        y_pred_new = model(x_train)
        loss_new = torch.nn.MSELoss()(y_pred_new, y_train)
        print(f"Loss after step: {loss_new.item():.6f}")
    
    print("✅ Training step test passed!")
    return True

def main():
    """Main test function"""
    print("FractionalPINO Simple Test")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_training_step()
        
        print("\n" + "=" * 40)
        print("✅ All tests passed! Ready for experiments.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
