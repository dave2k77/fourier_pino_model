#!/usr/bin/env python3
"""
CPU Test Script for FractionalPINO
Test basic functionality on CPU to avoid device mismatch issues
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

def test_cpu_functionality():
    """Test FractionalPINO functionality on CPU"""
    print("Testing FractionalPINO functionality on CPU...")
    
    # Force CPU usage
    device = torch.device('cpu')
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
    
    # Create model on CPU
    print("Creating model...")
    model = WorkingFractionalPINO(
        modes=8,
        width=32,
        alpha=0.5,
        fractional_method='caputo'
    ).to(device)
    
    # Data is already on CPU
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
    
    print("✅ CPU functionality test passed!")
    return True

def test_training_step_cpu():
    """Test a single training step on CPU"""
    print("\nTesting training step on CPU...")
    
    device = torch.device('cpu')
    
    # Generate test data
    train_data, _ = generate_fractional_heat_data(
        alpha=0.5,
        domain_size=16,
        time_steps=10,
        batch_size=4
    )
    
    x_train, y_train = train_data
    
    # Create model on CPU
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
    
    print("✅ CPU training step test passed!")
    return True

def test_multiple_epochs():
    """Test multiple training epochs"""
    print("\nTesting multiple training epochs...")
    
    device = torch.device('cpu')
    
    # Generate test data
    train_data, _ = generate_fractional_heat_data(
        alpha=0.5,
        domain_size=16,
        time_steps=10,
        batch_size=4
    )
    
    x_train, y_train = train_data
    
    # Create model on CPU
    model = WorkingFractionalPINO(
        modes=8,
        width=32,
        alpha=0.5,
        fractional_method='caputo'
    ).to(device)
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training for 10 epochs...")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        
        y_pred = model(x_train)
        loss = torch.nn.MSELoss()(y_pred, y_train)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_final = model(x_train)
        final_loss = torch.nn.MSELoss()(y_pred_final, y_train)
        l2_error = torch.norm(y_pred_final - y_train) / torch.norm(y_train)
        
        print(f"Final loss: {final_loss.item():.6f}")
        print(f"Final L2 error: {l2_error.item():.6f}")
    
    print("✅ Multiple epochs test passed!")
    return True

def main():
    """Main test function"""
    print("FractionalPINO CPU Test")
    print("=" * 40)
    
    try:
        test_cpu_functionality()
        test_training_step_cpu()
        test_multiple_epochs()
        
        print("\n" + "=" * 40)
        print("✅ All CPU tests passed! Basic functionality verified.")
        print("📊 Ready to proceed with experimental execution.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
