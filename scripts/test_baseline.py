#!/usr/bin/env python3
"""
Quick Test Script for Baseline Reproduction

This script runs a quick test to verify the baseline reproduction works
before running the full suite of experiments.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.reproduce_baseline import BaselineReproducer


def test_baseline_reproduction():
    """Test baseline reproduction with a quick experiment."""
    print("🧪 Testing Baseline Reproduction")
    print("=" * 50)
    
    # Initialize reproducer with test output directory
    reproducer = BaselineReproducer(output_dir="test_baseline_results")
    
    # Set deterministic training
    reproducer.set_deterministic_training(seed=42)
    
    print("✅ Deterministic training configured")
    
    # Test with a quick experiment (fewer epochs for testing)
    try:
        print("\n🚀 Running quick test experiment...")
        results = reproducer.run_experiment(
            "test_experiment",
            num_epochs=5,  # Quick test with fewer epochs
            batch_size=16   # Smaller batch size for testing
        )
        
        print(f"✅ Test experiment completed successfully!")
        print(f"📊 Final R² Score: {results['training_results']['final_r2_score']:.4f}")
        print(f"⏱️  Training Time: {results['training_results']['training_time']:.2f} seconds")
        
        # Verify results structure
        required_keys = [
            'experiment_params', 'training_results', 'loss_history',
            'model_path', 'plot_path'
        ]
        
        for key in required_keys:
            if key in results:
                print(f"✅ {key}: ✓")
            else:
                print(f"❌ {key}: Missing")
                return False
        
        print("\n🎉 All tests passed! Baseline reproduction is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("\n📊 Testing Data Loading")
    print("-" * 30)
    
    try:
        from config import PINOConfig, ModelConfig, TrainingConfig, DataConfig
        
        # Create test configuration
        config = PINOConfig(
            model=ModelConfig(input_size=64, hidden_dims=[128, 256, 128]),
            training=TrainingConfig(
                num_epochs=5, batch_size=16, learning_rate=0.005,
                physics_loss_coefficient=0.01, train_ratio=0.8,
                random_seed=42, device="auto"
            ),
            data=DataConfig(
                heatmap_folder="images/heatmaps",
                pde_solution_folder="images/pde_solutions",
                transform_size=(64, 64),
                output_dir="test_output"
            )
        )
        
        # Test dataset loading
        reproducer = BaselineReproducer(output_dir="test_data_results")
        train_dataset, test_dataset = reproducer.load_dataset(config)
        
        print(f"✅ Dataset loaded successfully")
        print(f"📈 Train samples: {len(train_dataset)}")
        print(f"📉 Test samples: {len(test_dataset)}")
        
        # Test data loader creation
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        print(f"✅ Data loaders created successfully")
        print(f"🔄 Train batches: {len(train_loader)}")
        print(f"🔄 Test batches: {len(test_loader)}")
        
        # Test single batch
        for heatmaps, pde_solutions in train_loader:
            print(f"✅ Batch shape - Heatmaps: {heatmaps.shape}, PDE Solutions: {pde_solutions.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation and forward pass."""
    print("\n🤖 Testing Model Creation")
    print("-" * 30)
    
    try:
        from src.models import PINO_2D_Heat_Equation
        
        # Create model
        model = PINO_2D_Heat_Equation(
            input_size=64,
            hidden_dims=[128, 256, 128]
        )
        
        print(f"✅ Model created successfully")
        
        # Get model info
        model_info = model.get_model_info()
        print(f"📊 Model parameters: {model_info['total_parameters']:,}")
        print(f"📊 Trainable parameters: {model_info['trainable_parameters']:,}")
        
        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Create dummy input (batch_size, channels, height, width)
        dummy_input = torch.randn(2, 1, 64, 64).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"📊 Input shape: {dummy_input.shape}")
        print(f"📊 Output shape: {output.shape}")
        print(f"📊 Output type: {output.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 PINO Baseline Reproduction Test Suite")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Baseline Reproduction", test_baseline_reproduction)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready to run full baseline reproduction.")
        print("\nNext steps:")
        print("1. Run: python scripts/reproduce_baseline.py")
        print("2. Check results in baseline_results/ directory")
        print("3. Review BASELINE_REPORT.md for detailed analysis")
    else:
        print("❌ SOME TESTS FAILED! Please fix issues before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
