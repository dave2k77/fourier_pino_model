#!/usr/bin/env python3
"""
Test Experimental Setup for FractionalPINO
Verify that all components are working correctly
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        from src.models.working_fractional_pino import WorkingFractionalPINO
        print("✅ WorkingFractionalPINO import successful")
    except ImportError as e:
        print(f"❌ WorkingFractionalPINO import failed: {e}")
        return False
    
    try:
        from src.models.working_fractional_pino import MultiMethodWorkingFractionalPINO
        print("✅ MultiMethodWorkingFractionalPINO import successful")
    except ImportError as e:
        print(f"❌ MultiMethodWorkingFractionalPINO import failed: {e}")
        return False
    
    try:
        from src.data.fractional_data_generator import generate_fractional_heat_data
        print("✅ Data generation import successful")
    except ImportError as e:
        print(f"❌ Data generation import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that models can be created"""
    print("\nTesting model creation...")
    
    try:
        from src.models.working_fractional_pino import WorkingFractionalPINO
        
        model = WorkingFractionalPINO(
            modes=12,
            width=64,
            alpha=0.5,
            fractional_method='caputo'
        )
        print("✅ WorkingFractionalPINO creation successful")
        
        # Test forward pass
        x = torch.randn(2, 1, 32, 32)
        y = model(x)
        print(f"✅ Forward pass successful: {x.shape} -> {y.shape}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test that data can be generated"""
    print("\nTesting data generation...")
    
    try:
        from src.data.fractional_data_generator import generate_fractional_heat_data
        
        train_data, test_data = generate_fractional_heat_data(alpha=0.5)
        print("✅ Data generation successful")
        print(f"   Train data: {train_data[0].shape}, {train_data[1].shape}")
        print(f"   Test data: {test_data[0].shape}, {test_data[1].shape}")
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False
    
    return True

def test_hpfracc_integration():
    """Test HPFRACC integration"""
    print("\nTesting HPFRACC integration...")
    
    try:
        import hpfracc as hf
        print("✅ HPFRACC import successful")
        
        # Test basic fractional operation
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hf.optimized_caputo(x, x, 0.5)
        print(f"✅ HPFRACC operation successful: {result.shape}")
        
    except Exception as e:
        print(f"❌ HPFRACC integration failed: {e}")
        return False
    
    return True

def test_device_setup():
    """Test device setup"""
    print("\nTesting device setup...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    return True

def main():
    """Main test function"""
    print("FractionalPINO Experimental Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_data_generation,
        test_hpfracc_integration,
        test_device_setup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Experimental setup is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
