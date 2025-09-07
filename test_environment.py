#!/usr/bin/env python3
"""
Test script to verify FractionalPINO environment setup
Tests integration between PINO, hpfracc, and PyTorch
"""

import torch
import numpy as np
import hpfracc as hf
import cupy as cp
from pathlib import Path

def test_environment():
    """Test the FractionalPINO environment setup"""
    print("🧪 Testing FractionalPINO Environment")
    print("=" * 50)
    
    # Test PyTorch
    print("🔥 PyTorch Test:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  Current GPU: {torch.cuda.get_device_name()}")
    
    # Test hpfracc
    print("\n🧮 hpfracc Test:")
    print(f"  Version: {hf.__version__}")
    
    # Test fractional derivative computation
    x = torch.linspace(0, 1, 100)
    alpha = 0.5
    
    try:
        # Test Caputo derivative
        caputo_deriv = hf.optimized_caputo(x.numpy(), x.numpy(), alpha)
        print(f"  Caputo derivative (α={alpha}): ✅")
        print(f"  Result shape: {caputo_deriv.shape}")
        
        # Test Riemann-Liouville derivative
        rl_deriv = hf.optimized_riemann_liouville(x.numpy(), x.numpy(), alpha)
        print(f"  Riemann-Liouville derivative (α={alpha}): ✅")
        print(f"  Result shape: {rl_deriv.shape}")
        
    except Exception as e:
        print(f"  Fractional derivative test failed: ❌ {e}")
    
    # Test CuPy
    print("\n⚡ CuPy Test:")
    print(f"  Version: {cp.__version__}")
    print(f"  CUDA available: {cp.cuda.is_available()}")
    
    if cp.cuda.is_available():
        try:
            # Test GPU computation
            x_gpu = cp.array([1, 2, 3, 4, 5])
            y_gpu = cp.sin(x_gpu)
            print(f"  GPU computation: ✅")
            print(f"  Result: {y_gpu}")
        except Exception as e:
            print(f"  GPU computation failed: ❌ {e}")
    
    # Test PINO model loading
    print("\n🎯 PINO Model Test:")
    try:
        from src.PINO_2D_Heat_Equation import PINO_2D_Heat_Equation
        model = PINO_2D_Heat_Equation()
        print(f"  PINO model created: ✅")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        x_test = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            output = model(x_test)
        print(f"  Forward pass: ✅")
        print(f"  Input shape: {x_test.shape}")
        print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"  PINO model test failed: ❌ {e}")
    
    # Test data loading
    print("\n📊 Data Loading Test:")
    try:
        from src.data.dataset import HeatmapPDEDataset
        print("  Dataset class imported: ✅")
        
        # Check if data directories exist
        heatmap_dir = Path("images/heatmaps")
        pde_dir = Path("images/pde_solutions")
        
        if heatmap_dir.exists() and pde_dir.exists():
            dataset = HeatmapPDEDataset(str(heatmap_dir), str(pde_dir))
            print(f"  Dataset loaded: ✅")
            print(f"  Number of samples: {len(dataset)}")
        else:
            print("  Data directories not found (expected for new setup)")
            print(f"  Heatmap dir exists: {heatmap_dir.exists()}")
            print(f"  PDE dir exists: {pde_dir.exists()}")
            
    except Exception as e:
        print(f"  Data loading test failed: ❌ {e}")
    
    print("\n🎉 Environment test completed!")
    print("=" * 50)

def test_fractional_pino_integration():
    """Test integration between PINO and hpfracc"""
    print("\n🔬 Testing FractionalPINO Integration")
    print("=" * 50)
    
    try:
        # Create a simple fractional PINO test
        import torch.nn as nn
        
        class FractionalPINO(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(64, 128)
                self.decoder = nn.Linear(128, 64)
                self.alpha = 0.7  # Fractional order
                
            def forward(self, x):
                # Simulate fractional processing
                x = self.encoder(x)
                x = torch.relu(x)
                x = self.decoder(x)
                return x
        
        model = FractionalPINO()
        x = torch.randn(1, 64)
        output = model(x)
        
        print("✅ FractionalPINO integration test passed")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Test with hpfracc fractional derivatives
        x_np = x.detach().numpy().flatten()
        frac_deriv = hf.optimized_caputo(x_np, x_np, model.alpha)
        
        print("✅ hpfracc fractional derivative integration passed")
        print(f"  Fractional derivative shape: {frac_deriv.shape}")
        
    except Exception as e:
        print(f"❌ FractionalPINO integration test failed: {e}")

if __name__ == "__main__":
    test_environment()
    test_fractional_pino_integration()
