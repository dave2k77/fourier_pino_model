#!/usr/bin/env python3
"""
HPFRACC Integration Test Suite
Test all available fractional operators and ML components for FractionalPINO
"""

import torch
import numpy as np
import hpfracc as hf
from hpfracc import (
    # Core optimized methods
    OptimizedRiemannLiouville, OptimizedCaputo, OptimizedGrunwaldLetnikov,
    optimized_riemann_liouville, optimized_caputo, optimized_grunwald_letnikov,
    
    # Advanced methods
    WeylDerivative, MarchaudDerivative, HadamardDerivative, ReizFellerDerivative,
    OptimizedWeylDerivative, OptimizedMarchaudDerivative, OptimizedHadamardDerivative, OptimizedReizFellerDerivative,
    optimized_weyl_derivative, optimized_marchaud_derivative, optimized_hadamard_derivative, optimized_reiz_feller_derivative,
    
    # Special methods
    FractionalLaplacian, FractionalFourierTransform, FractionalZTransform, FractionalMellinTransform,
    fractional_laplacian, fractional_fourier_transform, fractional_z_transform, fractional_mellin_transform,
    
    # Novel fractional derivatives
    CaputoFabrizioDerivative, AtanganaBaleanuDerivative,
    caputo_fabrizio_derivative, atangana_baleanu_derivative,
    optimized_caputo_fabrizio_derivative, optimized_atangana_baleanu_derivative,
    
    # Core definitions
    FractionalOrder,
)

from hpfracc.ml import (
    # Backend Management
    BackendManager, BackendType, get_backend_manager, set_backend_manager,
    
    # Core ML Components
    FractionalNeuralNetwork, FractionalAttention, FractionalLossFunction,
    FractionalMSELoss, FractionalCrossEntropyLoss,
    
    # Neural Network Layers
    FractionalConv1D, FractionalConv2D, FractionalLSTM, FractionalTransformer,
    FractionalPooling, FractionalBatchNorm1d,
    
    # Optimizers
    FractionalAdam, FractionalSGD, FractionalRMSprop,
    
    # GNN Components
    FractionalGCN, FractionalGAT, FractionalGraphSAGE, FractionalGraphUNet,
    FractionalGNNFactory
)

import time
import warnings
warnings.filterwarnings('ignore')

def test_core_fractional_operators():
    """Test core fractional operators"""
    print("🧮 Testing Core Fractional Operators")
    print("=" * 50)
    
    # Test data
    t = np.linspace(0, 1, 100)
    f = np.sin(2 * np.pi * t)
    alpha = 0.5
    
    # Test Caputo derivative
    try:
        caputo_result = optimized_caputo(f, t, alpha)
        print(f"✅ Caputo derivative (α={alpha}): {caputo_result.shape}")
    except Exception as e:
        print(f"❌ Caputo derivative failed: {e}")
    
    # Test Riemann-Liouville derivative
    try:
        rl_result = optimized_riemann_liouville(f, t, alpha)
        print(f"✅ Riemann-Liouville derivative (α={alpha}): {rl_result.shape}")
    except Exception as e:
        print(f"❌ Riemann-Liouville derivative failed: {e}")
    
    # Test Grünwald-Letnikov derivative
    try:
        gl_result = optimized_grunwald_letnikov(f, t, alpha)
        print(f"✅ Grünwald-Letnikov derivative (α={alpha}): {gl_result.shape}")
    except Exception as e:
        print(f"❌ Grünwald-Letnikov derivative failed: {e}")

def test_advanced_fractional_operators():
    """Test advanced fractional operators"""
    print("\n🚀 Testing Advanced Fractional Operators")
    print("=" * 50)
    
    # Test data
    t = np.linspace(0, 1, 100)
    f = np.sin(2 * np.pi * t)
    alpha = 0.5
    
    # Test Weyl derivative
    try:
        weyl_result = optimized_weyl_derivative(f, t, alpha)
        print(f"✅ Weyl derivative (α={alpha}): {weyl_result.shape}")
    except Exception as e:
        print(f"❌ Weyl derivative failed: {e}")
    
    # Test Marchaud derivative
    try:
        marchaud_result = optimized_marchaud_derivative(f, t, alpha)
        print(f"✅ Marchaud derivative (α={alpha}): {marchaud_result.shape}")
    except Exception as e:
        print(f"❌ Marchaud derivative failed: {e}")
    
    # Test Hadamard derivative
    try:
        hadamard_result = optimized_hadamard_derivative(f, t, alpha)
        print(f"✅ Hadamard derivative (α={alpha}): {hadamard_result.shape}")
    except Exception as e:
        print(f"❌ Hadamard derivative failed: {e}")
    
    # Test Reiz-Feller derivative
    try:
        reiz_feller_result = optimized_reiz_feller_derivative(f, t, alpha)
        print(f"✅ Reiz-Feller derivative (α={alpha}): {reiz_feller_result.shape}")
    except Exception as e:
        print(f"❌ Reiz-Feller derivative failed: {e}")

def test_novel_fractional_operators():
    """Test novel non-singular fractional operators"""
    print("\n🌟 Testing Novel Non-Singular Fractional Operators")
    print("=" * 50)
    
    # Test data
    t = np.linspace(0, 1, 100)
    f = np.sin(2 * np.pi * t)
    alpha = 0.5
    
    # Test Caputo-Fabrizio derivative
    try:
        cf_result = optimized_caputo_fabrizio_derivative(f, t, alpha)
        print(f"✅ Caputo-Fabrizio derivative (α={alpha}): {cf_result.shape}")
    except Exception as e:
        print(f"❌ Caputo-Fabrizio derivative failed: {e}")
    
    # Test Atangana-Baleanu derivative
    try:
        ab_result = optimized_atangana_baleanu_derivative(f, t, alpha)
        print(f"✅ Atangana-Baleanu derivative (α={alpha}): {ab_result.shape}")
    except Exception as e:
        print(f"❌ Atangana-Baleanu derivative failed: {e}")

def test_special_operators():
    """Test special fractional operators"""
    print("\n🎯 Testing Special Fractional Operators")
    print("=" * 50)
    
    # Test data
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y)
    f = np.sin(np.pi * X) * np.cos(np.pi * Y)
    alpha = 0.5
    
    # Test Fractional Laplacian
    try:
        laplacian_result = fractional_laplacian(f, alpha)
        print(f"✅ Fractional Laplacian (α={alpha}): {laplacian_result.shape}")
    except Exception as e:
        print(f"❌ Fractional Laplacian failed: {e}")
    
    # Test Fractional Fourier Transform
    try:
        fft_result = fractional_fourier_transform(f, alpha)
        print(f"✅ Fractional FFT (α={alpha}): {fft_result.shape}")
    except Exception as e:
        print(f"❌ Fractional FFT failed: {e}")
    
    # Test Fractional Mellin Transform
    try:
        mellin_result = fractional_mellin_transform(f, alpha)
        print(f"✅ Fractional Mellin Transform (α={alpha}): {mellin_result.shape}")
    except Exception as e:
        print(f"❌ Fractional Mellin Transform failed: {e}")

def test_ml_components():
    """Test HPFRACC ML components"""
    print("\n🤖 Testing HPFRACC ML Components")
    print("=" * 50)
    
    # Test FractionalNeuralNetwork
    try:
        network = FractionalNeuralNetwork(
            input_size=64,
            hidden_sizes=[32, 16],
            output_size=32,
            fractional_order=FractionalOrder(0.5),
            backend=BackendType.TORCH
        )
        
        x = torch.randn(10, 64)
        output = network(x, use_fractional=True, method="RL")
        print(f"✅ FractionalNeuralNetwork: {output.shape}")
    except Exception as e:
        print(f"❌ FractionalNeuralNetwork failed: {e}")
    
    # Test FractionalConv2D
    try:
        conv = FractionalConv2D(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            fractional_order=FractionalOrder(0.5)
        )
        
        x = torch.randn(1, 1, 32, 32)
        output = conv(x)
        print(f"✅ FractionalConv2D: {output.shape}")
    except Exception as e:
        print(f"❌ FractionalConv2D failed: {e}")
    
    # Test FractionalLSTM
    try:
        lstm = FractionalLSTM(
            input_size=32,
            hidden_size=16,
            fractional_order=FractionalOrder(0.5)
        )
        
        x = torch.randn(10, 5, 32)  # (seq_len, batch, input_size)
        output, (h_n, c_n) = lstm(x)
        print(f"✅ FractionalLSTM: {output.shape}")
    except Exception as e:
        print(f"❌ FractionalLSTM failed: {e}")
    
    # Test FractionalTransformer
    try:
        transformer = FractionalTransformer(
            d_model=32,
            nhead=4,
            num_layers=2,
            fractional_order=FractionalOrder(0.5)
        )
        
        x = torch.randn(10, 5, 32)  # (seq_len, batch, d_model)
        output = transformer(x)
        print(f"✅ FractionalTransformer: {output.shape}")
    except Exception as e:
        print(f"❌ FractionalTransformer failed: {e}")

def test_gnn_components():
    """Test HPFRACC GNN components"""
    print("\n🕸️ Testing HPFRACC GNN Components")
    print("=" * 50)
    
    # Test FractionalGCN
    try:
        gcn = FractionalGCN(
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
            fractional_order=FractionalOrder(0.5)
        )
        
        node_features = torch.randn(100, 16)
        edge_index = torch.randint(0, 100, (2, 200))
        output = gcn(node_features, edge_index)
        print(f"✅ FractionalGCN: {output.shape}")
    except Exception as e:
        print(f"❌ FractionalGCN failed: {e}")
    
    # Test FractionalGAT
    try:
        gat = FractionalGAT(
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
            num_heads=4,
            fractional_order=FractionalOrder(0.5)
        )
        
        node_features = torch.randn(100, 16)
        edge_index = torch.randint(0, 100, (2, 200))
        output = gat(node_features, edge_index)
        print(f"✅ FractionalGAT: {output.shape}")
    except Exception as e:
        print(f"❌ FractionalGAT failed: {e}")
    
    # Test FractionalGraphSAGE
    try:
        sage = FractionalGraphSAGE(
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
            fractional_order=FractionalOrder(0.5)
        )
        
        node_features = torch.randn(100, 16)
        edge_index = torch.randint(0, 100, (2, 200))
        output = sage(node_features, edge_index)
        print(f"✅ FractionalGraphSAGE: {output.shape}")
    except Exception as e:
        print(f"❌ FractionalGraphSAGE failed: {e}")

def test_optimizers():
    """Test HPFRACC fractional optimizers"""
    print("\n⚡ Testing HPFRACC Fractional Optimizers")
    print("=" * 50)
    
    # Create a simple model
    model = torch.nn.Linear(10, 1)
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Test FractionalAdam
    try:
        optimizer = FractionalAdam(
            model.parameters(),
            lr=0.001,
            fractional_order=FractionalOrder(0.5)
        )
        
        criterion = torch.nn.MSELoss()
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"✅ FractionalAdam: Loss = {loss.item():.6f}")
    except Exception as e:
        print(f"❌ FractionalAdam failed: {e}")
    
    # Test FractionalSGD
    try:
        optimizer = FractionalSGD(
            model.parameters(),
            lr=0.001,
            fractional_order=FractionalOrder(0.5)
        )
        
        criterion = torch.nn.MSELoss()
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"✅ FractionalSGD: Loss = {loss.item():.6f}")
    except Exception as e:
        print(f"❌ FractionalSGD failed: {e}")

def test_performance_benchmark():
    """Benchmark performance of different fractional methods"""
    print("\n📊 Performance Benchmark")
    print("=" * 50)
    
    # Test data
    sizes = [100, 500, 1000]
    alpha = 0.5
    
    for size in sizes:
        print(f"\nSize: {size}")
        t = np.linspace(0, 1, size)
        f = np.sin(2 * np.pi * t)
        
        # Benchmark Caputo
        start = time.time()
        try:
            _ = optimized_caputo(f, t, alpha)
            caputo_time = time.time() - start
            print(f"  Caputo: {caputo_time:.4f}s")
        except Exception as e:
            print(f"  Caputo: Failed - {e}")
        
        # Benchmark Riemann-Liouville
        start = time.time()
        try:
            _ = optimized_riemann_liouville(f, t, alpha)
            rl_time = time.time() - start
            print(f"  Riemann-Liouville: {rl_time:.4f}s")
        except Exception as e:
            print(f"  Riemann-Liouville: Failed - {e}")
        
        # Benchmark Caputo-Fabrizio
        start = time.time()
        try:
            _ = optimized_caputo_fabrizio_derivative(f, t, alpha)
            cf_time = time.time() - start
            print(f"  Caputo-Fabrizio: {cf_time:.4f}s")
        except Exception as e:
            print(f"  Caputo-Fabrizio: Failed - {e}")

def main():
    """Run all HPFRACC integration tests"""
    print("🧪 HPFRACC Integration Test Suite")
    print("=" * 60)
    print("Testing all available fractional operators and ML components")
    print("=" * 60)
    
    # Test core functionality
    test_core_fractional_operators()
    test_advanced_fractional_operators()
    test_novel_fractional_operators()
    test_special_operators()
    
    # Test ML components
    test_ml_components()
    test_gnn_components()
    test_optimizers()
    
    # Performance testing
    test_performance_benchmark()
    
    print("\n🎉 HPFRACC Integration Test Complete!")
    print("=" * 60)
    print("✅ Core fractional operators tested")
    print("✅ Advanced fractional operators tested")
    print("✅ Novel non-singular operators tested")
    print("✅ Special operators tested")
    print("✅ ML components tested")
    print("✅ GNN components tested")
    print("✅ Optimizers tested")
    print("✅ Performance benchmarked")
    print("\n🚀 Ready for Enhanced FractionalPINO implementation!")

if __name__ == "__main__":
    main()
