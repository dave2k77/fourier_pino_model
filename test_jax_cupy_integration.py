#!/usr/bin/env python3
"""
Test JAX-CuPy integration for GPU-accelerated scientific computing
"""

import jax
import jax.numpy as jnp
import cupy as cp
import numpy as np
import time

def test_jax_cupy_integration():
    """Test JAX working with CuPy for GPU acceleration"""
    print("🚀 JAX-CuPy Integration Test")
    print("=" * 50)
    
    # Test JAX
    print("🔥 JAX Test:")
    print(f"  Version: {jax.__version__}")
    print(f"  Default backend: {jax.default_backend()}")
    print(f"  Available devices: {jax.devices()}")
    
    # Test CuPy
    print("\n⚡ CuPy Test:")
    print(f"  Version: {cp.__version__}")
    print(f"  CUDA available: {cp.cuda.is_available()}")
    print(f"  GPU count: {cp.cuda.runtime.getDeviceCount()}")
    if cp.cuda.is_available():
        print(f"  Current GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    
    # Test JAX-CuPy data transfer
    print("\n🔄 JAX-CuPy Data Transfer Test:")
    try:
        # Create data in JAX
        x_jax = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        print(f"  JAX array: {x_jax}")
        
        # Convert to CuPy
        x_cupy = cp.asarray(x_jax)
        print(f"  CuPy array: {x_cupy}")
        
        # Perform computation in CuPy
        y_cupy = cp.sin(x_cupy) * cp.cos(x_cupy)
        print(f"  CuPy computation: {y_cupy}")
        
        # Convert back to JAX
        y_jax = jnp.array(y_cupy)
        print(f"  Back to JAX: {y_jax}")
        
        print("  ✅ JAX-CuPy data transfer: Success")
        
    except Exception as e:
        print(f"  ❌ JAX-CuPy data transfer failed: {e}")
    
    # Test performance comparison
    print("\n⚡ Performance Comparison:")
    size = 100000
    
    # NumPy (CPU)
    x_np = np.random.random(size)
    start = time.time()
    result_np = np.sin(x_np) * np.cos(x_np)
    time_np = time.time() - start
    
    # JAX (CPU)
    x_jax = jnp.array(x_np)
    start = time.time()
    result_jax = jnp.sin(x_jax) * jnp.cos(x_jax)
    result_jax.block_until_ready()
    time_jax = time.time() - start
    
    # CuPy (GPU)
    x_cupy = cp.array(x_np)
    start = time.time()
    result_cupy = cp.sin(x_cupy) * cp.cos(x_cupy)
    cp.cuda.Stream.null.synchronize()
    time_cupy = time.time() - start
    
    # JAX-CuPy hybrid
    start = time.time()
    x_jax = jnp.array(x_np)
    x_cupy = cp.asarray(x_jax)
    result_cupy = cp.sin(x_cupy) * cp.cos(x_cupy)
    result_jax = jnp.array(result_cupy)
    result_jax.block_until_ready()
    time_hybrid = time.time() - start
    
    print(f"  NumPy (CPU): {time_np:.4f}s")
    print(f"  JAX (CPU): {time_jax:.4f}s")
    print(f"  CuPy (GPU): {time_cupy:.4f}s")
    print(f"  JAX-CuPy hybrid: {time_hybrid:.4f}s")
    
    # Test fractional calculus with JAX-CuPy
    print("\n🧮 Fractional Calculus with JAX-CuPy:")
    try:
        import hpfracc as hf
        
        # Create test data
        x = jnp.linspace(0, 1, 100)
        alpha = 0.7
        
        # Convert to CuPy for fractional computation
        x_cupy = cp.asarray(x)
        
        # Compute fractional derivative using hpfracc
        x_np = np.array(x)
        frac_deriv = hf.optimized_caputo(x_np, x_np, alpha)
        
        # Convert result back to JAX
        frac_deriv_jax = jnp.array(frac_deriv)
        
        print(f"  Fractional derivative (α={alpha}): ✅")
        print(f"  Result shape: {frac_deriv_jax.shape}")
        print(f"  Result range: [{jnp.min(frac_deriv_jax):.3f}, {jnp.max(frac_deriv_jax):.3f}]")
        
    except Exception as e:
        print(f"  Fractional calculus test failed: ❌ {e}")
    
    print("\n🎉 JAX-CuPy Integration Test Complete!")
    print("=" * 50)

def test_jax_autodiff_with_cupy():
    """Test JAX automatic differentiation with CuPy data"""
    print("\n🎯 JAX Autodiff with CuPy Test")
    print("=" * 50)
    
    def loss_function(x):
        """Loss function that can work with JAX arrays"""
        # Convert to CuPy for computation
        x_cupy = cp.asarray(x)
        # Compute some function
        y = cp.sin(x_cupy) * cp.cos(x_cupy)
        # Convert back to JAX
        y_jax = jnp.array(y)
        return jnp.sum(y_jax ** 2)
    
    # Test gradient computation
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    
    try:
        # Compute gradient
        grad_fn = jax.grad(loss_function)
        gradient = grad_fn(x)
        
        print(f"  Input: {x}")
        print(f"  Gradient: {gradient}")
        print("  ✅ JAX autodiff with CuPy: Success")
        
    except Exception as e:
        print(f"  ❌ JAX autodiff with CuPy failed: {e}")

if __name__ == "__main__":
    test_jax_cupy_integration()
    test_jax_autodiff_with_cupy()
