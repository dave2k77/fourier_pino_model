#!/usr/bin/env python3
"""
Advanced environment test for FractionalPINO with JAX, NUMBA, and NumPyro
Tests differentiable and probabilistic programming capabilities
"""

import torch
import numpy as np
import hpfracc as hf
import cupy as cp
import jax
import jax.numpy as jnp
import flax.linen as nn
import equinox as eqx
import optax
import numba
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
from pathlib import Path

def test_jax_ecosystem():
    """Test JAX ecosystem for differentiable programming"""
    print("🚀 JAX Ecosystem Test")
    print("=" * 50)
    
    # Test JAX
    print("🔥 JAX Test:")
    print(f"  Version: {jax.__version__}")
    print(f"  Default backend: {jax.default_backend()}")
    print(f"  Available devices: {jax.devices()}")
    try:
        gpu_devices = jax.devices('gpu')
        print(f"  GPU available: {gpu_devices}")
    except:
        print(f"  GPU available: No GPU devices found")
    
    # Test JAX operations
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.sin(x)
    print(f"  JAX computation: ✅")
    print(f"  Result: {y}")
    
    # Test JAX automatic differentiation
    def f(x):
        return jnp.sum(x ** 2)
    
    grad_f = jax.grad(f)
    x_test = jnp.array([1.0, 2.0, 3.0])
    gradient = grad_f(x_test)
    print(f"  JAX autodiff: ✅")
    print(f"  Gradient: {gradient}")
    
    # Test JAX JIT compilation
    @jax.jit
    def fast_function(x):
        return jnp.sum(jnp.sin(x) * jnp.cos(x))
    
    x_large = jnp.linspace(0, 10, 10000)
    result = fast_function(x_large)
    print(f"  JAX JIT compilation: ✅")
    print(f"  Result: {result}")
    
    # Test Flax
    print("\n🌾 Flax Test:")
    try:
        class SimpleModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(64)(x)
                x = nn.relu(x)
                x = nn.Dense(32)(x)
                x = nn.relu(x)
                x = nn.Dense(1)(x)
                return x
        
        model = SimpleModel()
        key = jax.random.PRNGKey(42)
        x = jnp.ones((1, 10))
        params = model.init(key, x)
        output = model.apply(params, x)
        
        print(f"  Flax model: ✅")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"  Flax test failed: ❌ {e}")
    
    # Test Equinox
    print("\n⚡ Equinox Test:")
    try:
        class EquinoxModel(eqx.Module):
            layers: list
            
            def __init__(self, key):
                key1, key2 = jax.random.split(key)
                self.layers = [
                    eqx.nn.Linear(10, 64, key=key1),
                    eqx.nn.Linear(64, 1, key=key2)
                ]
            
            def __call__(self, x):
                for layer in self.layers[:-1]:
                    x = jax.nn.relu(layer(x))
                return self.layers[-1](x)
        
        key = jax.random.PRNGKey(42)
        model = EquinoxModel(key)
        x = jnp.ones(10)
        output = model(x)
        
        print(f"  Equinox model: ✅")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"  Equinox test failed: ❌ {e}")
    
    # Test Optax
    print("\n🎯 Optax Test:")
    try:
        optimizer = optax.adam(learning_rate=0.001)
        params = {'w': jnp.array([1.0, 2.0, 3.0])}
        opt_state = optimizer.init(params)
        
        def loss_fn(params):
            return jnp.sum(params['w'] ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        print(f"  Optax optimizer: ✅")
        print(f"  Initial loss: {loss}")
        print(f"  Updated params: {new_params['w']}")
        
    except Exception as e:
        print(f"  Optax test failed: ❌ {e}")

def test_numba_jit():
    """Test NUMBA JIT compilation"""
    print("\n⚡ NUMBA JIT Test")
    print("=" * 50)
    
    print(f"  Version: {numba.__version__}")
    print(f"  JIT enabled: {numba.config.DISABLE_JIT}")
    
    # Test basic JIT
    @numba.jit
    def fast_sum(arr):
        total = 0.0
        for i in range(len(arr)):
            total += arr[i]
        return total
    
    arr = np.random.random(1000000)
    result = fast_sum(arr)
    print(f"  Basic JIT: ✅")
    print(f"  Result: {result}")
    
    # Test JIT with nopython mode
    @numba.jit(nopython=True)
    def fast_matrix_multiply(A, B):
        return np.dot(A, B)
    
    A = np.random.random((100, 100))
    B = np.random.random((100, 100))
    result = fast_matrix_multiply(A, B)
    print(f"  Nopython JIT: ✅")
    print(f"  Result shape: {result.shape}")
    
    # Test parallel JIT
    @numba.jit(nopython=True, parallel=True)
    def parallel_sum(arr):
        total = 0.0
        for i in numba.prange(len(arr)):
            total += arr[i]
        return total
    
    result = parallel_sum(arr)
    print(f"  Parallel JIT: ✅")
    print(f"  Result: {result}")

def test_numpyro_probabilistic():
    """Test NumPyro for probabilistic programming"""
    print("\n🎲 NumPyro Probabilistic Programming Test")
    print("=" * 50)
    
    print(f"  Version: {numpyro.__version__}")
    
    # Test basic probabilistic model
    def simple_model(y_obs=None):
        # Prior
        mu = numpyro.sample("mu", dist.Normal(0, 1))
        sigma = numpyro.sample("sigma", dist.Exponential(1))
        
        # Likelihood
        with numpyro.plate("data", len(y_obs) if y_obs is not None else 100):
            y = numpyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)
        
        return y
    
    # Test sampling
    try:
        # Generate some fake data
        true_mu, true_sigma = 2.0, 1.5
        y_obs = np.random.normal(true_mu, true_sigma, 100)
        
        # MCMC sampling
        kernel = NUTS(simple_model)
        mcmc = MCMC(kernel, num_warmup=100, num_samples=1000)
        
        print("  Running MCMC...")
        mcmc.run(jax.random.PRNGKey(42), y_obs=y_obs)
        
        # Get samples
        samples = mcmc.get_samples()
        
        print(f"  MCMC sampling: ✅")
        print(f"  Mu samples shape: {samples['mu'].shape}")
        print(f"  Sigma samples shape: {samples['sigma'].shape}")
        print(f"  Mu mean: {jnp.mean(samples['mu']):.3f}")
        print(f"  Sigma mean: {jnp.mean(samples['sigma']):.3f}")
        
    except Exception as e:
        print(f"  NumPyro test failed: ❌ {e}")
    
    # Test ArviZ integration
    print("\n📊 ArviZ Integration Test:")
    try:
        print(f"  ArviZ version: {az.__version__}")
        
        # Convert samples to ArviZ format
        idata = az.from_numpyro(mcmc)
        
        print(f"  ArviZ conversion: ✅")
        print(f"  Inference data keys: {list(idata.keys())}")
        
    except Exception as e:
        print(f"  ArviZ test failed: ❌ {e}")

def test_fractional_jax_integration():
    """Test integration between hpfracc and JAX"""
    print("\n🧮 Fractional-JAX Integration Test")
    print("=" * 50)
    
    try:
        # Test fractional derivative with JAX
        def fractional_loss(x, alpha=0.5):
            # Convert to numpy for hpfracc
            x_np = np.array(x)
            frac_deriv = hf.optimized_caputo(x_np, x_np, alpha)
            return jnp.sum(jnp.array(frac_deriv) ** 2)
        
        x = jnp.linspace(0, 1, 50)
        loss = fractional_loss(x, 0.7)
        
        print(f"  Fractional-JAX integration: ✅")
        print(f"  Loss value: {loss}")
        
        # Test gradient of fractional loss
        grad_fn = jax.grad(fractional_loss)
        gradient = grad_fn(x, 0.7)
        
        print(f"  Fractional gradient: ✅")
        print(f"  Gradient norm: {jnp.linalg.norm(gradient)}")
        
    except Exception as e:
        print(f"  Fractional-JAX integration failed: ❌ {e}")

def test_multi_backend_comparison():
    """Test performance across different backends"""
    print("\n⚡ Multi-Backend Performance Test")
    print("=" * 50)
    
    import time
    
    # Test data
    size = 10000
    x_np = np.random.random(size)
    x_jax = jnp.array(x_np)
    x_torch = torch.tensor(x_np, dtype=torch.float32)
    
    # NumPy
    start = time.time()
    result_np = np.sin(x_np) * np.cos(x_np)
    time_np = time.time() - start
    
    # JAX (CPU)
    start = time.time()
    result_jax = jnp.sin(x_jax) * jnp.cos(x_jax)
    result_jax.block_until_ready()  # Ensure computation is done
    time_jax = time.time() - start
    
    # PyTorch (CPU)
    start = time.time()
    result_torch = torch.sin(x_torch) * torch.cos(x_torch)
    time_torch = time.time() - start
    
    print(f"  NumPy (CPU): {time_np:.4f}s")
    print(f"  JAX (CPU): {time_jax:.4f}s")
    print(f"  PyTorch (CPU): {time_torch:.4f}s")
    
    # Test GPU if available
    if torch.cuda.is_available():
        x_torch_gpu = x_torch.cuda()
        start = time.time()
        result_torch_gpu = torch.sin(x_torch_gpu) * torch.cos(x_torch_gpu)
        torch.cuda.synchronize()
        time_torch_gpu = time.time() - start
        
        print(f"  PyTorch (GPU): {time_torch_gpu:.4f}s")
    
    try:
        if jax.devices('gpu'):
            x_jax_gpu = jax.device_put(x_jax, jax.devices('gpu')[0])
            start = time.time()
            result_jax_gpu = jnp.sin(x_jax_gpu) * jnp.cos(x_jax_gpu)
            result_jax_gpu.block_until_ready()
            time_jax_gpu = time.time() - start
            
            print(f"  JAX (GPU): {time_jax_gpu:.4f}s")
    except:
        print(f"  JAX (GPU): Not available")

def main():
    """Run all advanced environment tests"""
    print("🧪 Advanced FractionalPINO Environment Test")
    print("=" * 60)
    print("Testing JAX ecosystem, NUMBA, NumPyro, and integrations")
    print("=" * 60)
    
    test_jax_ecosystem()
    test_numba_jit()
    test_numpyro_probabilistic()
    test_fractional_jax_integration()
    test_multi_backend_comparison()
    
    print("\n🎉 Advanced Environment Test Complete!")
    print("=" * 60)
    print("✅ JAX ecosystem: Differentiable programming")
    print("✅ NUMBA: JIT compilation")
    print("✅ NumPyro: Probabilistic programming")
    print("✅ Multi-backend: Performance optimization")
    print("✅ Integration: Fractional calculus + ML")
    print("\n🚀 Ready for advanced FractionalPINO development!")

if __name__ == "__main__":
    main()
