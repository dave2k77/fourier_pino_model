# FractionalPINO Environment Setup

## 🎯 Overview

This document describes the complete environment setup for the FractionalPINO project, which integrates Physics-Informed Neural Operators (PINO) with fractional calculus using the `hpfracc` library.

## 🚀 Quick Start

### 1. Activate Environment
```bash
# Option 1: Use the activation script
./activate_env.sh

# Option 2: Manual activation
conda activate fractional-pino
```

### 2. Verify Installation
```bash
python test_environment.py
```

## 📦 Environment Details

### Conda Environment: `fractional-pino`
- **Python**: 3.11
- **PyTorch**: 2.5.1 (with CUDA 12.1 support)
- **hpfracc**: 1.5.0 (fractional calculus library)
- **CuPy**: 13.6.0 (GPU acceleration)

### Key Dependencies
- **Core ML**: PyTorch, NumPy, SciPy, scikit-learn
- **Fractional Calculus**: hpfracc (with optimized implementations)
- **GPU Acceleration**: CuPy, CUDA 12.1
- **JAX Ecosystem**: JAX, Flax, Equinox, Optax, Chex, Distrax, JAXOpt
- **Probabilistic Programming**: NumPyro, ArviZ
- **JIT Compilation**: NUMBA (CPU optimization)
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Development**: Jupyter Lab, pytest, black, flake8
- **Experiment Tracking**: Weights & Biases, TensorBoard
- **Hyperparameter Optimization**: Optuna
- **Configuration**: Hydra

## 🔧 Installation Commands

### Create Environment
```bash
conda create -n fractional-pino python=3.11 -y
conda activate fractional-pino
```

### Install PyTorch with CUDA
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Install Scientific Computing Stack
```bash
conda install -c conda-forge scipy matplotlib pandas scikit-learn jupyter jupyterlab numba -y
```

### Install Specialized Packages
```bash
pip install hpfracc wandb optuna hydra-core tensorboard tqdm cupy-cuda12x
```

## 🧪 Testing

### Environment Test
```bash
python test_environment.py
```

Expected output:
- ✅ PyTorch with CUDA support
- ✅ hpfracc fractional derivatives
- ✅ CuPy GPU acceleration
- ✅ PINO model loading
- ✅ Data loading capabilities

### Individual Component Tests
```bash
# Test PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test hpfracc
python -c "import hpfracc as hf; print(f'Version: {hf.__version__}')"

# Test CuPy
python -c "import cupy as cp; print(f'CUDA: {cp.cuda.is_available()}')"

# Test JAX
python -c "import jax; print(f'JAX: {jax.__version__}')"

# Test JAX-CuPy Integration
python test_jax_cupy_workflow.py

# Test Advanced Environment
python test_advanced_environment.py
```

## 🎯 Key Features

### 1. Fractional Calculus Integration
- **Caputo Derivatives**: `hf.optimized_caputo(f, t, alpha)`
- **Riemann-Liouville**: `hf.optimized_riemann_liouville(f, t, alpha)`
- **High-Performance**: Optimized implementations with GPU support

### 2. PINO Model Architecture
- **Encoder**: Fourier Transform Layer
- **Neural Operator**: Multi-layer perceptron in frequency domain
- **Decoder**: Inverse Fourier Transform Layer
- **Physics Loss**: Heat equation constraints

### 3. GPU Acceleration
- **CUDA 12.1**: Latest CUDA support
- **CuPy**: GPU-accelerated array operations
- **PyTorch**: Automatic GPU utilization
- **JAX-CuPy Integration**: Seamless data transfer between frameworks

## 🔄 Development Workflow

### 1. Start Development
```bash
conda activate fractional-pino
jupyter lab  # Start Jupyter Lab
```

### 2. Run Experiments
```bash
python train_pino.py  # Train PINO model
python scripts/enhanced_training.py  # Enhanced training
```

### 3. Testing
```bash
pytest tests/  # Run test suite
python -m pytest tests/test_model.py -v  # Specific tests
```

### 4. Code Quality
```bash
black src/  # Format code
flake8 src/  # Lint code
mypy src/  # Type checking
```

## 📁 Project Structure

```
fractional-pino/
├── src/                    # Source code
│   ├── PINO_2D_Heat_Equation.py
│   ├── layers/            # Neural network layers
│   ├── data/              # Data handling
│   └── utils/             # Utility functions
├── scripts/               # Training scripts
├── tests/                 # Test suite
├── examples/              # Usage examples
├── environment.yml        # Conda environment
├── requirements.txt       # Python dependencies
├── test_environment.py    # Environment verification
└── activate_env.sh        # Activation script
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Not Available**
   ```bash
   # Check CUDA installation
   nvidia-smi
   # Reinstall PyTorch with CUDA
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

2. **hpfracc Import Error**
   ```bash
   # Reinstall hpfracc
   pip install --upgrade hpfracc
   ```

3. **CuPy Installation Issues**
   ```bash
   # Install correct CuPy version
   pip install cupy-cuda12x
   ```

### Environment Reset
```bash
# Remove and recreate environment
conda env remove -n fractional-pino
conda create -n fractional-pino python=3.11 -y
# Follow installation steps again
```

## 🎉 Success Indicators

Your environment is correctly set up when:
- ✅ `python test_environment.py` runs without errors
- ✅ All fractional derivative tests pass
- ✅ GPU acceleration is working
- ✅ PINO model loads successfully
- ✅ Jupyter Lab starts without issues

## 📚 Next Steps

1. **Run Baseline Experiments**: `python scripts/reproduce_baseline.py`
2. **Explore FractionalPINO**: Start with `examples/basic_usage.py`
3. **Develop New Features**: Use the established architecture
4. **Contribute**: Follow the development workflow

---

**Environment Status**: ✅ Ready for FractionalPINO Development
**Last Updated**: January 2025
**Maintainer**: FractionalPINO Development Team
