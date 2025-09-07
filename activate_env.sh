#!/bin/bash
# Activation script for FractionalPINO project

echo "🚀 Activating FractionalPINO Environment"
echo "========================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fractional-pino

# Verify environment
echo "✅ Environment activated: fractional-pino"
echo "🐍 Python version: $(python --version)"
echo "🔥 PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "🎯 CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "🧮 hpfracc version: $(python -c 'import hpfracc; print(hpfracc.__version__)')"
echo "⚡ CuPy version: $(python -c 'import cupy; print(cupy.__version__)')"

echo ""
echo "🎉 Ready for FractionalPINO development!"
echo "📁 Project directory: $(pwd)"
echo ""
echo "Available commands:"
echo "  make help          - Show available commands"
echo "  python train_pino.py - Train PINO model"
echo "  jupyter lab        - Start Jupyter Lab"
echo "  pytest            - Run tests"
echo ""
