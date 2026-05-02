#!/bin/bash
# Activation helper for the focused Fourier PINO environment.

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fourier-pino

echo "Environment: fourier-pino"
python --version
python -c 'import torch; print(f"PyTorch: {torch.__version__}")'
python -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}")'
echo "Run 'make help' for project commands."
