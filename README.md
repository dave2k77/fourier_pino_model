# Fourier PINO Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Physics-Informed Neural Operator (PINO) implementation for solving 2D Heat Equation using Fourier analysis techniques. This project was developed as part of a Master's thesis on balancing data fitting and physical properties in neural PDE solvers.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a Physics-Informed Neural Operator (PINO) that combines Fourier analysis with neural networks to solve partial differential equations efficiently. The model architecture consists of three main components:

1. **Encoder**: Fourier Transform Layer that converts spatial domain data to frequency domain
2. **Neural Operator**: Multi-layer perceptron that learns the PDE operator
3. **Decoder**: Inverse Fourier Transform Layer that converts back to spatial domain

The PINO approach offers several advantages over traditional numerical methods:
- Reduced computational complexity through Fourier domain processing
- Physics-informed loss functions that enforce conservation laws
- Ability to handle incomplete or noisy data
- Fast inference once trained

## ✨ Features

- **Modular Architecture**: Clean separation of model components, data handling, and training utilities
- **Physics-Informed Loss**: Combines data fitting with physical conservation laws
- **Fourier Analysis**: Efficient frequency domain processing
- **Configurable Training**: Easy hyperparameter tuning and experiment management
- **Comprehensive Documentation**: Detailed API documentation and usage examples
- **Reproducible Results**: Fixed random seeds and experiment tracking

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/fourier_pino_model.git
cd fourier_pino_model

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 🏃 Quick Start

### Basic Training

```python
from src.models import PINO_2D_Heat_Equation
from src.data import HeatmapPDEDataset, split_data
from src.utils import train, loss_function
from torch.utils.data import DataLoader
import torch.optim as optim

# Initialize model
model = PINO_2D_Heat_Equation()

# Load data
dataset = HeatmapPDEDataset("images/heatmaps", "images/pde_solutions")
train_dataset, test_dataset = split_data(dataset)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.005)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train model
train_loss, test_loss, r2_score = train(
    model=model,
    loss_fn=loss_function,
    optimizer=optimizer,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=100,
    physics_loss_coefficient=0.01,
    device=device
)
```

### Command Line Training

```bash
# Train with default configuration
python train_pino.py

# Train with custom parameters
python train_pino.py --epochs 200 --lr 0.001 --physics_coeff 0.1 --verbose

# Train with specific experiment configuration
python train_pino.py --config experiment_a --epochs 100
```

## 📁 Project Structure

```
fourier_pino_model/
├── src/                          # Source code
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   └── pino_model.py         # Main PINO model
│   ├── layers/                   # Neural network layers
│   │   ├── __init__.py
│   │   ├── fourier_transform.py  # Fourier transform layer
│   │   ├── neural_operator.py    # Neural operator layer
│   │   └── inverse_transform.py  # Inverse transform layer
│   ├── data/                     # Data handling
│   │   ├── __init__.py
│   │   └── dataset.py            # Dataset classes
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── training.py           # Training functions
│   │   ├── visualization.py      # Plotting utilities
│   │   └── metrics.py            # Evaluation metrics
│   └── __init__.py
├── config.py                     # Configuration management
├── train_pino.py                 # Main training script
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── README.md                     # This file
├── images/                       # Data and visualizations
│   ├── heatmaps/                 # Heatmap images
│   ├── pde_solutions/            # PDE solution files
│   └── *.svg                     # Architecture diagrams
├── outputs/                      # Training outputs
│   ├── models/                   # Saved models
│   ├── plots/                    # Training plots
│   └── logs/                     # Training logs
└── graphs/                       # Generated graphs
```

## 📖 Usage

### Model Architecture

The PINO model consists of three main components:

#### 1. Fourier Transform Layer (Encoder)
```python
from src.layers import FourierTransformLayer

encoder = FourierTransformLayer()
# Transforms spatial domain → frequency domain
```

#### 2. Neural Operator
```python
from src.layers import NeuralOperator

operator = NeuralOperator(input_size=64, hidden_dims=[128, 256, 128])
# Learns PDE operator in frequency domain
```

#### 3. Inverse Fourier Transform Layer (Decoder)
```python
from src.layers import InverseFourierTransformLayer

decoder = InverseFourierTransformLayer()
# Transforms frequency domain → spatial domain
```

### Data Loading

```python
from src.data import HeatmapPDEDataset, split_data

# Load dataset
dataset = HeatmapPDEDataset(
    heatmap_folder="images/heatmaps",
    pde_solution_folder="images/pde_solutions",
    transform_size=(64, 64)
)

# Split into train/test
train_dataset, test_dataset = split_data(dataset, train_ratio=0.8)
```

### Training Configuration

```python
from config import PINOConfig, ModelConfig, TrainingConfig

# Create custom configuration
config = PINOConfig(
    model=ModelConfig(
        input_size=64,
        hidden_dims=[128, 256, 128]
    ),
    training=TrainingConfig(
        num_epochs=100,
        learning_rate=0.005,
        physics_loss_coefficient=0.01
    )
)
```

## ⚙️ Configuration

The project uses a centralized configuration system defined in `config.py`. Key configuration options:

### Model Configuration
- `input_size`: Size of input grid (default: 64)
- `hidden_dims`: Hidden layer dimensions for neural operator

### Training Configuration
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `physics_loss_coefficient`: Weight for physics loss component
- `batch_size`: Training batch size
- `device`: Training device (auto/cpu/cuda)

### Data Configuration
- `heatmap_folder`: Path to heatmap images
- `pde_solution_folder`: Path to PDE solution files
- `transform_size`: Image resize dimensions
- `output_dir`: Output directory for results

## 📊 Results

### Training Performance

The model achieves competitive performance on 2D heat equation solving:

- **R² Score**: > 0.95 on test data
- **Training Time**: ~30 minutes on GPU
- **Memory Usage**: ~2GB GPU memory

### Experiment Results

| Experiment | Optimizer | Learning Rate | Physics Coeff | R² Score |
|------------|-----------|---------------|---------------|----------|
| A1         | SGD       | 0.001         | 0.001         | 0.92     |
| A2         | SGD       | 0.005         | 0.01          | 0.94     |
| A3         | SGD       | 0.01          | 0.1           | 0.91     |
| B1         | Adam      | 0.001         | 0.001         | 0.93     |
| B2         | Adam      | 0.005         | 0.01          | 0.96     |
| B3         | Adam      | 0.01          | 0.1           | 0.94     |

### Visual Results

The model generates accurate heat equation solutions:

![Heat Equation Solution](movies/heat_equation_solution_alpha10.gif)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint code
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Li, Z., et al. "Physics-informed neural operator for learning partial differential equations." arXiv preprint arXiv:2103.03485 (2021).
2. Raissi, M., et al. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.

## 🙏 Acknowledgments

- This work was conducted as part of a Master's thesis at the University of East London.
- The PyTorch team for the excellent deep learning framework

---

**Note**: This is a research implementation. For production use, additional testing and optimization may be required.

