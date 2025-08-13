"""
Configuration file for PINO Model

This module contains all configuration parameters for the PINO model,
including hyperparameters, file paths, and experiment settings.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ModelConfig:
    """Configuration for the PINO model architecture."""
    input_size: int = 64
    hidden_dims: List[int] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 256, 128]


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.005
    physics_loss_coefficient: float = 0.01
    train_ratio: float = 0.8
    random_seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    heatmap_folder: str = "images/heatmaps"
    pde_solution_folder: str = "images/pde_solutions"
    transform_size: Tuple[int, int] = (64, 64)
    output_dir: str = "outputs"


@dataclass
class ExperimentConfig:
    """Configuration for different experiments."""
    # Experiment A: SGD optimizer with different learning rates
    experiment_a = {
        "optimizer": "SGD",
        "learning_rates": [0.001, 0.005, 0.01],
        "physics_loss_coefficients": [0.001, 0.01, 0.1]
    }
    
    # Experiment B: Adam optimizer with different learning rates
    experiment_b = {
        "optimizer": "Adam",
        "learning_rates": [0.001, 0.005, 0.01],
        "physics_loss_coefficients": [0.001, 0.01, 0.1]
    }


@dataclass
class PINOConfig:
    """Main configuration class combining all configurations."""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    experiment: ExperimentConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()
    
    def create_output_dirs(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.data.output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.data.output_dir, "logs"), exist_ok=True)
    
    def validate_paths(self):
        """Validate that required data paths exist."""
        if not os.path.exists(self.data.heatmap_folder):
            raise FileNotFoundError(f"Heatmap folder not found: {self.data.heatmap_folder}")
        if not os.path.exists(self.data.pde_solution_folder):
            raise FileNotFoundError(f"PDE solution folder not found: {self.data.pde_solution_folder}")


# Default configuration
DEFAULT_CONFIG = PINOConfig()

# Example configurations for different experiments
EXPERIMENT_A_CONFIG = PINOConfig(
    training=TrainingConfig(
        num_epochs=100,
        learning_rate=0.005,
        physics_loss_coefficient=0.01
    )
)

EXPERIMENT_B_CONFIG = PINOConfig(
    training=TrainingConfig(
        num_epochs=100,
        learning_rate=0.005,
        physics_loss_coefficient=0.01
    )
)
