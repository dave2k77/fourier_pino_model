from .training import (
    LossBreakdown,
    train,
    loss_function,
    compute_loss_breakdown,
    fourier_derivative_2d,
    energy_conservation_loss,
)
from .visualization import plot_loss, compare_solutions, plot_r2_vs_physics_loss_coefficients
from .metrics import calculate_r2_score

__all__ = [
    "train",
    "LossBreakdown",
    "loss_function", 
    "compute_loss_breakdown",
    "fourier_derivative_2d",
    "energy_conservation_loss",
    "plot_loss",
    "compare_solutions",
    "plot_r2_vs_physics_loss_coefficients",
    "calculate_r2_score"
]
