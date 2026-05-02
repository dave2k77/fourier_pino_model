#!/usr/bin/env python3
"""Generate a compact 2D heat-equation dataset for PINO experiments."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def heat_solution(alpha, time_value, grid_size):
    """Analytical solution for u_t = alpha * (u_xx + u_yy)."""
    x = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    initial = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    decay = np.exp(-2.0 * (np.pi ** 2) * alpha * time_value)
    return (initial * decay).astype(np.float32)


def to_heatmap(solution):
    """Convert a solution array to an 8-bit grayscale heatmap."""
    minimum = float(solution.min())
    maximum = float(solution.max())
    if maximum == minimum:
        scaled = np.zeros_like(solution, dtype=np.uint8)
    else:
        scaled = ((solution - minimum) / (maximum - minimum) * 255.0).astype(np.uint8)
    return scaled


def main():
    parser = argparse.ArgumentParser(description="Generate 2D heat-equation PINO data")
    parser.add_argument("--output_dir", default="data", help="Directory for generated data")
    parser.add_argument("--grid_size", type=int, default=64, help="Square grid size")
    parser.add_argument("--time_steps", type=int, default=20, help="Number of time samples")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="Thermal diffusivity values",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    heatmap_dir = output_dir / "heatmaps"
    solution_dir = output_dir / "pde_solutions"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)

    times = np.linspace(0.0, 1.0, args.time_steps, dtype=np.float32)
    for alpha in args.alphas:
        alpha_label = str(alpha).replace(".", "p")
        for index, time_value in enumerate(times):
            solution = heat_solution(alpha, float(time_value), args.grid_size)
            stem = f"alpha_{alpha_label}_timestep_{index:03d}"
            Image.fromarray(to_heatmap(solution)).save(heatmap_dir / f"{stem}.png")
            np.savez_compressed(
                solution_dir / f"{stem}.npz",
                solution=solution,
                alpha=np.float32(alpha),
                time=np.float32(time_value),
            )

    print(f"Generated {len(args.alphas) * args.time_steps} samples in {output_dir}")
    print(f"Heatmaps: {heatmap_dir}")
    print(f"PDE solutions: {solution_dir}")


if __name__ == "__main__":
    main()
