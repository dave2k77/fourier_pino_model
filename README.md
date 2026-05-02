# Fourier PINO Model

A focused research artifact demonstrating Physics-Informed Neural Operators (PINOs)
for the 2D heat equation.

The project investigates how a Fourier-domain neural operator performs when the
training objective balances two terms:

- data-driven loss, which fits generated PDE solution fields;
- physics-informed loss, which encourages consistency with heat-equation
  conservation behavior.

The maintained code path is intentionally small: a Fourier encoder, a neural
operator over complex spectral coefficients, an inverse Fourier decoder, dataset
loading utilities, and a reproducible optimizer x physics-loss sweep.

## Project Structure

```text
src/
  layers/       Fourier transform, neural operator, inverse transform layers
  models/       PINO_2D_Heat_Equation
  data/         Heatmap/PDE solution dataset loader
  utils/        training, losses, metrics, and visualisation helpers
train_pino.py   CLI for single training runs and the canonical experiment sweep
tests/          unit tests for the active implementation
archive/        old research notes, fractional PINO future work, and prior results
images/         curated diagrams and representative heat-equation visual
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

The active project uses a lean dependency set. Fractional-PINO, HPFRACC, JAX,
CuPy, and broader biophysics explorations are archived as future work and are not
required for the 2D heat-equation experiments.

## Quick Checks

```bash
make test
```

or directly:

```bash
python -m compileall -q src train_pino.py config.py tests
python -m pytest tests/ -q
```

## Training

The default data paths are:

- `data/heatmaps`
- `data/pde_solutions`

Large generated datasets are not part of the cleaned active tree. Provide those
folders locally, or pass custom paths:

```bash
python scripts/generate_heat_equation_data.py --output_dir data --grid_size 64 --time_steps 20
```

```bash
python train_pino.py \
  --mode single \
  --heatmap_folder data/heatmaps \
  --pde_folder data/pde_solutions \
  --epochs 100 \
  --optimizer Adam \
  --physics_coeff 0.01 \
  --verbose
```

Outputs are written under `outputs/`:

- `outputs/models/*.pth`
- `outputs/results/*.json`
- `outputs/results/*.csv`

## Canonical Experiment

The core research comparison is a 2 x 3 sweep:

- optimizers: `SGD`, `Adam`
- physics-loss coefficients: `0.001`, `0.01`, `0.1`

Run it with:

```bash
python train_pino.py \
  --mode sweep \
  --heatmap_folder data/heatmaps \
  --pde_folder data/pde_solutions \
  --epochs 100
```

The sweep records total loss, data loss, physics loss, and R2 score for each
configuration so the tradeoff between data fitting and physics constraints can be
analysed directly.

## Active API

```python
from src.models import PINO_2D_Heat_Equation
from src.data import HeatmapPDEDataset, split_data
from src.utils import train, loss_function, compute_loss_breakdown
```

`loss_function` returns the scalar training objective. Use
`compute_loss_breakdown` when reporting separate data and physics terms.

## Archive

`archive/` contains prior drafts, result summaries, and fractional-PINO work.
Those files are preserved for context but are not part of the supported package,
test suite, or dependency set.
