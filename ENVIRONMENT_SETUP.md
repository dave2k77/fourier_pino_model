# Environment Setup

Use a Python virtual environment for the active 2D heat-equation PINO project.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Verify the installation:

```bash
python -m compileall -q src train_pino.py config.py tests
python -m pytest tests/ -q
```

The cleaned core does not require HPFRACC, JAX, CuPy, wandb, optuna, or Hydra.
Those dependencies belonged to archived future-work experiments.
