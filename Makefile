# Makefile for the focused Fourier PINO research artifact

.PHONY: help install install-dev test lint format clean data train sweep check

help:
	@echo "Available commands:"
	@echo "  install      Install runtime dependencies"
	@echo "  install-dev  Install runtime and development dependencies"
	@echo "  test         Run unit tests"
	@echo "  lint         Run static checks"
	@echo "  format       Format active source and tests"
	@echo "  data         Generate a compact heat-equation dataset"
	@echo "  train        Run one baseline training job"
	@echo "  sweep        Run the canonical optimizer x physics-loss sweep"
	@echo "  clean        Remove local caches and generated outputs"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	python -m pytest tests/ -q

lint:
	python -m compileall -q src train_pino.py config.py scripts tests
	flake8 src/ tests/ scripts/ train_pino.py config.py --max-line-length=120 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ train_pino.py config.py --line-length=100

data:
	python scripts/generate_heat_equation_data.py --output_dir data --grid_size 64 --time_steps 20

train:
	python train_pino.py --mode single --heatmap_folder data/heatmaps \
		--pde_folder data/pde_solutions --verbose

sweep:
	python train_pino.py --mode sweep --heatmap_folder data/heatmaps \
		--pde_folder data/pde_solutions

check: lint test

clean:
	rm -rf build/ dist/ *.egg-info/ outputs/ .pytest_cache/ .mypy_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
