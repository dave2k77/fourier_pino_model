# Makefile for Fourier PINO Model

.PHONY: help install install-dev test clean lint format docs run-example

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo "  docs         - Generate documentation"
	@echo "  run-example  - Run basic usage example"
	@echo "  train        - Train model with default settings"
	@echo ""
	@echo "Baseline Reproduction:"
	@echo "  baseline-test        - Test baseline reproduction setup"
	@echo "  baseline-reproduce   - Run full baseline reproduction"
	@echo "  baseline-reproduce-single - Run single experiment (B2)"
	@echo "  validate-baseline    - Quick validation of baseline"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Run tests
test:
	python -m pytest tests/ -v

# Run linting
lint:
	flake8 src/ tests/ --max-line-length=88 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ tests/ --line-length=88
	isort src/ tests/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/*/__pycache__/
	rm -rf tests/__pycache__/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Generate documentation
docs:
	@echo "Documentation generation not implemented yet"
	@echo "Please check the README.md for documentation"

# Run basic usage example
run-example:
	python examples/basic_usage.py

# Train model with default settings
train:
	python train_pino.py --verbose

# Train with custom settings
train-custom:
	python train_pino.py --epochs 50 --lr 0.001 --physics_coeff 0.1 --verbose

# Run experiment A
train-exp-a:
	python train_pino.py --config experiment_a --verbose

# Run experiment B
train-exp-b:
	python train_pino.py --config experiment_b --verbose

# Baseline reproduction
baseline-test:
	python scripts/test_baseline.py

baseline-reproduce:
	python scripts/reproduce_baseline.py

baseline-reproduce-single:
	python scripts/reproduce_baseline.py --experiment experiment_b2

# Quick validation
validate-baseline: baseline-test
	@echo "Baseline validation completed"

# Check code quality
check: lint test
	@echo "Code quality check completed"

# Setup development environment
setup-dev: install-dev
	@echo "Development environment setup completed"
	@echo "Run 'make test' to verify installation"

# Show project info
info:
	@echo "Fourier PINO Model"
	@echo "=================="
	@echo "Python version: $(shell python --version)"
	@echo "PyTorch version: $(shell python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())')"
	@echo "Project structure:"
	@tree -I '__pycache__|*.pyc|*.egg-info|build|dist' -L 3
