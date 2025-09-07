#!/usr/bin/env python3
"""
FractionalPINO Experimental Execution Script
Generate comprehensive experimental data for paper submission
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import FractionalPINO components
from src.models.working_fractional_pino import (
    WorkingFractionalPINO, 
    MultiMethodWorkingFractionalPINO,
    WorkingFractionalPhysicsLoss
)

# Import baseline methods
from src.PINO_2D_Heat_Equation import PINO_2D_Heat_Equation
from src.layers.neural_operator import NeuralOperator

# Import data generation
from src.data.fractional_data_generator import (
    generate_fractional_heat_data, 
    generate_fractional_wave_data,
    generate_fractional_diffusion_data,
    generate_multi_scale_data
)

class FractionalPINOExperimentRunner:
    """Main experiment runner for FractionalPINO validation"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_directories()
        
    def setup_directories(self):
        """Setup experiment directories"""
        self.results_dir = Path("research/experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path("research/experiments/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def run_single_method_experiment(self, problem_type, method, alpha, config):
        """Run experiment for single fractional method"""
        print(f"Running {method} experiment for {problem_type} with α={alpha}")
        
        # Initialize model
        model = WorkingFractionalPINO(
            modes=config['modes'],
            width=config['width'],
            alpha=alpha,
            fractional_method=method
        ).to(self.device)
        
        # Generate data
        if problem_type == 'heat':
            train_data, test_data = generate_fractional_heat_data(alpha=alpha)
        elif problem_type == 'wave':
            train_data, test_data = generate_fractional_wave_data(alpha=alpha)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        # Training
        start_time = time.time()
        training_history = self.train_model(model, train_data, config)
        training_time = time.time() - start_time
        
        # Evaluation
        metrics = self.evaluate_model(model, test_data)
        metrics['training_time'] = training_time
        metrics['memory_usage'] = self.get_memory_usage()
        
        return {
            'method': method,
            'alpha': alpha,
            'problem_type': problem_type,
            'config': config,
            'training_history': training_history,
            'metrics': metrics
        }
    
    def run_multi_method_experiment(self, problem_type, methods, alpha, config):
        """Run experiment for multi-method FractionalPINO"""
        print(f"Running multi-method experiment for {problem_type} with α={alpha}")
        
        # Initialize multi-method model
        model = MultiMethodWorkingFractionalPINO(
            modes=config['modes'],
            width=config['width'],
            alpha=alpha,
            methods=methods
        ).to(self.device)
        
        # Generate data
        if problem_type == 'heat':
            train_data, test_data = generate_fractional_heat_data(alpha=alpha)
        elif problem_type == 'wave':
            train_data, test_data = generate_fractional_wave_data(alpha=alpha)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        # Training
        start_time = time.time()
        training_history = self.train_model(model, train_data, config)
        training_time = time.time() - start_time
        
        # Evaluation
        metrics = self.evaluate_model(model, test_data)
        metrics['training_time'] = training_time
        metrics['memory_usage'] = self.get_memory_usage()
        
        return {
            'methods': methods,
            'alpha': alpha,
            'problem_type': problem_type,
            'config': config,
            'training_history': training_history,
            'metrics': metrics
        }
    
    def run_baseline_comparison(self, problem_type, alpha, config):
        """Run comparison with baseline methods"""
        print(f"Running baseline comparison for {problem_type} with α={alpha}")
        
        results = {}
        
        # Traditional PINN
        try:
            pinn_model = self.create_traditional_pinn()
            results['pinn'] = self.train_and_evaluate(pinn_model, problem_type, alpha, config)
        except Exception as e:
            print(f"PINN experiment failed: {e}")
            results['pinn'] = None
        
        # FNO
        try:
            fno_model = self.create_fno(config)
            results['fno'] = self.train_and_evaluate(fno_model, problem_type, alpha, config)
        except Exception as e:
            print(f"FNO experiment failed: {e}")
            results['fno'] = None
        
        # PINO
        try:
            pino_model = self.create_pino(config)
            results['pino'] = self.train_and_evaluate(pino_model, problem_type, alpha, config)
        except Exception as e:
            print(f"PINO experiment failed: {e}")
            results['pino'] = None
        
        # FractionalPINO
        try:
            fractional_pino_model = WorkingFractionalPINO(
                modes=config['modes'],
                width=config['width'],
                alpha=alpha,
                fractional_method='caputo'
            ).to(self.device)
            results['fractional_pino'] = self.train_and_evaluate(fractional_pino_model, problem_type, alpha, config)
        except Exception as e:
            print(f"FractionalPINO experiment failed: {e}")
            results['fractional_pino'] = None
        
        return results
    
    def train_model(self, model, train_data, config):
        """Train model with given configuration"""
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        
        training_history = {
            'loss': [],
            'l2_error': [],
            'learning_rate': []
        }
        
        for epoch in range(config['epochs']):
            # Training step
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            x, y_true = train_data
            x, y_true = x.to(self.device), y_true.to(self.device)
            
            y_pred = model(x)
            
            # Compute loss
            data_loss = nn.MSELoss()(y_pred, y_true)
            
            # Physics loss (if applicable)
            if hasattr(model, 'physics_loss'):
                physics_loss = model.physics_loss(x, y_pred)
                total_loss = data_loss + config['physics_weight'] * physics_loss
            else:
                total_loss = data_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Record history
            training_history['loss'].append(total_loss.item())
            training_history['learning_rate'].append(scheduler.get_last_lr()[0])
            
            # Validation
            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(x)
                    l2_error = torch.norm(val_pred - y_true) / torch.norm(y_true)
                    training_history['l2_error'].append(l2_error.item())
                
                print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}, L2 Error = {l2_error.item():.6f}")
        
        return training_history
    
    def evaluate_model(self, model, test_data):
        """Evaluate model on test data"""
        model.eval()
        x_test, y_test = test_data
        x_test, y_test = x_test.to(self.device), y_test.to(self.device)
        
        with torch.no_grad():
            y_pred = model(x_test)
            
            # Compute metrics
            l2_error = torch.norm(y_pred - y_test) / torch.norm(y_test)
            linf_error = torch.max(torch.abs(y_pred - y_test))
            mse = nn.MSELoss()(y_pred, y_test)
            mae = nn.L1Loss()(y_pred, y_test)
            
            return {
                'l2_error': l2_error.item(),
                'linf_error': linf_error.item(),
                'mse': mse.item(),
                'mae': mae.item()
            }
    
    def get_memory_usage(self):
        """Get current memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            import psutil
            return psutil.Process().memory_info().rss / 1024**3  # GB
    
    def save_results(self, results, filename):
        """Save results to file"""
        results_file = self.results_dir / filename
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {results_file}")
    
    def run_comprehensive_experiments(self):
        """Run comprehensive experimental validation"""
        print("Starting comprehensive FractionalPINO experiments...")
        
        # Configuration
        config = {
            'modes': 12,
            'width': 64,
            'epochs': 1000,
            'learning_rate': 0.001,
            'physics_weight': 0.1
        }
        
        # Problem configurations
        problems = ['heat', 'wave']
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        methods = ['caputo', 'riemann_liouville', 'caputo_fabrizio', 'atangana_baleanu']
        
        # Phase 1: Single method experiments
        print("\n=== Phase 1: Single Method Experiments ===")
        single_method_results = {}
        
        for problem in problems:
            single_method_results[problem] = {}
            for alpha in alphas:
                single_method_results[problem][alpha] = {}
                for method in methods:
                    try:
                        result = self.run_single_method_experiment(problem, method, alpha, config)
                        single_method_results[problem][alpha][method] = result
                    except Exception as e:
                        print(f"Error in {method} experiment: {e}")
                        single_method_results[problem][alpha][method] = None
        
        self.save_results(single_method_results, 'single_method_results.json')
        
        # Phase 2: Multi-method experiments
        print("\n=== Phase 2: Multi-Method Experiments ===")
        multi_method_results = {}
        
        for problem in problems:
            multi_method_results[problem] = {}
            for alpha in alphas:
                try:
                    result = self.run_multi_method_experiment(problem, methods, alpha, config)
                    multi_method_results[problem][alpha] = result
                except Exception as e:
                    print(f"Error in multi-method experiment: {e}")
                    multi_method_results[problem][alpha] = None
        
        self.save_results(multi_method_results, 'multi_method_results.json')
        
        # Phase 3: Baseline comparison
        print("\n=== Phase 3: Baseline Comparison ===")
        baseline_results = {}
        
        for problem in problems:
            baseline_results[problem] = {}
            for alpha in alphas:
                try:
                    result = self.run_baseline_comparison(problem, alpha, config)
                    baseline_results[problem][alpha] = result
                except Exception as e:
                    print(f"Error in baseline comparison: {e}")
                    baseline_results[problem][alpha] = None
        
        self.save_results(baseline_results, 'baseline_comparison_results.json')
        
        print("\n=== Experiments Complete ===")
        print(f"Results saved in: {self.results_dir}")
        
        return {
            'single_method': single_method_results,
            'multi_method': multi_method_results,
            'baseline': baseline_results
        }

def main():
    """Main execution function"""
    print("FractionalPINO Experimental Execution")
    print("=" * 50)
    
    # Configuration
    config = {
        'modes': 12,
        'width': 64,
        'epochs': 1000,
        'learning_rate': 0.001,
        'physics_weight': 0.1
    }
    
    # Initialize experiment runner
    runner = FractionalPINOExperimentRunner(config)
    
    # Run comprehensive experiments
    results = runner.run_comprehensive_experiments()
    
    print("\nExperimental execution completed successfully!")
    print(f"Total experiments run: {len(results)}")
    print(f"Results directory: {runner.results_dir}")

if __name__ == "__main__":
    main()
