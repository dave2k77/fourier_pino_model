#!/usr/bin/env python3
"""
Fractional Diffusion Equation Benchmark
Specific experimental script for fractional diffusion equation validation
"""

import sys
import torch
import numpy as np
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.working_fractional_pino import WorkingFractionalPINO, MultiMethodWorkingFractionalPINO
from src.data.fractional_data_generator import generate_fractional_diffusion_data

class FractionalDiffusionEquationBenchmark:
    """Benchmark for fractional diffusion equation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def run_single_method_benchmark(self, alpha=0.5, method='caputo'):
        """Run benchmark for single fractional method"""
        print(f"Running fractional diffusion equation benchmark: α={alpha}, method={method}")
        
        # Configuration
        config = {
            'modes': 12,
            'width': 64,
            'epochs': 500,
            'learning_rate': 0.001,
            'physics_weight': 0.1
        }
        
        # Generate data
        train_data, test_data = generate_fractional_diffusion_data(
            alpha=alpha,
            domain_size=32,
            time_steps=50,
            batch_size=50
        )
        
        # Initialize model
        model = WorkingFractionalPINO(
            modes=config['modes'],
            width=config['width'],
            alpha=alpha,
            fractional_method=method
        ).to(self.device)
        
        # Training
        start_time = time.time()
        training_history = self.train_model(model, train_data, config)
        training_time = time.time() - start_time
        
        # Evaluation
        metrics = self.evaluate_model(model, test_data)
        metrics['training_time'] = training_time
        metrics['memory_usage'] = self.get_memory_usage()
        
        return {
            'alpha': alpha,
            'method': method,
            'config': config,
            'training_history': training_history,
            'metrics': metrics
        }
    
    def run_multi_method_benchmark(self, alpha=0.5, methods=['caputo', 'riemann_liouville', 'caputo_fabrizio']):
        """Run benchmark for multi-method approach"""
        print(f"Running multi-method benchmark: α={alpha}, methods={methods}")
        
        # Configuration
        config = {
            'modes': 12,
            'width': 64,
            'epochs': 500,
            'learning_rate': 0.001,
            'physics_weight': 0.1
        }
        
        # Generate data
        train_data, test_data = generate_fractional_diffusion_data(
            alpha=alpha,
            domain_size=32,
            time_steps=50,
            batch_size=50
        )
        
        # Initialize multi-method model
        model = MultiMethodWorkingFractionalPINO(
            modes=config['modes'],
            width=config['width'],
            alpha=alpha,
            methods=methods
        ).to(self.device)
        
        # Training
        start_time = time.time()
        training_history = self.train_model(model, train_data, config)
        training_time = time.time() - start_time
        
        # Evaluation
        metrics = self.evaluate_model(model, test_data)
        metrics['training_time'] = training_time
        metrics['memory_usage'] = self.get_memory_usage()
        
        return {
            'alpha': alpha,
            'methods': methods,
            'config': config,
            'training_history': training_history,
            'metrics': metrics
        }
    
    def run_alpha_sweep_benchmark(self, alphas=[0.1, 0.3, 0.5, 0.7, 0.9], method='caputo'):
        """Run benchmark across different alpha values"""
        print(f"Running alpha sweep benchmark: α={alphas}, method={method}")
        
        results = {}
        for alpha in alphas:
            try:
                result = self.run_single_method_benchmark(alpha=alpha, method=method)
                results[alpha] = result
                print(f"✅ Alpha {alpha}: L2 Error = {result['metrics']['l2_error']:.6f}")
            except Exception as e:
                print(f"❌ Alpha {alpha} failed: {e}")
                results[alpha] = None
        
        return results
    
    def run_method_comparison_benchmark(self, alpha=0.5, methods=['caputo', 'riemann_liouville', 'caputo_fabrizio', 'atangana_baleanu']):
        """Run benchmark comparing different fractional methods"""
        print(f"Running method comparison benchmark: α={alpha}, methods={methods}")
        
        results = {}
        for method in methods:
            try:
                result = self.run_single_method_benchmark(alpha=alpha, method=method)
                results[method] = result
                print(f"✅ Method {method}: L2 Error = {result['metrics']['l2_error']:.6f}")
            except Exception as e:
                print(f"❌ Method {method} failed: {e}")
                results[method] = None
        
        return results
    
    def train_model(self, model, train_data, config):
        """Train model with given configuration"""
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        
        training_history = {
            'loss': [],
            'l2_error': [],
            'learning_rate': []
        }
        
        x_train, y_train = train_data
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        
        for epoch in range(config['epochs']):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(x_train)
            
            # Compute loss
            data_loss = torch.nn.MSELoss()(y_pred, y_train)
            total_loss = data_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Record history
            training_history['loss'].append(total_loss.item())
            training_history['learning_rate'].append(scheduler.get_last_lr()[0])
            
            # Validation
            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(x_train)
                    l2_error = torch.norm(val_pred - y_train) / torch.norm(y_train)
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
            mse = torch.nn.MSELoss()(y_pred, y_test)
            mae = torch.nn.L1Loss()(y_pred, y_test)
            
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
        results_dir = Path("research/experiments/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / filename
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {results_file}")
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite"""
        print("Starting Fractional Diffusion Equation Comprehensive Benchmark")
        print("=" * 60)
        
        # 1. Single method benchmark
        print("\n1. Single Method Benchmark")
        single_method_results = self.run_single_method_benchmark(alpha=0.5, method='caputo')
        self.save_results(single_method_results, 'diffusion_equation_single_method.json')
        
        # 2. Multi-method benchmark
        print("\n2. Multi-Method Benchmark")
        multi_method_results = self.run_multi_method_benchmark(
            alpha=0.5, 
            methods=['caputo', 'riemann_liouville', 'caputo_fabrizio']
        )
        self.save_results(multi_method_results, 'diffusion_equation_multi_method.json')
        
        # 3. Alpha sweep benchmark
        print("\n3. Alpha Sweep Benchmark")
        alpha_sweep_results = self.run_alpha_sweep_benchmark(
            alphas=[0.1, 0.3, 0.5, 0.7, 0.9], 
            method='caputo'
        )
        self.save_results(alpha_sweep_results, 'diffusion_equation_alpha_sweep.json')
        
        # 4. Method comparison benchmark
        print("\n4. Method Comparison Benchmark")
        method_comparison_results = self.run_method_comparison_benchmark(
            alpha=0.5,
            methods=['caputo', 'riemann_liouville', 'caputo_fabrizio', 'atangana_baleanu']
        )
        self.save_results(method_comparison_results, 'diffusion_equation_method_comparison.json')
        
        print("\n" + "=" * 60)
        print("Fractional Diffusion Equation Benchmark Complete!")
        print("Results saved in research/experiments/results/")
        
        return {
            'single_method': single_method_results,
            'multi_method': multi_method_results,
            'alpha_sweep': alpha_sweep_results,
            'method_comparison': method_comparison_results
        }

def main():
    """Main execution function"""
    print("Fractional Diffusion Equation Benchmark")
    print("=" * 40)
    
    benchmark = FractionalDiffusionEquationBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\nBenchmark completed successfully!")
    print(f"Total experiments: {len(results)}")

if __name__ == "__main__":
    main()
