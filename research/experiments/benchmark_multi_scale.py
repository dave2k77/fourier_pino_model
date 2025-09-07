#!/usr/bin/env python3
"""
Multi-Scale Fractional PDE Benchmark
Specific experimental script for multi-scale fractional PDE validation
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
from src.data.fractional_data_generator import generate_multi_scale_data

class MultiScaleFractionalPDEBenchmark:
    """Benchmark for multi-scale fractional PDE"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def run_single_method_benchmark(self, alpha=0.5, method='caputo', scales=[1, 10, 100]):
        """Run benchmark for single fractional method"""
        print(f"Running multi-scale fractional PDE benchmark: α={alpha}, method={method}, scales={scales}")
        
        # Configuration
        config = {
            'modes': 12,
            'width': 64,
            'epochs': 500,
            'learning_rate': 0.001,
            'physics_weight': 0.1
        }
        
        # Generate data
        train_data, test_data = generate_multi_scale_data(
            alpha=alpha,
            domain_size=32,
            time_steps=50,
            batch_size=50,
            scales=scales
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
            'scales': scales,
            'config': config,
            'training_history': training_history,
            'metrics': metrics
        }
    
    def run_multi_method_benchmark(self, alpha=0.5, methods=['caputo', 'riemann_liouville', 'caputo_fabrizio'], scales=[1, 10, 100]):
        """Run benchmark for multi-method approach"""
        print(f"Running multi-method benchmark: α={alpha}, methods={methods}, scales={scales}")
        
        # Configuration
        config = {
            'modes': 12,
            'width': 64,
            'epochs': 500,
            'learning_rate': 0.001,
            'physics_weight': 0.1
        }
        
        # Generate data
        train_data, test_data = generate_multi_scale_data(
            alpha=alpha,
            domain_size=32,
            time_steps=50,
            batch_size=50,
            scales=scales
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
            'scales': scales,
            'config': config,
            'training_history': training_history,
            'metrics': metrics
        }
    
    def run_scale_comparison_benchmark(self, alpha=0.5, method='caputo', scale_combinations=[[1], [1, 10], [1, 10, 100], [1, 100, 1000]]):
        """Run benchmark comparing different scale combinations"""
        print(f"Running scale comparison benchmark: α={alpha}, method={method}")
        
        results = {}
        for scales in scale_combinations:
            try:
                result = self.run_single_method_benchmark(alpha=alpha, method=method, scales=scales)
                results[str(scales)] = result
                print(f"✅ Scales {scales}: L2 Error = {result['metrics']['l2_error']:.6f}")
            except Exception as e:
                print(f"❌ Scales {scales} failed: {e}")
                results[str(scales)] = None
        
        return results
    
    def run_alpha_sweep_benchmark(self, alphas=[0.1, 0.3, 0.5, 0.7, 0.9], method='caputo', scales=[1, 10, 100]):
        """Run benchmark across different alpha values"""
        print(f"Running alpha sweep benchmark: α={alphas}, method={method}, scales={scales}")
        
        results = {}
        for alpha in alphas:
            try:
                result = self.run_single_method_benchmark(alpha=alpha, method=method, scales=scales)
                results[alpha] = result
                print(f"✅ Alpha {alpha}: L2 Error = {result['metrics']['l2_error']:.6f}")
            except Exception as e:
                print(f"❌ Alpha {alpha} failed: {e}")
                results[alpha] = None
        
        return results
    
    def run_method_comparison_benchmark(self, alpha=0.5, methods=['caputo', 'riemann_liouville', 'caputo_fabrizio', 'atangana_baleanu'], scales=[1, 10, 100]):
        """Run benchmark comparing different fractional methods"""
        print(f"Running method comparison benchmark: α={alpha}, methods={methods}, scales={scales}")
        
        results = {}
        for method in methods:
            try:
                result = self.run_single_method_benchmark(alpha=alpha, method=method, scales=scales)
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
        print("Starting Multi-Scale Fractional PDE Comprehensive Benchmark")
        print("=" * 60)
        
        # 1. Single method benchmark
        print("\n1. Single Method Benchmark")
        single_method_results = self.run_single_method_benchmark(alpha=0.5, method='caputo', scales=[1, 10, 100])
        self.save_results(single_method_results, 'multi_scale_single_method.json')
        
        # 2. Multi-method benchmark
        print("\n2. Multi-Method Benchmark")
        multi_method_results = self.run_multi_method_benchmark(
            alpha=0.5, 
            methods=['caputo', 'riemann_liouville', 'caputo_fabrizio'],
            scales=[1, 10, 100]
        )
        self.save_results(multi_method_results, 'multi_scale_multi_method.json')
        
        # 3. Scale comparison benchmark
        print("\n3. Scale Comparison Benchmark")
        scale_comparison_results = self.run_scale_comparison_benchmark(
            alpha=0.5,
            method='caputo',
            scale_combinations=[[1], [1, 10], [1, 10, 100], [1, 100, 1000]]
        )
        self.save_results(scale_comparison_results, 'multi_scale_scale_comparison.json')
        
        # 4. Alpha sweep benchmark
        print("\n4. Alpha Sweep Benchmark")
        alpha_sweep_results = self.run_alpha_sweep_benchmark(
            alphas=[0.1, 0.3, 0.5, 0.7, 0.9], 
            method='caputo',
            scales=[1, 10, 100]
        )
        self.save_results(alpha_sweep_results, 'multi_scale_alpha_sweep.json')
        
        # 5. Method comparison benchmark
        print("\n5. Method Comparison Benchmark")
        method_comparison_results = self.run_method_comparison_benchmark(
            alpha=0.5,
            methods=['caputo', 'riemann_liouville', 'caputo_fabrizio', 'atangana_baleanu'],
            scales=[1, 10, 100]
        )
        self.save_results(method_comparison_results, 'multi_scale_method_comparison.json')
        
        print("\n" + "=" * 60)
        print("Multi-Scale Fractional PDE Benchmark Complete!")
        print("Results saved in research/experiments/results/")
        
        return {
            'single_method': single_method_results,
            'multi_method': multi_method_results,
            'scale_comparison': scale_comparison_results,
            'alpha_sweep': alpha_sweep_results,
            'method_comparison': method_comparison_results
        }

def main():
    """Main execution function"""
    print("Multi-Scale Fractional PDE Benchmark")
    print("=" * 40)
    
    benchmark = MultiScaleFractionalPDEBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\nBenchmark completed successfully!")
    print(f"Total experiments: {len(results)}")

if __name__ == "__main__":
    main()
