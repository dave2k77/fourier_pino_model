#!/usr/bin/env python3
"""
Master Benchmark Runner
Run all benchmark experiments for FractionalPINO paper
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import benchmark scripts
from benchmark_heat_equation import FractionalHeatEquationBenchmark
from benchmark_wave_equation import FractionalWaveEquationBenchmark
from benchmark_diffusion_equation import FractionalDiffusionEquationBenchmark
from benchmark_multi_scale import MultiScaleFractionalPDEBenchmark

class MasterBenchmarkRunner:
    """Master runner for all benchmark experiments"""
    
    def __init__(self):
        self.start_time = time.time()
        self.results = {}
        self.results_dir = Path("research/experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_heat_equation_benchmark(self):
        """Run fractional heat equation benchmark"""
        print("\n" + "="*80)
        print("FRACTIONAL HEAT EQUATION BENCHMARK")
        print("="*80)
        
        benchmark = FractionalHeatEquationBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        self.results['heat_equation'] = results
        return results
    
    def run_wave_equation_benchmark(self):
        """Run fractional wave equation benchmark"""
        print("\n" + "="*80)
        print("FRACTIONAL WAVE EQUATION BENCHMARK")
        print("="*80)
        
        benchmark = FractionalWaveEquationBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        self.results['wave_equation'] = results
        return results
    
    def run_diffusion_equation_benchmark(self):
        """Run fractional diffusion equation benchmark"""
        print("\n" + "="*80)
        print("FRACTIONAL DIFFUSION EQUATION BENCHMARK")
        print("="*80)
        
        benchmark = FractionalDiffusionEquationBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        self.results['diffusion_equation'] = results
        return results
    
    def run_multi_scale_benchmark(self):
        """Run multi-scale fractional PDE benchmark"""
        print("\n" + "="*80)
        print("MULTI-SCALE FRACTIONAL PDE BENCHMARK")
        print("="*80)
        
        benchmark = MultiScaleFractionalPDEBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        self.results['multi_scale'] = results
        return results
    
    def save_master_results(self):
        """Save all results to master file"""
        master_results = {
            'timestamp': datetime.now().isoformat(),
            'total_runtime': time.time() - self.start_time,
            'results': self.results
        }
        
        master_file = self.results_dir / 'master_benchmark_results.json'
        with open(master_file, 'w') as f:
            json.dump(master_results, f, indent=2, default=str)
        
        print(f"\nMaster results saved to: {master_file}")
        return master_file
    
    def generate_summary_report(self):
        """Generate summary report of all benchmarks"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY REPORT")
        print("="*80)
        
        total_experiments = 0
        successful_experiments = 0
        
        for benchmark_name, benchmark_results in self.results.items():
            print(f"\n{benchmark_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            for result_type, result_data in benchmark_results.items():
                if result_data is not None:
                    if isinstance(result_data, dict):
                        if 'metrics' in result_data:
                            l2_error = result_data['metrics'].get('l2_error', 'N/A')
                            training_time = result_data['metrics'].get('training_time', 'N/A')
                            print(f"  {result_type}: L2 Error = {l2_error:.6f}, Time = {training_time:.2f}s")
                            successful_experiments += 1
                        else:
                            # Handle nested results (like alpha_sweep, method_comparison)
                            for key, value in result_data.items():
                                if value is not None and isinstance(value, dict) and 'metrics' in value:
                                    l2_error = value['metrics'].get('l2_error', 'N/A')
                                    print(f"    {key}: L2 Error = {l2_error:.6f}")
                                    successful_experiments += 1
                    total_experiments += 1
        
        print(f"\nSUMMARY:")
        print(f"  Total Experiments: {total_experiments}")
        print(f"  Successful: {successful_experiments}")
        print(f"  Failed: {total_experiments - successful_experiments}")
        print(f"  Success Rate: {successful_experiments/total_experiments*100:.1f}%")
        print(f"  Total Runtime: {time.time() - self.start_time:.2f} seconds")
    
    def run_all_benchmarks(self):
        """Run all benchmark experiments"""
        print("FRACTIONALPINO COMPREHENSIVE BENCHMARK SUITE")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results directory: {self.results_dir}")
        
        try:
            # Run all benchmarks
            self.run_heat_equation_benchmark()
            self.run_wave_equation_benchmark()
            self.run_diffusion_equation_benchmark()
            self.run_multi_scale_benchmark()
            
            # Save results
            self.save_master_results()
            
            # Generate summary
            self.generate_summary_report()
            
            print("\n" + "="*80)
            print("ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Benchmark suite failed: {e}")
            return False

def main():
    """Main execution function"""
    print("FractionalPINO Master Benchmark Runner")
    print("=" * 50)
    
    runner = MasterBenchmarkRunner()
    success = runner.run_all_benchmarks()
    
    if success:
        print("\n✅ All benchmarks completed successfully!")
        print("📊 Results available in research/experiments/results/")
        print("📈 Ready for paper data analysis and visualization")
    else:
        print("\n❌ Some benchmarks failed. Check logs for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
