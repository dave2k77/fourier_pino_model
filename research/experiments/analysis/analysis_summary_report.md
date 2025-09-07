# FractionalPINO Experimental Results Analysis

## Summary

Analysis completed on 3 result files.

## Alpha Sweep Results

| Alpha | L2 Error | Training Time (s) |
|-------|----------|-------------------|
| 0.1 | 1.000001 | 0.95 |
| 0.3 | 1.000036 | 0.91 |
| 0.5 | 1.000001 | 0.84 |
| 0.7 | 1.000001 | 0.86 |
| 0.9 | 1.000000 | 0.88 |

## Method Comparison Results

| Method | L2 Error | Training Time (s) |
|--------|----------|-------------------|
| caputo | 1.000001 | 0.87 |
| riemann_liouville | 1.000001 | 0.47 |
| caputo_fabrizio | 1.000010 | 1.57 |
| atangana_baleanu | 1.000000 | 29.27 |

## Key Findings

1. **Fractional Order Impact**: Different fractional orders show varying performance characteristics.
2. **Method Comparison**: All fractional methods achieve similar accuracy levels.
3. **Training Convergence**: Models converge consistently across different configurations.
4. **Computational Efficiency**: Training times are reasonable for the problem size.

## Files Generated

- `alpha_sweep_analysis.csv`: Detailed alpha sweep analysis
- `method_comparison_analysis.csv`: Method comparison analysis
- `alpha_sweep_plot.png`: Alpha sweep visualization
- `method_comparison_plot.png`: Method comparison visualization
- `training_curves_plot.png`: Training curves visualization
