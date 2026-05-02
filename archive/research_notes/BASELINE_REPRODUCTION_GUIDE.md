# PINO Project: Baseline Reproduction Guide

## Overview

This guide provides step-by-step instructions for reproducing the baseline results of the PINO (Physics-Informed Neural Operator) model implementation. This establishes a solid foundation before implementing the advanced features outlined in the project roadmap.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA-compatible GPU (recommended for faster training)
- 8GB+ RAM
- 10GB+ disk space for results

### Dependencies
All required dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Test the Setup
Before running the full baseline reproduction, test that everything works:

```bash
python scripts/test_baseline.py
```

This will run a quick test with:
- Data loading verification
- Model creation and forward pass
- Quick training experiment (5 epochs)

Expected output:
```
🧪 PINO Baseline Reproduction Test Suite
============================================================

==================== Data Loading ====================
✅ Dataset loaded successfully
📈 Train samples: [number]
📉 Test samples: [number]

==================== Model Creation ====================
✅ Model created successfully
📊 Model parameters: [number]

==================== Baseline Reproduction ====================
✅ Test experiment completed successfully!
📊 Final R² Score: [score]

============================================================
📋 TEST SUMMARY
============================================================
Data Loading              ✅ PASSED
Model Creation            ✅ PASSED
Baseline Reproduction     ✅ PASSED

🎉 ALL TESTS PASSED! Ready to run full baseline reproduction.
```

### 2. Run Full Baseline Reproduction

Once tests pass, run the complete baseline reproduction:

```bash
python scripts/reproduce_baseline.py
```

This will:
- Run all 6 experiments from the thesis (A1-A3, B1-B3)
- Generate comprehensive results and visualizations
- Create detailed reports and documentation

### 3. Run Specific Experiment (Optional)

To run a specific experiment:

```bash
python scripts/reproduce_baseline.py --experiment experiment_b2
```

Available experiments:
- `experiment_a1`: SGD, lr=0.001, physics_coeff=0.001
- `experiment_a2`: SGD, lr=0.005, physics_coeff=0.01
- `experiment_a3`: SGD, lr=0.01, physics_coeff=0.1
- `experiment_b1`: Adam, lr=0.001, physics_coeff=0.001
- `experiment_b2`: Adam, lr=0.005, physics_coeff=0.01
- `experiment_b3`: Adam, lr=0.01, physics_coeff=0.1

## Expected Results

### Performance Metrics
Based on the thesis findings, you should expect:

| Experiment | Optimizer | Learning Rate | Physics Coeff | Expected R² Score |
|------------|-----------|---------------|---------------|-------------------|
| A1         | SGD       | 0.001         | 0.001         | ~0.92             |
| A2         | SGD       | 0.005         | 0.01          | ~0.94             |
| A3         | SGD       | 0.01          | 0.1           | ~0.91             |
| B1         | Adam      | 0.001         | 0.001         | ~0.93             |
| B2         | Adam      | 0.005         | 0.01          | ~0.96             |
| B3         | Adam      | 0.01          | 0.1           | ~0.94             |

### Key Findings to Validate
1. **Best Performance**: Experiment B2 (Adam, lr=0.005, physics_coeff=0.01) should achieve the highest R² score
2. **Physics Loss Coefficient**: Optimal range should be 0.001-0.01
3. **Optimizer Comparison**: Adam should generally outperform SGD
4. **Convergence**: All experiments should converge within 100 epochs

## Output Structure

After running the baseline reproduction, you'll find the following structure:

```
baseline_results/
├── models/                          # Saved model checkpoints
│   ├── experiment_a1_model.pth
│   ├── experiment_a2_model.pth
│   ├── experiment_a3_model.pth
│   ├── experiment_b1_model.pth
│   ├── experiment_b2_model.pth
│   └── experiment_b3_model.pth
├── plots/                           # Training visualizations
│   ├── experiment_a1_loss.png
│   ├── experiment_a2_loss.png
│   ├── experiment_a3_loss.png
│   ├── experiment_b1_loss.png
│   ├── experiment_b2_loss.png
│   ├── experiment_b3_loss.png
│   └── baseline_summary.png
├── data/                            # Results data
│   ├── baseline_results.json        # Complete results
│   └── baseline_summary.csv         # Summary table
├── logs/                            # Detailed logs
│   └── reproduction.log
└── BASELINE_REPORT.md               # Comprehensive report
```

## Understanding the Results

### 1. Training Loss Curves
Each experiment generates a loss plot showing:
- Training loss over epochs
- Test loss over epochs
- Convergence behavior
- Final loss values

### 2. Summary Visualizations
The `baseline_summary.png` provides:
- R² score comparison across experiments
- Training time analysis
- Loss comparison (train vs test)
- Physics loss coefficient impact

### 3. Detailed Reports
The `BASELINE_REPORT.md` contains:
- Complete experimental results
- Performance analysis
- Key findings and conclusions
- Next steps for improvement

## Troubleshooting

### Common Issues

#### 1. Data Loading Errors
**Error**: `FileNotFoundError: Heatmap folder not found`
**Solution**: Ensure data files are in the correct locations:
```
images/
├── heatmaps/           # PNG heatmap images
└── pde_solutions/      # NPZ solution files
```

#### 2. CUDA Memory Issues
**Error**: `RuntimeError: CUDA out of memory`
**Solution**: Reduce batch size in the configuration:
```python
# In config.py or script arguments
batch_size=16  # Reduce from 32
```

#### 3. Training Convergence Issues
**Issue**: Loss not decreasing or R² scores too low
**Solutions**:
- Check data quality and preprocessing
- Verify physics loss coefficient range
- Ensure proper random seed setting
- Check learning rate values

### Performance Optimization

#### For Faster Training
1. **Use GPU**: Ensure CUDA is available and being used
2. **Reduce Epochs**: For testing, use fewer epochs (e.g., 50 instead of 100)
3. **Increase Batch Size**: If memory allows, larger batch sizes can speed up training

#### For Memory Optimization
1. **Reduce Batch Size**: Start with 16 or 8 if memory is limited
2. **Use Mixed Precision**: Enable FP16 training (future enhancement)
3. **Gradient Checkpointing**: For very large models (future enhancement)

## Validation Checklist

Before proceeding to the next phase, ensure:

- [ ] All 6 experiments complete successfully
- [ ] R² scores match expected ranges (0.90-0.97)
- [ ] Best performance achieved with Experiment B2
- [ ] Training loss curves show proper convergence
- [ ] No errors in logs
- [ ] All output files generated correctly
- [ ] Results are reproducible (same seed produces same results)

## Next Steps

Once baseline reproduction is complete and validated:

### Phase 1: Foundation Enhancement
1. **Experiment Tracking**: Integrate Weights & Biases for experiment tracking
2. **Configuration Management**: Implement Hydra for advanced configuration
3. **Performance Optimization**: Add mixed precision training and memory optimization

### Phase 2: Multi-PDE Extension
1. **Universal PINO Framework**: Extend beyond heat equation
2. **Advanced Physics Loss**: Implement Lie symmetry and variational formulations
3. **Multiple PDE Types**: Wave, Burgers, Navier-Stokes equations

### Phase 3: Technical Innovation
1. **Fractional Calculus**: Implement memory-dependent phenomena modeling
2. **Layered Fourier Reduction**: Memory optimization (28.6%-69.3% reduction)
3. **Transformer Integration**: Physics-informed attention mechanisms

## Contributing

When making changes to the baseline reproduction:

1. **Test First**: Always run `test_baseline.py` before making changes
2. **Document Changes**: Update this guide and related documentation
3. **Maintain Reproducibility**: Ensure deterministic training is preserved
4. **Version Control**: Commit changes with clear messages

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the logs in `baseline_results/logs/`
3. Verify system requirements and dependencies
4. Consult the main project documentation

## References

- Original thesis: "BALANCING DATA FITTING AND PHYSICAL PROPERTIES A COMPARATIVE STUDY ON PHYSICS LOSS COEFFICIENTS AND FOURIER ANALYSIS TECHNIQUES IN PINO MODELS FOR PDES"
- Project roadmap: `PROJECT_ROADMAP.md`
- Main documentation: `README.md`

---

**Note**: This baseline reproduction establishes the foundation for all future improvements. Ensure all tests pass and results are validated before proceeding to advanced features.
