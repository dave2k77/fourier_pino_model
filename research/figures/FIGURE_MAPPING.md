# Figure Mapping - FractionalPINO Paper
## Clear Guide: Which Figure Goes Where in the Manuscript

**Updated**: January 2025  
**Status**: Cleaned and Organized  

---

## 🎯 **FIGURES ACTUALLY USED IN THE PAPER**

### **✅ FIGURES REFERENCED IN LATEX (4 figures)**

| Figure | File Name | Location in Paper | Content | Status |
|--------|-----------|-------------------|---------|---------|
| **Figure 1** | `figure_01_motivation.pdf` | Introduction | Research motivation and problem overview | ❌ **NEEDS CREATION** |
| **Figure 5** | `figure_05_architecture.pdf` | Methodology | FractionalPINO architecture overview | ❌ **NEEDS CREATION** |
| **Figure 11** | `figure_11_benchmark_solutions.pdf` | Results | Benchmark problem solutions | ❌ **NEEDS CREATION** |
| **Figure 12** | `figure_12_method_comparison.pdf` | Results | Method comparison results | ❌ **NEEDS CREATION** |

### **✅ ACTUAL GENERATED FIGURES (3 figures)**

| Figure | File Name | Location | Content | Status |
|--------|-----------|----------|---------|---------|
| **Alpha Sweep** | `alpha_sweep_plot.png` | Results Analysis | Performance across fractional orders α = 0.1 to 0.9 | ✅ **READY** |
| **Method Comparison** | `method_comparison_plot.png` | Results Analysis | Comparison of 4 fractional derivative methods | ✅ **READY** |
| **Training Curves** | `training_curves_plot.png` | Results Analysis | Training convergence curves for all methods | ✅ **READY** |

---

## 📋 **FIGURE REPLACEMENT STRATEGY**

### **Option 1: Use Generated Figures (RECOMMENDED)**
Replace the placeholder references with your actual generated figures:

```latex
% Replace Figure 11 (benchmark solutions) with:
\includegraphics[width=16cm]{figures/alpha_sweep_plot.png}
\includegraphics[width=16cm]{figures/method_comparison_plot.png}

% Replace Figure 12 (method comparison) with:
\includegraphics[width=14cm]{figures/training_curves_plot.png}
```

### **Option 2: Create Missing Figures**
Create the 4 missing figures:
1. `figure_01_motivation.pdf` - Research motivation diagram
2. `figure_05_architecture.pdf` - Architecture diagram
3. `figure_11_benchmark_solutions.pdf` - Benchmark solutions
4. `figure_12_method_comparison.pdf` - Method comparison

---

## 🗂️ **CLEAN FIGURE FOLDER STRUCTURE**

```
research/figures/
├── FIGURE_MAPPING.md                    # This file - clear mapping guide
├── FIGURE_PLACEHOLDERS.md              # Detailed specifications
├── README.md                           # Figure creation guide
├── 
├── # ACTUAL GENERATED FIGURES (READY TO USE)
├── alpha_sweep_plot.png                # ✅ Real data - Alpha sweep results
├── method_comparison_plot.png          # ✅ Real data - Method comparison
├── training_curves_plot.png            # ✅ Real data - Training curves
├── 
└── # PLACEHOLDER FILES (TO BE CREATED)
    ├── figure_01_motivation.pdf        # ❌ Needs creation
    ├── figure_05_architecture.pdf      # ❌ Needs creation
    ├── figure_11_benchmark_solutions.pdf # ❌ Needs creation
    └── figure_12_method_comparison.pdf # ❌ Needs creation
```

---

## 🚀 **IMMEDIATE ACTION PLAN**

### **For Overleaf Upload (RECOMMENDED APPROACH)**

**Step 1: Copy your generated figures**
```bash
# Copy from analysis folder to figures folder
cp /home/davianc/fourier_pino_model/research/experiments/analysis/alpha_sweep_plot.png /home/davianc/fourier_pino_model/research/figures/
cp /home/davianc/fourier_pino_model/research/experiments/analysis/method_comparison_plot.png /home/davianc/fourier_pino_model/research/figures/
cp /home/davianc/fourier_pino_model/research/experiments/analysis/training_curves_plot.png /home/davianc/fourier_pino_model/research/figures/
```

**Step 2: Update LaTeX references**
Replace the placeholder figure references with your actual figures:

```latex
% In the paper, replace:
\includegraphics[width=16cm]{figures/figure_11_benchmark_solutions.pdf}
% With:
\includegraphics[width=16cm]{figures/alpha_sweep_plot.png}

% And replace:
\includegraphics[width=14cm]{figures/figure_12_method_comparison.pdf}
% With:
\includegraphics[width=14cm]{figures/method_comparison_plot.png}
```

**Step 3: Create simple placeholders for Figures 1 and 5**
For now, create simple placeholder images or use text-based figures for:
- `figure_01_motivation.pdf` (Research motivation)
- `figure_05_architecture.pdf` (Architecture overview)

---

## 📊 **CURRENT FIGURE STATUS**

### **✅ READY FOR OVERLEAF (3 figures)**
- `alpha_sweep_plot.png` - Shows performance across fractional orders
- `method_comparison_plot.png` - Compares 4 fractional methods
- `training_curves_plot.png` - Training convergence curves

### **❌ NEEDS CREATION (4 figures)**
- `figure_01_motivation.pdf` - Research motivation diagram
- `figure_05_architecture.pdf` - Architecture diagram  
- `figure_11_benchmark_solutions.pdf` - Benchmark solutions (can use generated plots)
- `figure_12_method_comparison.pdf` - Method comparison (can use generated plots)

---

## 🎯 **RECOMMENDATION**

**For immediate Overleaf upload:**

1. **Use your 3 generated figures** (they contain real experimental data)
2. **Create simple placeholders** for Figures 1 and 5 (motivation and architecture)
3. **Update the LaTeX** to reference the correct figure files
4. **Upload to Overleaf** with the clean, organized structure

This gives you a complete paper with actual experimental results while keeping the process straightforward and manageable.

---

**The figure folder is now clean and organized with a clear mapping system!** 🎉
