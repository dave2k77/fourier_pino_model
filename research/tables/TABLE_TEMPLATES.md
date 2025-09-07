# Table Templates: FractionalPINO Paper
## Experimental Results Tables for Journal Submission

**Total Tables**: 8-12 tables planned  
**Format**: LaTeX table format with booktabs  
**Style**: Professional academic tables  

---

## 📊 **Table Allocation by Section**

### **Section 3: Methodology (2 tables)**
- **Table 1**: Fractional derivative method specifications
- **Table 2**: Architecture component comparison

### **Section 4: Implementation (1 table)**
- **Table 3**: Computational complexity analysis

### **Section 5: Experimental Validation (6 tables)**
- **Table 4**: Fractional heat equation results
- **Table 5**: Fractional wave equation results
- **Table 6**: Fractional diffusion equation results
- **Table 7**: Fractional method performance analysis
- **Table 8**: Multi-method fusion analysis
- **Table 9**: Scalability analysis

### **Section 6: Discussion (2 tables)**
- **Table 10**: Ablation study results
- **Table 11**: Performance summary

---

## 📋 **Detailed Table Specifications**

### **Table 1: Fractional Derivative Method Specifications**
**File**: `tables/table_01_method_specifications.tex`
**Content**: Method properties, kernel functions, numerical properties
**Size**: 4 columns × 8 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Fractional Derivative Method Specifications}
\label{tab:method_specifications}
\begin{tabular}{@{}lccc@{}}
\toprule
Method & Kernel Function & Numerical Stability & Computational Cost \\
\midrule
Caputo & $\frac{1}{\Gamma(1-\alpha)} (x-y)^{-\alpha}$ & Good & $\mathcal{O}(N^2)$ \\
Riemann-Liouville & $\frac{1}{\Gamma(1-\alpha)} (x-y)^{-\alpha}$ & Good & $\mathcal{O}(N^2)$ \\
Caputo-Fabrizio & $\frac{M(\alpha)}{1-\alpha} \exp\left(-\frac{\alpha(x-y)}{1-\alpha}\right)$ & Excellent & $\mathcal{O}(N \log N)$ \\
Atangana-Baleanu & $\frac{AB(\alpha)}{1-\alpha} E_\alpha\left(-\frac{\alpha(x-y)^\alpha}{1-\alpha}\right)$ & Excellent & $\mathcal{O}(N \log N)$ \\
Weyl & $\frac{1}{\Gamma(-\alpha)} \int_x^\infty \frac{f(t)}{(t-x)^{\alpha+1}} dt$ & Good & $\mathcal{O}(N^2)$ \\
Marchaud & $\frac{\alpha}{\Gamma(1-\alpha)} \int_0^\infty \frac{f(x) - f(x-t)}{t^{\alpha+1}} dt$ & Good & $\mathcal{O}(N^2)$ \\
Hadamard & $\frac{1}{\Gamma(n-\alpha)} \left(x \frac{d}{dx}\right)^n \int_a^x \frac{f(t)}{(\ln(x/t))^{\alpha-n+1}} \frac{dt}{t}$ & Good & $\mathcal{O}(N^2)$ \\
Reiz-Feller & $\frac{1}{\Gamma(1-\alpha)} \int_0^\infty \frac{f(x-t) - f(x+t)}{t^\alpha} dt$ & Good & $\mathcal{O}(N^2)$ \\
\bottomrule
\end{tabular}
\end{table}
```

### **Table 2: Architecture Component Comparison**
**File**: `tables/table_02_architecture_components.tex`
**Content**: Component specifications, parameters, functionality
**Size**: 5 columns × 4 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Architecture Component Comparison}
\label{tab:architecture_components}
\begin{tabular}{@{}lcccc@{}}
\toprule
Component & Input Size & Output Size & Parameters & Functionality \\
\midrule
Fractional Encoder & $N \times N$ & $N \times N \times M$ & $M \times 8$ & HPFRACC integration \\
Neural Operator & $N \times N \times M$ & $N \times N \times M$ & $M \times 64$ & Function space mapping \\
Fusion Layer & $N \times N \times M$ & $N \times N$ & $M \times 1$ & Multi-method combination \\
Physics Loss Module & $N \times N$ & Scalar & $4 \times 1$ & Constraint computation \\
\bottomrule
\end{tabular}
\end{table}
```

### **Table 3: Computational Complexity Analysis**
**File**: `tables/table_03_complexity_analysis.tex`
**Content**: Time complexity, space complexity, scalability
**Size**: 4 columns × 6 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Computational Complexity Analysis}
\label{tab:complexity_analysis}
\begin{tabular}{@{}lccc@{}}
\toprule
Operation & Time Complexity & Space Complexity & Scalability \\
\midrule
Fractional Encoding & $\mathcal{O}(N^2)$ & $\mathcal{O}(N^2)$ & Linear \\
Neural Operator & $\mathcal{O}(N \log N)$ & $\mathcal{O}(N)$ & Excellent \\
Fusion Layer & $\mathcal{O}(N^2)$ & $\mathcal{O}(N^2)$ & Good \\
Physics Loss & $\mathcal{O}(N^2)$ & $\mathcal{O}(N^2)$ & Good \\
Total Forward Pass & $\mathcal{O}(N^2)$ & $\mathcal{O}(N^2)$ & Good \\
Total Training & $\mathcal{O}(N^2 \log N)$ & $\mathcal{O}(N^2)$ & Good \\
\bottomrule
\end{tabular}
\end{table}
```

### **Table 4: Fractional Heat Equation Results**
**File**: `tables/table_04_heat_equation_results.tex`
**Content**: L2 error, L∞ error, training time, memory usage
**Size**: 5 columns × 5 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Fractional Heat Equation Results}
\label{tab:heat_equation_results}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & L2 Error & L∞ Error & Training Time (s) & Memory (GB) \\
\midrule
Traditional PINN & $2.3 \times 10^{-2}$ & $4.1 \times 10^{-2}$ & 1,200 & 2.1 \\
FNO & $1.8 \times 10^{-2}$ & $3.2 \times 10^{-2}$ & 800 & 1.8 \\
PINO & $1.5 \times 10^{-2}$ & $2.8 \times 10^{-2}$ & 900 & 1.9 \\
fPINN & $1.2 \times 10^{-2}$ & $2.1 \times 10^{-2}$ & 1,500 & 2.3 \\
\textbf{FractionalPINO} & \textbf{$6.8 \times 10^{-3}$} & \textbf{$1.2 \times 10^{-2}$} & \textbf{750} & \textbf{1.6} \\
\bottomrule
\end{tabular}
\end{table}
```

### **Table 5: Fractional Wave Equation Results**
**File**: `tables/table_05_wave_equation_results.tex`
**Content**: L2 error, L∞ error, training time, memory usage
**Size**: 5 columns × 5 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Fractional Wave Equation Results}
\label{tab:wave_equation_results}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & L2 Error & L∞ Error & Training Time (s) & Memory (GB) \\
\midrule
Traditional PINN & $3.1 \times 10^{-2}$ & $5.2 \times 10^{-2}$ & 1,400 & 2.2 \\
FNO & $2.4 \times 10^{-2}$ & $4.1 \times 10^{-2}$ & 900 & 1.9 \\
PINO & $2.1 \times 10^{-2}$ & $3.6 \times 10^{-2}$ & 1,000 & 2.0 \\
fPINN & $1.8 \times 10^{-2}$ & $3.1 \times 10^{-2}$ & 1,600 & 2.4 \\
\textbf{FractionalPINO} & \textbf{$8.9 \times 10^{-3}$} & \textbf{$1.5 \times 10^{-2}$} & \textbf{850} & \textbf{1.7} \\
\bottomrule
\end{tabular}
\end{table}
```

### **Table 6: Fractional Diffusion Equation Results**
**File**: `tables/table_06_diffusion_equation_results.tex`
**Content**: L2 error, L∞ error, training time, memory usage
**Size**: 5 columns × 5 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Fractional Diffusion Equation Results}
\label{tab:diffusion_equation_results}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & L2 Error & L∞ Error & Training Time (s) & Memory (GB) \\
\midrule
Traditional PINN & $2.8 \times 10^{-2}$ & $4.8 \times 10^{-2}$ & 1,300 & 2.0 \\
FNO & $2.1 \times 10^{-2}$ & $3.8 \times 10^{-2}$ & 850 & 1.7 \\
PINO & $1.8 \times 10^{-2}$ & $3.2 \times 10^{-2}$ & 950 & 1.8 \\
fPINN & $1.5 \times 10^{-2}$ & $2.7 \times 10^{-2}$ & 1,400 & 2.2 \\
\textbf{FractionalPINO} & \textbf{$7.2 \times 10^{-3}$} & \textbf{$1.3 \times 10^{-2}$} & \textbf{800} & \textbf{1.5} \\
\bottomrule
\end{tabular}
\end{table}
```

### **Table 7: Fractional Method Performance Analysis**
**File**: `tables/table_07_fractional_method_performance.tex`
**Content**: Method comparison, L2 error, training time, stability
**Size**: 4 columns × 8 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Fractional Method Performance Analysis}
\label{tab:fractional_method_performance}
\begin{tabular}{@{}lccc@{}}
\toprule
Method & L2 Error & Training Time (s) & Numerical Stability \\
\midrule
Caputo & $7.2 \times 10^{-3}$ & 720 & Good \\
Riemann-Liouville & $6.9 \times 10^{-3}$ & 740 & Good \\
Caputo-Fabrizio & $6.5 \times 10^{-3}$ & 680 & Excellent \\
Atangana-Baleanu & $6.3 \times 10^{-3}$ & 690 & Excellent \\
Weyl & $7.1 \times 10^{-3}$ & 750 & Good \\
Marchaud & $6.8 \times 10^{-3}$ & 730 & Good \\
Hadamard & $7.0 \times 10^{-3}$ & 760 & Good \\
Reiz-Feller & $6.7 \times 10^{-3}$ & 720 & Good \\
\bottomrule
\end{tabular}
\end{table}
```

### **Table 8: Multi-Method Fusion Analysis**
**File**: `tables/table_08_fusion_analysis.tex`
**Content**: Fusion strategy, L2 error, training time, memory
**Size**: 4 columns × 4 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Multi-Method Fusion Analysis}
\label{tab:fusion_analysis}
\begin{tabular}{@{}lccc@{}}
\toprule
Fusion Strategy & L2 Error & Training Time (s) & Memory (GB) \\
\midrule
Single Method (Caputo) & $7.2 \times 10^{-3}$ & 720 & 1.6 \\
Weighted Combination & $6.1 \times 10^{-3}$ & 850 & 1.8 \\
Attention-Based & $5.8 \times 10^{-3}$ & 900 & 1.9 \\
Hierarchical & $5.5 \times 10^{-3}$ & 950 & 2.0 \\
\bottomrule
\end{tabular}
\end{table}
```

### **Table 9: Scalability Analysis**
**File**: `tables/table_09_scalability_analysis.tex`
**Content**: Resolution, L2 error, training time, memory
**Size**: 4 columns × 4 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Scalability Analysis}
\label{tab:scalability_analysis}
\begin{tabular}{@{}cccc@{}}
\toprule
Resolution & L2 Error & Training Time (s) & Memory (GB) \\
\midrule
$32 \times 32$ & $6.8 \times 10^{-3}$ & 450 & 1.2 \\
$64 \times 64$ & $6.9 \times 10^{-3}$ & 750 & 1.6 \\
$128 \times 128$ & $7.1 \times 10^{-3}$ & 1,200 & 2.1 \\
$256 \times 256$ & $7.3 \times 10^{-3}$ & 2,100 & 3.2 \\
\bottomrule
\end{tabular}
\end{table}
```

### **Table 10: Ablation Study Results**
**File**: `tables/table_10_ablation_study.tex`
**Content**: Configuration, L2 error, training time, memory
**Size**: 4 columns × 4 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Ablation Study Results}
\label{tab:ablation_study}
\begin{tabular}{@{}lccc@{}}
\toprule
Configuration & L2 Error & Training Time (s) & Memory (GB) \\
\midrule
Full FractionalPINO & $6.8 \times 10^{-3}$ & 750 & 1.6 \\
Without Fusion & $7.5 \times 10^{-3}$ & 720 & 1.5 \\
Without Spectral Processing & $8.2 \times 10^{-3}$ & 800 & 1.7 \\
Without HPFRACC & $9.1 \times 10^{-3}$ & 1,100 & 2.0 \\
\bottomrule
\end{tabular}
\end{table}
```

### **Table 11: Performance Summary**
**File**: `tables/table_11_performance_summary.tex`
**Content**: Metric, improvement, baseline, FractionalPINO
**Size**: 4 columns × 5 rows

**LaTeX Code**:
```latex
\begin{table}[H]
\centering
\caption{Performance Summary}
\label{tab:performance_summary}
\begin{tabular}{@{}lccc@{}}
\toprule
Metric & Improvement & Baseline & FractionalPINO \\
\midrule
L2 Error & 20--50\% & $2.3 \times 10^{-2}$ & $6.8 \times 10^{-3}$ \\
Training Time & 2--3x & 1,200s & 750s \\
Memory Usage & 1.3x & 2.1GB & 1.6GB \\
GPU Utilisation & 1.2x & 85\% & 95\% \\
Scalability & 1.5x & Linear & Near-linear \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 📁 **Table File Structure**

```
tables/
├── table_01_method_specifications.tex
├── table_02_architecture_components.tex
├── table_03_complexity_analysis.tex
├── table_04_heat_equation_results.tex
├── table_05_wave_equation_results.tex
├── table_06_diffusion_equation_results.tex
├── table_07_fractional_method_performance.tex
├── table_08_fusion_analysis.tex
├── table_09_scalability_analysis.tex
├── table_10_ablation_study.tex
├── table_11_performance_summary.tex
└── TABLE_TEMPLATES.md
```

---

## 🎨 **Table Formatting Guidelines**

### **LaTeX Packages Required**
```latex
\usepackage{booktabs}    % Professional table formatting
\usepackage{array}       % Enhanced array functionality
\usepackage{multirow}    % Multi-row cells
\usepackage{float}       % Table positioning
\usepackage{siunitx}     % Scientific notation
```

### **Table Style Guidelines**
- **Top Rule**: `\toprule` for table header
- **Mid Rule**: `\midrule` for section separators
- **Bottom Rule**: `\bottomrule` for table footer
- **No Vertical Lines**: Clean, professional appearance
- **Consistent Spacing**: Proper column spacing
- **Bold Headers**: Emphasise important results

### **Number Formatting**
- **Scientific Notation**: Use `\times 10^{-3}` format
- **Decimals**: Consistent decimal places
- **Units**: Include units in headers
- **Bold Results**: Highlight FractionalPINO results

---

## 📋 **Table Checklist**

### **Content Checklist**
- [ ] **Accuracy**: Data accurately represented
- [ ] **Completeness**: All necessary information included
- [ ] **Consistency**: Consistent formatting across tables
- [ ] **Clarity**: Clear and unambiguous presentation
- [ ] **Professional**: Academic publication quality

### **Format Checklist**
- [ ] **LaTeX Format**: Proper LaTeX table formatting
- [ ] **Booktabs**: Professional table styling
- [ ] **Alignment**: Proper column alignment
- [ ] **Spacing**: Consistent spacing and padding
- [ ] **Captions**: Descriptive table captions

### **Integration Checklist**
- [ ] **Placement**: Appropriate table placement
- [ ] **References**: Proper cross-referencing
- [ ] **Labels**: Clear table labels
- [ ] **Compilation**: Error-free compilation
- [ ] **Consistency**: Consistent with figure style

---

## 🚀 **Implementation Status**

### **✅ Completed**
- **Table Specifications**: All 11 tables specified
- **LaTeX Code**: Complete LaTeX table code
- **File Structure**: Organised table file structure
- **Formatting Guidelines**: Professional formatting guidelines

### **🔄 Next Steps**
- **Data Collection**: Collect experimental data
- **Table Population**: Fill tables with actual results
- **Formatting**: Apply consistent formatting
- **Integration**: Integrate into LaTeX document

---

**Last Updated**: January 2025  
**Status**: Table Templates Complete  
**Next Steps**: Collect experimental data and populate tables
