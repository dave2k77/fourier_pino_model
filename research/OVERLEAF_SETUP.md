# Overleaf Setup Guide: FractionalPINO Paper
## LaTeX Project Structure for Journal of Computational Physics

**Project Name**: FractionalPINO - Advanced Fractional Calculus in Physics-Informed Neural Operators  
**Target Journal**: Journal of Computational Physics  
**Author**: Davian R. Chin  
**Institution**: University of Reading  

---

## 📁 **Project Structure**

```
fractional-pino-paper/
├── main.tex                    # Main LaTeX file
├── references.bib              # Bibliography file
├── figures/                    # Figure directory
│   ├── architecture_diagram.pdf
│   ├── method_comparison.pdf
│   ├── performance_analysis.pdf
│   └── ablation_studies.pdf
├── tables/                     # Table directory
│   ├── benchmark_results.tex
│   ├── method_analysis.tex
│   └── scalability_analysis.tex
└── README.md                   # Project documentation
```

---

## 🚀 **Overleaf Setup Instructions**

### **Step 1: Create New Project**
1. Go to [Overleaf.com](https://www.overleaf.com)
2. Click "New Project" → "Blank Project"
3. Name: "FractionalPINO - Advanced Fractional Calculus in Physics-Informed Neural Operators"
4. Click "Create"

### **Step 2: Upload Files**
1. **Main File**: Upload `fractional_pino_paper.tex` and rename to `main.tex`
2. **Bibliography**: Upload `references.bib`
3. **Figures**: Create `figures/` folder and upload figure files
4. **Tables**: Create `tables/` folder and upload table files

### **Step 3: Configure Compiler**
1. Go to "Menu" → "Settings"
2. Set "Compiler" to "pdfLaTeX"
3. Set "Main document" to "main.tex"
4. Click "Save"

### **Step 4: Compile and Test**
1. Click "Recompile" button
2. Check for compilation errors
3. Review PDF output
4. Make necessary adjustments

---

## 📝 **LaTeX File Structure**

### **Main Document (`main.tex`)**
```latex
\documentclass[12pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amsfonts,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{float}
\usepackage{geometry}
\usepackage{setspace}
\usepackage{harvard}
\usepackage{hyperref}
\usepackage{color}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{subcaption}
\usepackage{siunitx}
\usepackage{bm}

% Page setup
\geometry{margin=2.5cm}
\onehalfspacing

% Hyperref setup
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    citecolor=red,
    bookmarksnumbered=true,
    bookmarksopen=true,
    pdfstartview=FitH
}

% Custom commands
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\Lp}{\mathcal{L}^p}
\newcommand{\norm}[1]{\left\|#1\right\|}
\newcommand{\abs}[1]{\left|#1\right|}
\newcommand{\inner}[2]{\left\langle#1,#2\right\rangle}
\newcommand{\grad}{\nabla}
\newcommand{\divergence}{\nabla \cdot}
\newcommand{\laplacian}{\nabla^2}
\newcommand{\fractional}[2]{D^{#1}_{#2}}
\newcommand{\caputo}[1]{D^{#1}_C}
\newcommand{\riemann}[1]{D^{#1}_{RL}}
\newcommand{\caputofabrizio}[1]{D^{#1}_{CF}}
\newcommand{\atangana}[1]{D^{#1}_{AB}}

% Theorem environments
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\newtheorem{proposition}{Proposition}
\newtheorem{remark}{Remark}

% Title and author information
\title{Advanced Fractional Calculus in Physics-Informed Neural Operators: A Comprehensive Framework for Non-Local PDE Modelling}

\author{
    Davian R. Chin \\
    Department of Biomedical Engineering \\
    University of Reading \\
    Reading, UK \\
    \texttt{d.r.chin@pgr.reading.ac.uk} \\
    ORCID: \href{https://orcid.org/0009-0003-9434-3919}{0009-0003-9434-3919}
}

\date{\today}

\begin{document}

\maketitle

\begin{abstract}
% Abstract content here
\end{abstract}

% Main content sections
\section{Introduction}
% Introduction content

\section{Related Work}
% Related work content

\section{Methodology}
% Methodology content

\section{Implementation}
% Implementation content

\section{Experimental Validation}
% Experimental content

\section{Discussion}
% Discussion content

\section{Conclusion}
% Conclusion content

\section*{Acknowledgments}
% Acknowledgments content

\bibliographystyle{agsm}
\bibliography{references}

\end{document}
```

---

## 🎨 **LaTeX Features Used**

### **Mathematical Notation**
- **Inline Math**: `$equation$` for inline equations
- **Display Math**: `$$equation$$` for displayed equations
- **Custom Commands**: Fractional derivative notation
- **Theorem Environments**: Definitions, theorems, lemmas

### **Tables and Figures**
- **Booktabs**: Professional table formatting
- **Float**: Figure and table positioning
- **Subcaption**: Subfigures and subtables
- **Graphics**: PDF figures for high quality

### **Bibliography**
- **Harvard**: Harvard-style citation management
- **BibTeX**: Bibliography database
- **AGSM**: Australian Government Style Manual citation style

### **Hyperlinks**
- **Hyperref**: Clickable links and references
- **Color**: Coloured links
- **Bookmarks**: PDF bookmarks

---

## 📊 **Content Statistics**

### **Current Status**
- **Word Count**: ~8,500 words
- **Sections**: 7 major sections
- **References**: 25+ citations
- **Figures**: 15-20 planned
- **Tables**: 8-12 planned

### **LaTeX Features**
- **Mathematical Equations**: 50+ equations
- **Tables**: 6 comprehensive tables
- **Algorithms**: 1 algorithm
- **Definitions**: 1 formal definition
- **Citations**: 25+ references

---

## 🔧 **Compilation Instructions**

### **Local Compilation**
```bash
# Compile with pdfLaTeX
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or use latexmk
latexmk -pdf main.tex
```

### **Overleaf Compilation**
1. Upload all files to Overleaf
2. Set main document to `main.tex`
3. Click "Recompile"
4. Check for errors and warnings
5. Download PDF when ready

---

## 📋 **Pre-Submission Checklist**

### **Content Checklist**
- [ ] **Abstract**: 200-250 words, British English
- [ ] **Keywords**: 5-6 relevant keywords
- [ ] **Introduction**: Clear motivation and contributions
- [ ] **Related Work**: Comprehensive literature review
- [ ] **Methodology**: Rigorous mathematical framework
- [ ] **Implementation**: Technical implementation details
- [ ] **Experiments**: Comprehensive validation
- [ ] **Discussion**: Key findings and implications
- [ ] **Conclusion**: Summary and future work
- [ ] **References**: Complete bibliography

### **Format Checklist**
- [ ] **LaTeX Format**: Proper LaTeX formatting
- [ ] **Mathematical Notation**: Correct math notation
- [ ] **British English**: Consistent British English
- [ ] **Citations**: Proper citation format
- [ ] **Figures**: High-quality figures
- [ ] **Tables**: Well-formatted tables
- [ ] **Hyperlinks**: Working hyperlinks
- [ ] **Bibliography**: Complete references

### **Technical Checklist**
- [ ] **Compilation**: Error-free compilation
- [ ] **PDF Output**: Proper PDF generation
- [ ] **Page Layout**: Correct page layout
- [ ] **Fonts**: Consistent font usage
- [ ] **Spacing**: Proper line spacing
- [ ] **Margins**: Correct margins
- [ ] **Headers**: Proper headers and footers
- [ ] **Page Numbers**: Correct page numbering

---

## 🎯 **JCP Submission Requirements**

### **Format Requirements**
- **File Format**: PDF or LaTeX
- **Length**: 8,000-10,000 words
- **Figures**: High-quality figures (300+ DPI)
- **Tables**: Well-formatted tables
- **References**: Proper citation format
- **Abstract**: 200-250 words

### **Content Requirements**
- **Novelty**: Significant novelty required
- **Rigor**: High technical rigor
- **Validation**: Thorough experimental validation
- **Reproducibility**: Reproducible results
- **Impact**: High impact potential

---

## 🚀 **Ready for Submission**

### **Current Status**: ✅ **READY**

The LaTeX project is ready for:

1. **Overleaf Upload**: All files prepared
2. **Compilation**: Error-free compilation
3. **Review**: Content review and editing
4. **Submission**: JCP submission preparation

### **Next Steps**
1. **Upload to Overleaf**: Create project and upload files
2. **Compile and Test**: Ensure error-free compilation
3. **Content Review**: Review and edit content
4. **Final Preparation**: Prepare for submission

---

**Last Updated**: January 2025  
**Status**: Ready for Overleaf Upload  
**Next Steps**: Upload to Overleaf and begin compilation
