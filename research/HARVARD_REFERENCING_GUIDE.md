# Harvard Referencing Guide: FractionalPINO Paper
## LaTeX Harvard Citation Style Implementation

**Project**: FractionalPINO - Advanced Fractional Calculus in Physics-Informed Neural Operators  
**Citation Style**: Harvard (AGSM - Australian Government Style Manual)  
**LaTeX Package**: `harvard`  

---

## 📚 **Harvard Referencing Overview**

### **What is Harvard Referencing?**
Harvard referencing is an author-date citation style that uses in-text citations with author names and publication years, followed by a reference list at the end of the document.

### **Key Features:**
- **In-text citations**: Author(s) and year in parentheses
- **Reference list**: Alphabetical by author surname
- **No footnotes**: All citations in text
- **Consistent format**: Standardised citation format

---

## 🎯 **LaTeX Implementation**

### **Package Setup**
```latex
\usepackage{harvard}
\bibliographystyle{agsm}
```

### **Citation Commands**
- **`\citeasnoun{key}`**: Author (Year) - for narrative citations
- **`\cite{key}`**: (Author, Year) - for parenthetical citations
- **`\citeyear{key}`**: (Year) - for year-only citations
- **`\citeauthor{key}`**: Author - for author-only citations

---

## 📝 **Citation Examples**

### **Single Author**
```latex
% In text: Smith (2020) showed that...
\citeasnoun{smith2020} showed that...

% In text: ...as demonstrated (Smith, 2020).
...as demonstrated \cite{smith2020}.
```

### **Multiple Authors**
```latex
% In text: Smith and Jones (2020) found that...
\citeasnoun{smith2020} found that...

% In text: ...as shown (Smith & Jones, 2020).
...as shown \cite{smith2020}.
```

### **Multiple Citations**
```latex
% In text: Several studies (Smith, 2020; Jones, 2021; Brown, 2022) have shown...
Several studies \cite{smith2020,jones2021,brown2022} have shown...
```

### **Page Numbers**
```latex
% In text: Smith (2020, p. 15) stated that...
\citeasnoun[p. 15]{smith2020} stated that...

% In text: ...as shown (Smith, 2020, pp. 15-20).
...as shown \cite[p. 15-20]{smith2020}.
```

---

## 📋 **Reference List Format**

### **Journal Articles**
```bibtex
@article{smith2020,
  author = {Smith, John and Jones, Mary},
  title = {Advanced fractional calculus in neural networks},
  journal = {Journal of Computational Physics},
  volume = {400},
  pages = {123--145},
  year = {2020},
  publisher = {Elsevier}
}
```

### **Books**
```bibtex
@book{jones2021,
  author = {Jones, Mary},
  title = {Fractional Calculus: Theory and Applications},
  publisher = {Academic Press},
  year = {2021},
  address = {London}
}
```

### **Conference Papers**
```bibtex
@inproceedings{brown2022,
  author = {Brown, David},
  title = {Neural operators for fractional PDEs},
  booktitle = {Proceedings of the International Conference on Computational Physics},
  pages = {45--52},
  year = {2022},
  publisher = {IEEE}
}
```

### **Online Sources**
```bibtex
@misc{website2023,
  author = {Author, Name},
  title = {Title of Web Page},
  year = {2023},
  url = {https://example.com},
  note = {Accessed: 15 January 2025}
}
```

---

## 🎨 **Formatting Guidelines**

### **In-Text Citations**

#### **Narrative Citations (Author as part of sentence)**
- **Single author**: Smith (2020) demonstrated...
- **Two authors**: Smith and Jones (2020) found...
- **Three or more authors**: Smith et al. (2020) showed...

#### **Parenthetical Citations (Author in parentheses)**
- **Single author**: ...as shown (Smith, 2020).
- **Two authors**: ...as demonstrated (Smith & Jones, 2020).
- **Three or more authors**: ...as found (Smith et al., 2020).

### **Reference List Formatting**

#### **Author Names**
- **Format**: Surname, Initials
- **Example**: Smith, J. A.
- **Multiple authors**: Smith, J. A. & Jones, M. B.

#### **Titles**
- **Journal articles**: Title in sentence case
- **Books**: Title in Title Case
- **Conference papers**: Title in sentence case

#### **Punctuation**
- **Commas**: Separate authors, between elements
- **Periods**: End of each reference
- **Colons**: Before page numbers, after titles

---

## 📊 **Current Implementation Status**

### **✅ Completed**
- **Package Setup**: `harvard` package loaded
- **Citation Style**: AGSM style selected
- **Citation Commands**: `\citeasnoun` implemented
- **Reference Format**: BibTeX format ready

### **🔄 In Progress**
- **Citation Updates**: Converting all citations to Harvard style
- **Reference List**: Ensuring proper formatting
- **Consistency Check**: Verifying citation consistency

### **⏳ Pending**
- **Final Review**: Complete citation review
- **Format Check**: Final formatting verification
- **Compilation Test**: Testing with Overleaf

---

## 🔧 **LaTeX Commands Used**

### **Citation Commands**
```latex
% Narrative citation
\citeasnoun{key}          % Author (Year)

% Parenthetical citation  
\cite{key}                % (Author, Year)

% Year only
\citeyear{key}            % (Year)

% Author only
\citeauthor{key}          % Author

% With page numbers
\citeasnoun[p. 15]{key}   % Author (Year, p. 15)
\cite[p. 15]{key}         % (Author, Year, p. 15)
```

### **Bibliography Commands**
```latex
% Bibliography style
\bibliographystyle{agsm}

% Bibliography file
\bibliography{references}
```

---

## 📋 **Citation Checklist**

### **In-Text Citations**
- [ ] **Format**: Consistent Harvard style
- [ ] **Authors**: Proper author formatting
- [ ] **Years**: Correct publication years
- [ ] **Multiple citations**: Proper separation
- [ ] **Page numbers**: When applicable

### **Reference List**
- [ ] **Alphabetical order**: By author surname
- [ ] **Format consistency**: Uniform formatting
- [ ] **Complete information**: All required details
- [ ] **Punctuation**: Correct punctuation
- [ ] **Italics**: Proper italicisation

### **LaTeX Compilation**
- [ ] **Package loading**: `harvard` package
- [ ] **Style selection**: AGSM style
- [ ] **Compilation**: Error-free compilation
- [ ] **Output**: Proper citation formatting

---

## 🎯 **Benefits of Harvard Referencing**

### **Academic Advantages**
1. **Standardisation**: Widely recognised format
2. **Clarity**: Clear author-date system
3. **Flexibility**: Easy to modify and update
4. **Professional**: Academic standard format

### **Technical Advantages**
1. **LaTeX Support**: Excellent LaTeX integration
2. **Automation**: Automatic reference management
3. **Consistency**: Consistent formatting
4. **Maintenance**: Easy to maintain and update

---

## 🚀 **Ready for Implementation**

### **Current Status**: ✅ **IMPLEMENTED**

The Harvard referencing system is now implemented with:

1. **Package Setup**: `harvard` package loaded
2. **Citation Style**: AGSM style selected
3. **Citation Commands**: `\citeasnoun` implemented
4. **Reference Format**: BibTeX format ready

### **Next Steps**
1. **Upload to Overleaf**: Test compilation
2. **Citation Review**: Verify all citations
3. **Format Check**: Ensure proper formatting
4. **Final Compilation**: Complete compilation test

---

**Last Updated**: January 2025  
**Status**: Harvard Referencing Implemented  
**Next Steps**: Test compilation and final review
