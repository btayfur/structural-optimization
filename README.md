# Structural Optimization Techniques (Yapısal Optimizasyon Teknikleri)

This repository contains course materials for "Structural Optimization Techniques" (Yapısal Optimizasyon Teknikleri), an introduction course on optimization methods in structural engineering. AI is used to create some sections of the course materials, so still some improvements are needed. If you want to contribute, its open for any suggestion or commits.

## Download

### English Version
You can download the English version of the course notes directly from this repository:
- [Download English Course Notes (PDF)](https://github.com/btayfur/structural-optimization/blob/main/EN-en/main.pdf)

### Türkçe Versiyon
Dersin Türkçe ders notları bu deponun TR-tr dizininde mevcuttur. Örnekler ve birçok alt klasörde de ayrıca Türkçe readme-tr dosyaları bulunmaktadır.
- [Türkçe Ders Notlarını İndir (PDF)](https://github.com/btayfur/structural-optimization/blob/main/TR-tr/main.pdf)

## Overview

This collection is a complete set of lecture notes covering basic theoretical foundations and practical applications of structural optimization. The course is structured in a 14-week format, each week covering a specific topic in the field of optimization or structural optimization.

## Contents

The repository includes:

- **main.tex**: The main LaTeX file that compiles all weekly content
- **main.pdf**: The compiled PDF containing the complete course notes
- **Stil.sty**: Custom LaTeX style file for document formatting
- **weeks_new/**: Directory containing individual LaTeX files for each week's content:
1. Introduction to Optimization Theory
2. Fundamental Optimization Concepts
3. Theoretical Foundations of Classical Optimization Algorithms
4. Benchmark Test Functions
5. Metaheuristic Optimization Algorithms I
6. Metaheuristic Optimization Algorithms II
7. Optimization of Discrete Parameters
8. Optimization of Continuous Parameters
9. Introduction to Structural Optimization
10. Topological Optimization
11. Size and Shape Optimization
12. Multi-Objective Optimization
13. Application I (10 Parameter Cantilever Beam's optimization with SA)
14. Application II (2D 4-Bay 8-Storey Steel Structure's optimization with SA)

## How to Use This Repository

### Example Codes
The repository contains example code implementations in the `Code/Examples` directory. These examples demonstrate practical applications of optimization techniques covered in the course:

Each example includes source code, documentation, and sample results to help students understand the practical implementation of theoretical concepts. You can scan the QR codes (or just click) in the lecture notes to access specific examples directly from GitHub.


### Prerequisites to Edit Lecture Notes

To compile the LaTeX documents, you need:
- A LaTeX distribution (e.g., TeX Live, MiKTeX)
- A LaTeX editor (optional, but recommended)

### Compiling the Document

To generate the PDF from source:

1. Make sure you have a LaTeX distribution installed on your system
2. Compile the main.tex file with pdfLaTeX:
   ```
   pdflatex main.tex
   ```
3. Run the compilation multiple times to ensure proper cross-references and table of contents:
   ```
   pdflatex main.tex
   pdflatex main.tex
   ```

Alternatively, you can directly use the pre-compiled `main.pdf` file.

## Recommended Resources

Additional resources recommended for those who want to deepen their understanding of optimization:

### Books

1. Cottle, R. W., & Thapa, M. N. (2017). **Linear and Nonlinear Optimization**. Springer.
   - Provides a comprehensive introduction to linear and nonlinear optimization. An excellent reference for theoretical foundations and practical applications.

2. Rao, S. S. (2019). **Engineering Optimization: Theory and Practice, 5th Edition**. John Wiley & Sons.
   - One of the fundamental resources for engineering optimization. Addresses classical and modern optimization methods from an engineering perspective.

3. Arora, J. S. (2016). **Introduction to Optimum Design, 4th Edition**. Academic Press.
   - Focusing on optimum design, this book explains structural optimization with practical examples.

4. Yang, X. S. (2021). **Nature-Inspired Optimization Algorithms, 2nd Edition**. Elsevier.
   - Examines nature-inspired metaheuristic optimization algorithms in detail.

5. Bendsøe, M. P., & Sigmund, O. (2003). **Topology Optimization: Theory, Methods, and Applications**. Springer.
   - Considered a fundamental resource in the field of topology optimization.

### Online Resources

1. [Benchmark Functions](https://www.sfu.ca/~ssurjano/optimization.html) - Common mathematical benchmark functions

## Author

Bilal TAYFUR

## Language

The material is written in Turkish and English.

## License

This repository and its contents are for educational purposes.
