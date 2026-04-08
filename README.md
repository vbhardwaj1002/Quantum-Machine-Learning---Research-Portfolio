# Quantum Machine Learning — Research Portfolio

**Vaishnavi Bhardwaj**  
Department of Computer Science Engineering  
J.C. Bose University of Science and Technology, YMCA, Faridabad, India  
📧 vbhardwaj1002@gmail.com

---

## About This Repository

This repository contains four research projects at the intersection of **Quantum Machine Learning (QML)** and real-world prediction and security problems. Each project investigates quantum kernel methods, variational quantum classifiers, and hybrid quantum-classical architectures applied to domains ranging from materials science to cybersecurity and clinical genomics.

The work spans my M.Tech thesis and independent research publications, collectively demonstrating the practical viability of near-term NISQ quantum algorithms across diverse high-stakes domains.

---

## Projects

### 1. 🔐 Traffic Pattern Analysis Using QSVMs for Side-Channel Cryptanalysis
> *M.Tech Thesis Project*

Applies Quantum Support Vector Machines to identify cryptographic key leakage from power/EM side-channel traces. Uses quantum kernel methods to distinguish traffic patterns associated with different key hypotheses — a task that demands high-dimensional feature discrimination where quantum feature maps offer a natural advantage.

📁 [`traffic-pattern-qsvm-side-channel/`](./Traffic Pattern Analysis/)

---

### 2. ⚛️ Hybrid Quantum Kernel SVR for Bandgap Prediction in Halide Perovskites
> *IEEE Conference Paper*

Proposes a Hybrid Quantum Kernel Support Vector Regression (QKSVR) framework for predicting the electronic bandgap (Eᵍ) of hybrid organic-inorganic halide perovskites. Achieves R² = 0.9806 and RMSE = 0.0618 eV using only 3 features selected via Kernel-Target Alignment (KTA = 0.8337).

📁 [`quantum-kernel-svr-perovskite-bandgap/`](./quantum-kernel-svr-perovskite-bandgap/)

---

### 3. 🦠 Alignment-Aware QSVMs for Antimicrobial Resistance Prediction
> *Research Paper — AMR / Clinical Genomics*

Uses fidelity-based quantum kernels with KTA-driven feature selection to predict resistance to Ampicillin, Ciprofloxacin, and Cefotaxime in *E. coli* from whole-genome data (NCBI MicroBIGG-E). Achieves 96–97% accuracy across all three drug classes on datasets of up to 49,847 isolates.

📁 [`alignment-aware-qsvm-amr/`](./alignment-aware-qsvm-amr/)

---

### 4. 🏭 Quantum ML for Anomaly Detection in Cyber-Physical Systems
> *Research Paper — ICS / CPS Security*

Evaluates a ZZFeatureMap-based Quantum Kernel SVM and a Variational Quantum Classifier (VQC) on the SWaT water treatment testbed dataset. Proposes a novel 4-stage preprocessing pipeline (group-wise normalization → CORAL domain adaptation → ICA → angular scaling) achieving F1 = 0.9851 and AUC = 0.9998.

📁 [`qml-anomaly-detection-cps/`](./qml-anomaly-detection-cps/)

---

## Common Themes & Methodology

| Theme | Description |
|---|---|
| **Quantum Kernels** | All projects use quantum kernel estimation (QKE) — computing inner products in exponentially large Hilbert spaces |
| **KTA-Guided Feature Selection** | Kernel-Target Alignment used across projects to select the most quantum-compatible feature subsets |
| **Hybrid Architecture** | Quantum feature maps combined with classical SVMs/SVR for tractable training on NISQ hardware |
| **Real-World Datasets** | Every project is evaluated on real experimental or industrial datasets, not synthetic benchmarks |

---

## Tech Stack

| Library | Purpose |
|---|---|
| [PennyLane](https://pennylane.ai/) | Quantum circuit simulation and differentiation |
| [Qiskit](https://qiskit.org/) | Quantum circuit construction and feature maps |
| [scikit-learn](https://scikit-learn.org/) | SVM/SVR, preprocessing, cross-validation |
| [NumPy / SciPy](https://numpy.org/) | Numerical computation |
| [Pandas](https://pandas.pydata.org/) | Data handling |
| [Matplotlib / Seaborn](https://matplotlib.org/) | Visualization |

---

## Repository Structure

```
quantum-ml-portfolio/
│
├── README.md                              ← You are here
│
├── traffic-pattern-qsvm-side-channel/     ← M.Tech Thesis
│   ├── README.md
│   ├── requirements.txt
│   └── src/
│
├── quantum-kernel-svr-perovskite-bandgap/ ← IEEE Paper
│   ├── README.md
│   ├── requirements.txt
│   └── src/
│
├── alignment-aware-qsvm-amr/              ← AMR Paper
│   ├── README.md
│   ├── requirements.txt
│   └── src/
│
└── qml-anomaly-detection-cps/             ← CPS Security Paper
    ├── README.md
    ├── requirements.txt
    └── src/
```

---

## How to Run Any Project

Each project folder contains its own `README.md` with detailed setup and execution instructions. The general steps are:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/quantum-ml-portfolio.git
cd quantum-ml-portfolio

# 2. Navigate to the project of interest
cd quantum-kernel-svr-perovskite-bandgap/

# 3. Install dependencies
pip install -r requirements.txt

# 4. Follow project-specific README instructions
```

> **Python version:** 3.9+ recommended for all projects.

---

## Publications & Status

| Project | Status |
|---|---|
| Traffic Pattern QSVM | M.Tech Thesis (submitted) |
| Perovskite Bandgap QKSVR | Under review — IEEE |
| AMR Prediction QSVM | Manuscript prepared |
| CPS Anomaly Detection QML | Manuscript prepared |

---

*For questions or collaboration inquiries, feel free to reach out via email.*
