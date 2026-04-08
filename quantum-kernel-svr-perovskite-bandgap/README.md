# Hybrid Quantum Kernel SVR for Bandgap Prediction in Halide Perovskites

> **IEEE Conference Paper**  
> Vaishnavi Bhardwaj — J.C. Bose University of Science and Technology, YMCA

---

## Overview

Predicting the electronic bandgap (Eᵍ) of hybrid organic-inorganic halide perovskites (HOIPs) is a central bottleneck in designing high-efficiency solar cells and LEDs. This project proposes a **Hybrid Quantum Kernel Support Vector Regression (QKSVR)** framework that integrates a quantum feature map with a classical SVR optimizer to predict Eᵍ from only three compositional descriptors.

The model achieves **R² = 0.9806** and **RMSE = 0.0618 eV** — near the tolerance of experimental spectroscopic uncertainty — using a 3-qubit, 1-layer quantum circuit with RY encoding, T-gate enhanced nonlinearity, and cyclic CZ entanglement.

---

## Problem Statement

Classical ML models (SVR, Random Forest) have achieved R² > 0.95 on perovskite bandgap datasets, but are constrained by fixed kernel spaces that may miss subtle quantum mechanical correlations governing Eᵍ variations. This work investigates whether:

- Quantum feature maps in high-dimensional Hilbert spaces can capture structure-property relationships inaccessible to classical kernels
- Kernel-Target Alignment (KTA) can serve as a quantum-native criterion for selecting the most informative compositional descriptors
- A minimal 3-qubit circuit achieves competitive or superior accuracy compared to classical baselines

---

## Dataset

- **Source:** Sadhu et al. (2023), *Journal of The Institution of Engineers (India): Series D* — DOI: [10.1007/s40033-023-00553-Z](https://doi.org/10.1007/s40033-023-00553-Z)
- **Size:** 155 unique HOIP compositions with experimentally validated bandgap values
- **Bandgap range:** 1.45 eV to 3.00 eV
- **Initial features (7):** Molar fractions of A-site cations (Cs, FA, MA), halide anions (Cl, Br, I), and Goldschmidt tolerance factor (t)
- **Selected features (3):** {Cs, I, t} — identified via KTA maximization

---

## Methodology

### Preprocessing

Two-step normalization pipeline:

1. **Standardization:** All features scaled to zero mean and unit variance
2. **Phase-friendly normalization:** Standardized features rescaled to `[0, π/2]` for compatibility with RY encoding gates

```
x̃ᵢ = (π/2) × (xᵢ - min(x)) / (max(x) - min(x))
```

Target variable (Eᵍ) is Z-score standardized for the SVR model.

### Quantum Feature Map

A **3-qubit, 1-layer** quantum circuit `U(x̃)` composed of:

1. **Encoding Layer:** Hadamard gates followed by RY rotations encoding the 3 selected features
   ```
   Uenc(x̃) = ⊗ Hᵢ · RYᵢ(β·x̃ᵢ)  for i = 1, 2, 3
   ```

2. **Entangling Layer:** Cyclic CZ gates for quantum correlation mapping
   ```
   Uent = CZ₁,₂ · CZ₂,₃ · CZ₃,₁
   ```

3. **Nonlinearity Enhancement:** T-gate (π/4 Z-rotation) on qubit 2 — a non-Clifford gate that increases expressivity without added circuit depth

The quantum kernel is computed as the fidelity between quantum states:
```
K(x, x') = |⟨0|U†(x) U(x')|0⟩|²
```

### KTA-Driven Feature Selection

All possible 3-feature subsets from the 7 initial features were evaluated by Kernel-Target Alignment:

```
A(K, y) = ⟨K, yy^T⟩_F / sqrt(⟨K,K⟩_F · ⟨yy^T, yy^T⟩_F)
```

The subset **{Cs, I, t}** achieved the highest centered KTA = **0.8337**, outperforming all other combinations including those suggested by classical mutual information.

**Physical interpretation:**
- **I (iodide):** Drives valence band contributions via p-orbital interactions
- **Cs:** Enhances lattice stability and influences band curvature
- **t (tolerance factor):** Captures crystal tolerance and structural strain

### QKSVR Model

The precomputed quantum kernel matrix is passed directly to a classical SVM regressor:

- Regularization: C = 2000.0
- Epsilon: ε = 0.010
- Hyperparameter search: Grid search with cross-validation

---

## Results

| Metric | Value |
|---|---|
| Selected feature set | {Cs, I, t} |
| Circuit architecture | RY + T-gate + CZ entanglement (1 layer) |
| Centered KTA | 0.8337 |
| Test R² | **0.9806** |
| Test RMSE | **0.0618 eV** |
| Mean prediction bias | ~−0.0007 eV |
| SVR parameters | C = 2000, ε = 0.010 |

The KTA of the optimal subset (0.8337) substantially outperforms the classical Mutual Information-guided subset {Cl, Br, I} (KTA = 0.7733), demonstrating that quantum kernel geometry encodes physically relevant correlations beyond classical statistics.

---

## Repository Structure

```
quantum-kernel-svr-perovskite-bandgap/
│
├── README.md
├── requirements.txt
│
└── src/
    ├── preprocess.py           # Standardization and phase-friendly normalization
    ├── quantum_feature_map.py  # 3-qubit RY + T-gate + CZ circuit construction
    ├── kta_feature_selection.py# Exhaustive KTA search over 3-feature subsets
    ├── compute_kernel.py       # Quantum kernel matrix computation
    ├── train_qksvr.py          # SVR training with precomputed quantum kernel
    ├── evaluate.py             # R², RMSE, parity plot, error distribution
    └── visualize.py            # Correlation heatmap, KTA bar chart, parity plot
```

---

## Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

**Key dependencies:**
```
pennylane>=0.35.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Running the Pipeline

```bash
# Step 1: Load and preprocess the dataset
python src/preprocess.py --data_path data/perovskite_dataset.csv

# Step 2: Run KTA-based feature selection
python src/kta_feature_selection.py --n_features 3

# Step 3: Compute the quantum kernel matrix
python src/compute_kernel.py --features Cs I t

# Step 4: Train the QKSVR model
python src/train_qksvr.py --C 2000 --epsilon 0.01

# Step 5: Evaluate and visualize
python src/evaluate.py
python src/visualize.py
```

---

## Key Concepts

**Quantum Kernel Estimation (QKE):** Maps classical input vectors into exponentially large Hilbert spaces via a parameterized quantum circuit, enabling the SVM to learn decision boundaries that are intractable for classical kernels.

**T-gate (Non-Clifford):** A π/4 rotation around the Z-axis. Including a T-gate in the circuit significantly increases expressivity by generating states outside the Clifford group, without meaningfully increasing circuit depth.

**Kernel-Target Alignment (KTA):** Measures how well the kernel matrix structure mirrors the target variable distribution. Maximizing KTA during feature selection is a principled, quantum-native alternative to classical filter methods like Pearson correlation or mutual information.

---

## Citation

If you use this work, please cite:

```
V. Bhardwaj, "Hybrid Quantum Kernel Support Vector Regression for Bandgap Prediction in Halide Perovskites," 2025 International Conference on Digital Innovations for Sustainable Solutions (ICDISS), Faridabad, India, 2025, pp. 1-5, doi: 10.1109/ICDISS68238.2025.11320746.
```

