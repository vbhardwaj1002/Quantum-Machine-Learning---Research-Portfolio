# Quantum Machine Learning for Anomaly Detection in Cyber-Physical Systems

> **Research Paper — ICS / CPS Security**  
> Vaishnavi Bhardwaj, Dr. Ashutosh Dixit  
> J.C. Bose University of Science and Technology, YMCA

---

## Overview

Cyber-physical systems (CPS) such as industrial water treatment plants are increasingly targeted by sophisticated cyberattacks. This project evaluates **Quantum Machine Learning (QML)** as a complementary paradigm for CPS anomaly detection, demonstrating that quantum kernel methods can outperform classical baselines when paired with carefully engineered quantum-compatible feature representations.

Two quantum models are evaluated on the **Secure Water Treatment (SWaT)** testbed dataset:
- A **ZZFeatureMap-based Quantum Kernel SVM (QKSVM)** — achieves **F1 = 0.9851**, **AUC = 0.9998**
- A **Variational Quantum Classifier (VQC)** — achieves F1 = 0.9796 with perfect precision

A novel **4-stage preprocessing pipeline** is the central contribution of this work, addressing the quantum feature engineering problem that prior literature has largely ignored.

---

## Problem Statement

A critical and underexplored challenge in applying QML to real CPS data is quantum-compatible feature engineering. Naive dimensionality reduction via PCA collapses 36 SWaT sensor dimensions such that PC1 captures 97.3% of variance — rendering the quantum circuit effectively 1-dimensional. This work directly addresses this challenge.

A secondary finding is the identification of a **cross-year dataset shift artifact**: when classical supervised models are trained on a naive merge of 2026 normal data and 2015 attack data, all models achieve perfect scores (F1 = AUC = 1.0000) — not because they detect attacks, but because they detect year-specific distributional signatures. Domain adaptation is therefore mandatory before any model training.

---

## Dataset

**Secure Water Treatment (SWaT) Testbed** — iTrust Centre, Singapore University of Technology and Design (SUTD)

| Source | Description | Samples |
|---|---|---|
| SWaT A10 (Feb 2026) | Normal operation — 2 days of recordings | 58,320 |
| SWaT 2015 Attack | 36 distinct cyberattack scenarios, labelled | 54,584 |
| **Combined** | Balanced, cross-year, 36 features | **109,168** |

**Sensor modalities:** Chemical (AIT, 8 sensors), Flow (FIT, 7), Tank level (LIT, 3), Pressure (PIT, 3), Actuators (15)

---

## Methodology

### Four-Stage Preprocessing Pipeline

The core contribution is a preprocessing pipeline that transforms 36-dimensional raw sensor data into 6 statistically independent, quantum-compatible features.

#### Stage 1 — Group-Wise Sensor Normalization
Z-score normalization applied independently within each sensor modality group (AIT, FIT, LIT, PIT, Actuators), preventing high-variance actuator binary states from dominating the covariance structure.

#### Stage 2 — CORAL Domain Adaptation
CORrelation ALignment (CORAL) aligns the second-order statistics (covariance) of the 2026 normal distribution toward the 2015 attack distribution, mitigating cross-year distributional shift without requiring target labels.

```
Minimizes: ||Cov_source - Cov_target||_F
```

#### Stage 3 — ICA Feature Extraction + MI Selection
FastICA extracts 15 statistically independent source signals from the 36-dimensional CORAL-aligned space. ICA maximizes statistical independence (rather than variance like PCA), which aligns naturally with the entanglement structure of quantum circuits.

The top 6 components are selected by mutual information with the binary attack label:

| Component | MI Score |
|---|---|
| IC12 | 0.6232 |
| IC11 | 0.2991 |
| IC7 | 0.2881 |
| IC10 | 0.2460 |
| IC3 | 0.2130 |
| IC5 | 0.2128 |

**Result:** PC1 variance drops from 97.3% (naive PCA) to **44.2%** (proposed ICA), creating a semantically diverse 6-qubit quantum feature space.

#### Stage 4 — Angular Scaling
All 6 ICA components rescaled to `[0, π]` via MinMaxScaler for quantum angle encoding (Ry gates, AngleEmbedding, ZZFeatureMap).

### Preprocessing Comparison

| Method | PC1 Variance | Off-diagonal Correlation | Mean MI |
|---|---|---|---|
| Naive PCA | 97.3% | ~0.40 | 0.693 |
| Group-norm + MI + Decorr. | 79.3% | ~0.23 | 0.341 |
| **Proposed (CORAL + ICA)** | **44.2%** | **0.187** | **0.314** |

### Quantum Models

#### Quantum Kernel SVM (QKSVM)
- Feature map: **ZZFeatureMap** with reps=2 (Hadamard + ZZ-entanglement blocks)
- Hilbert space dimension: 2⁶ = 64 (6 qubits)
- Kernel: Fidelity kernel K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²
- Precomputed kernel matrix (400×400) passed to SVC with `kernel='precomputed'`, C=1.0

#### Variational Quantum Classifier (VQC)
- Encoding: AngleEmbedding (Ry rotations on 6 qubits)
- Ansatz: 3 layers of StronglyEntanglingLayers (54 trainable parameters)
- Output: Expectation value of PauliZ on qubit 0
- Optimizer: Adam, learning rate = 0.05, batch size = 20, 40 epochs
- Loss: Mean squared error between PauliZ expectation and ±1 labels

---

## Results

| Model | F1 | Precision | Recall | AUC |
|---|---|---|---|---|
| Classical SVM (RBF) | 0.9796 | 1.0000 | 0.9600 | 0.9986 |
| Classical Random Forest | 0.9848 | 1.0000 | 0.9700 | 0.9997 |
| **Quantum Kernel SVM** | **0.9851** | 0.9802 | **0.9900** | **0.9998** |
| VQC | 0.9796 | 1.0000 | 0.9600 | 0.9981 |

The **QKSVM achieves the highest F1 and AUC** of all four models, operating in a 64-dimensional Hilbert space with no RBF hyperparameter tuning. This is a quantum advantage result, not merely quantum-classical parity.

---

## Repository Structure

```
qml-anomaly-detection-cps/
│
├── README.md
├── requirements.txt
│
└── src/
    ├── data_loader.py          # Load and merge SWaT 2026 normal + 2015 attack data
    ├── preprocessing.py        # 4-stage pipeline: group-norm → CORAL → ICA → angular scaling
    ├── dataset_shift.py        # Demonstrate and quantify the cross-year shift artifact
    ├── quantum_kernel_svm.py   # ZZFeatureMap kernel computation + SVC training
    ├── vqc.py                  # VQC architecture, training loop, loss tracking
    ├── classical_baselines.py  # Random Forest and SVM-RBF baselines
    ├── evaluate.py             # F1, precision, recall, AUC, confusion matrices
    └── visualize.py            # ROC curves, bar charts, training loss, PCA variance plots
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
qiskit>=0.45.0
qiskit-machine-learning>=0.7.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Running the Pipeline

```bash
# Step 1: Load and merge the SWaT dataset
python src/data_loader.py --normal_path data/swat_2026_normal.csv --attack_path data/swat_2015_attack.csv

# Step 2: Demonstrate the dataset shift artifact
python src/dataset_shift.py

# Step 3: Run the 4-stage preprocessing pipeline
python src/preprocessing.py --n_ica_components 15 --n_selected 6

# Step 4: Train and evaluate QKSVM
python src/quantum_kernel_svm.py --train_size 400 --test_size 200

# Step 5: Train and evaluate VQC
python src/vqc.py --layers 3 --epochs 40 --lr 0.05

# Step 6: Run classical baselines for comparison
python src/classical_baselines.py

# Step 7: Generate all result plots
python src/visualize.py
```

> **Note on data access:** The SWaT dataset is available upon request from the iTrust Centre for Research in Cyber Security at SUTD: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

---

## Key Concepts

**Cyber-Physical System (CPS):** A system integrating computational elements with physical processes. Industrial control systems (ICS) like water treatment plants are critical infrastructure CPS.

**Domain Adaptation (CORAL):** Aligns the second-order statistics (covariance) of source and target distributions without requiring target labels. Critical here because training and attack data come from different years with different distributional signatures.

**ICA vs PCA for Quantum Circuits:** PCA maximizes variance; ICA maximizes statistical independence. For ZZFeatureMap circuits where entanglement creates correlations between qubits, providing ICA-independent features as input ensures each qubit encodes genuinely distinct physical information, maximizing the effective dimensionality of the quantum feature space.

**Barren Plateaus:** Exponential vanishing of gradients with qubit count, a fundamental challenge for VQC training. The Adam optimizer partially mitigates this through adaptive learning rates.

---



---

## Acknowledgements

The SWaT dataset was provided by the iTrust Centre for Research in Cyber Security at the Singapore University of Technology and Design (SUTD).

---

