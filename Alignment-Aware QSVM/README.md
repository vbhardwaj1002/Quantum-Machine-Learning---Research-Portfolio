# Alignment-Aware Quantum Support Vector Machines for Predicting Beta-Lactam and Fluoroquinolone Resistance

> **Research Paper — Quantum ML for Clinical Genomics**  
> Vaishnavi Bhardwaj, Dr. Ashutosh Dixit, Dr. Manish Kumar  
> J.C. Bose University of Science and Technology, YMCA & Shri Vishwakarma Skill University

---

## Overview

Antimicrobial resistance (AMR) is a critical global health emergency projected to cause ~40 million deaths annually by 2050. This project applies **Quantum Support Vector Machines (QSVMs)** with fidelity-based quantum kernels and Kernel-Target Alignment (KTA)-driven feature selection to predict antibiotic resistance in *Escherichia coli* from whole-genome sequencing data.

The model predicts resistance to three clinically important antibiotics — **Ampicillin (AMP)**, **Ciprofloxacin (CIP)**, and **Cefotaxime (CTX)** — from large-scale NCBI genomic datasets, achieving **96–97% accuracy** across all three drug classes while using interpretable, biologically meaningful feature subsets.

---

## Problem Statement

Classical ML models for AMR prediction face challenges with:
- **Extreme class imbalance** (AMP: 0.39% susceptible; CIP: 9.01%; CTX: 34.98%)
- **High-dimensional sparse genomic features** (thousands of resistance genes/mutations)
- **Generalizability** across geographic settings and data sources
- **Interpretability** — "black-box" models reduce clinical trust

This work investigates whether **quantum kernel methods** combined with KTA-guided feature selection can:
- Maintain high accuracy under severe class imbalance
- Identify compact, biologically interpretable genomic feature subsets
- Achieve competitive performance against classical ML benchmarks on large clinical datasets

---

## Dataset

- **Source:** NCBI Pathogen Detection (MicroBIGG-E) database
- **Organism:** *Escherichia coli*
- **Antibiotics:** Ampicillin (AMP), Ciprofloxacin (CIP), Cefotaxime (CTX)

| Antibiotic | Total Isolates | Susceptible | Resistant |
|---|---|---|---|
| Ampicillin (AMP) | 49,847 | 193 (0.39%) | 49,654 (99.61%) |
| Ciprofloxacin (CIP) | 30,259 | 2,727 (9.01%) | 27,532 (90.99%) |
| Cefotaxime (CTX) | 20,687 | 7,239 (34.98%) | 13,448 (65.02%) |

**Features:** Acquired resistance genes and chromosomal mutations extracted from TSV files. Features include known AMR genes (e.g., *blaCTX-M-15* for CTX resistance).

---

## Methodology

### Feature Construction & Selection

A two-step pipeline reduces high-dimensional genomic data to a compact quantum-compatible feature set:

1. **Mutual Information (MI) Ranking:** Features ranked by non-linear dependency with resistance labels. For CTX, the *blaCTX-M-15* gene had the highest MI score (0.2144).

2. **Correlation Pruning:** Features with pairwise Pearson correlation above a threshold are pruned to remove redundant resistance markers, retaining biological diversity in the selected set.

### Quantum Feature Map

- **Kernel type:** Fidelity-based quantum kernel (ZZ-Feature Map or RY-entangled circuit)
- **Encoding:** Angular encoding of normalized genomic features into qubit rotation angles
- **Entanglement:** CZ or ZZ-interaction layers to capture feature correlations in Hilbert space
- **Kernel computation:**
  ```
  K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²
  ```

### KTA-Guided Circuit Alignment

Kernel-Target Alignment is used as a diagnostic metric to:
- Evaluate how well the quantum kernel geometry aligns with resistance labels
- Select the feature subset that maximizes alignment for each antibiotic separately
- Guide circuit design to prevent overfitting on imbalanced data

**KTA scores by antibiotic:**
- **AMP:** KTA = 0.9434 (very high alignment)
- **CIP:** KTA ≈ 0.58–0.60 (moderate but predictive)
- **CTX:** KTA ≈ 0.58–0.60 (moderate but predictive)

### Classifier

- Classical SVM with precomputed quantum kernel (`kernel='precomputed'`)
- Class weights adjusted for imbalance (`class_weight='balanced'`)
- Evaluation: Stratified k-fold cross-validation + held-out test set

---

## Results

| Antibiotic | Accuracy | Precision | Recall | F1 | AUC ROC |
|---|---|---|---|---|---|
| Ampicillin (AMP) | **97%** | 98% | 97% | 97% | 0.93–0.99 |
| Ciprofloxacin (CIP) | **96%** | 96% | 96% | 96% | 0.93–0.99 |
| Cefotaxime (CTX) | **97%** | 97% | 97% | 97% | 0.93–0.99 |

The QSVM achieves performance comparable or superior to classical approaches (classical ML benchmark: 93–97% accuracy from Moradigaravand et al., 2018) while operating on biologically interpretable, compact feature subsets selected via quantum-native KTA diagnostics.

---

## Repository Structure

```
alignment-aware-qsvm-amr/
│
├── README.md
├── requirements.txt
│
└── src/
    ├── data_loader.py          # Load and parse NCBI MicroBIGG-E TSV files
    ├── feature_selection.py    # MI ranking + correlation pruning pipeline
    ├── quantum_kernel.py       # Fidelity-based quantum kernel computation
    ├── kta_alignment.py        # KTA computation and feature subset evaluation
    ├── train_qsvm.py           # SVM with precomputed quantum kernel, class balancing
    ├── evaluate.py             # Accuracy, precision, recall, F1, AUC, confusion matrix
    └── visualize.py            # KTA scores, ROC curves, feature importance plots
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
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Running the Pipeline

```bash
# Step 1: Load and prepare genomic feature matrices
python src/data_loader.py --antibiotic AMP --data_path data/microbigge/

# Step 2: Feature selection via MI ranking + correlation pruning
python src/feature_selection.py --antibiotic AMP --top_k 20

# Step 3: Evaluate KTA scores for candidate feature subsets
python src/kta_alignment.py --antibiotic AMP

# Step 4: Compute quantum kernel matrix on selected features
python src/quantum_kernel.py --antibiotic AMP

# Step 5: Train and evaluate QSVM
python src/train_qsvm.py --antibiotic AMP

# Step 6: Visualize results
python src/visualize.py --antibiotic AMP
```

Repeat Steps 1–6 with `--antibiotic CIP` and `--antibiotic CTX` for the other drug classes.

---

## Key Concepts

**Antimicrobial Resistance (AMR):** The ability of microorganisms to resist antimicrobial drugs, driven by resistance genes (e.g., *blaCTX-M-15* for cephalosporin resistance) and chromosomal mutations.

**Kernel-Target Alignment (KTA):** A diagnostic metric measuring how well the quantum kernel matrix structure mirrors resistance labels. Antibiotic-specific KTA differences reflect genuine variation in quantum feature map informativeness across drug classes.

**Class Imbalance:** AMP has 0.39% susceptible isolates. Without handling imbalance, even a trivially resistant-predicting model achieves ~99.6% accuracy. Class-weighted SVM training and stratified evaluation are critical.

---

## Comparison with Classical Baselines

| Study | Method | AMP Acc. | CIP Acc. | CTX Acc. |
|---|---|---|---|---|
| Moradigaravand et al. (2018) | Classical ML (E. coli pan-genome) | 93% | 93% | 97% |
| Nsubuga et al. (2024) | ML (multi-country) | 58% | 87% | 92% |
| **This Work** | **QSVM + KTA** | **97%** | **96%** | **97%** |

---

## References

- Naghavi, M., et al. (2024). Global burden of antimicrobial resistance. *The Lancet*.
- Moradigaravand, D., et al. (2018). Prediction of antibiotic resistance in *E. coli* from large-scale pan-genome data. *PLOS Computational Biology*.
- You, Y., et al. (2025). Hybrid LLM-quantum platform for *Salmonella* AMR prediction.
- Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567, 209–212.
