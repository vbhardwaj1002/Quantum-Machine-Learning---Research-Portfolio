# Traffic Pattern Analysis Using Quantum Support Vector Machines for Side-Channel Cryptanalysis

> **M.Tech Thesis Project**
> Vaishnavi Bhardwaj, Dr. Ashutosh Dixit
> Department of Computer Science Engineering, J.C. Bose University of Science and Technology, YMCA, Faridabad, India

---

## Overview

Encrypted network traffic, despite securing payload content, inevitably leaks observable metadata — packet sizes, inter-arrival times, and flow-level behavioral signatures. This project investigates whether **Quantum Support Vector Machines (QSVMs)** can exploit these flow-level statistical patterns to classify encrypted traffic types and detect side-channel vulnerabilities, without ever decrypting the content.

A QSVM model and a classical SVM baseline are trained on the **CICIDS2017** dataset to distinguish between HTTPS and SSH encrypted traffic. The QSVM consistently outperforms the classical model, with the most significant advantage emerging under **data-constrained conditions**: on just 100 samples per class, QSVM maintains **96.6% accuracy** while classical SVM degrades to **81.03%**.

---

## Problem Statement

Traditional cryptanalysis requires direct access to encryption keys or algorithms. Modern side-channel attacks are more subtle — they exploit observable metadata that leaks through the physical or behavioral implementation of protocols. Flow-level characteristics such as packet size, inter-arrival time, and flow duration can fingerprint the encryption protocol in use or flag anomalous activity, even without decryption.

This work investigates whether QSVMs — by mapping traffic features into high-dimensional quantum Hilbert spaces — can more reliably identify protocol signatures and behavioral anomalies than classical kernel methods, especially under limited data availability.

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset 2017

| Property | Value |
|---|---|
| Source | Raw `.pcap` packet captures |
| Feature extraction tool | CICFlowMeter |
| Total raw features | 78 flow-level features |
| Filtered dataset size | 288,625 instances (HTTPS + SSH) |
| Sampled size | 500 instances per class |
| Target classes | HTTPS (Port 443) → Label 0, SSH (Port 22) → Label 1 |
| Train/Test split | 70% train / 30% test (stratified) |

> Dataset citation: Ranjit Panigrahi (2025). CICIDS2017. IEEE Dataport. https://dx.doi.org/10.21227/akxq-9v09

---

## Methodology

### Preprocessing Pipeline

1. **Data loading:** Raw `.pcap` files processed using CICFlowMeter to extract 78 flow-based features
2. **Target labeling:** Traffic filtered to HTTPS (port 443) and SSH (port 22), mapped to binary labels
3. **Data cleaning:** Rows with missing values dropped
4. **Sampling:** 500 balanced instances per class selected from 288,625 filtered records
5. **Normalization:** MinMaxScaler applied to scale all features to `[0, 1]`
6. **Train/test split:** 70/30 stratified split
7. **SMOTE oversampling:** Applied on training data to address any class imbalance

### Feature Selection

Mutual Information (MI) scoring used to identify the 5 most discriminative flow-level features — identical feature set used for both classical SVM and QSVM to ensure a fair comparison:

| Feature | MI Score (Classical) | MI Score (QSVM) |
|---|---|---|
| `Init_Win_bytes_backward` | 0.139356 | 0.139164 |
| `Bwd Packets/s` | 0.112138 | 0.112141 |
| `Fwd IAT Max` | 0.105387 | 0.105095 |
| `Flow IAT Max` | 0.105325 | 0.105055 |
| `Max Packet Length` | 0.104704 | 0.104788 |

**Physical interpretation of features:**
- `Max Packet Length` — HTTPS transfers large packets (images, video, web content); SSH handles small command-line packets
- `Flow IAT Max` / `Fwd IAT Max` — Capture temporal session dynamics: handshake and keep-alive mechanisms differ between protocols
- `Init_Win_bytes_backward` / `Bwd Packets/s` — Reflect protocol-specific session initialization and response behavior

### Classical SVM Baseline

- Model: `sklearn.svm.SVC`
- Hyperparameter tuning: `GridSearchCV` (C, kernel, gamma)
- Evaluation: Accuracy, Precision, Recall, F1-score, ROC/AUC, Confusion Matrix
- SHAP analysis applied for feature-level interpretability

### Quantum Feature Map and QSVM

- **Framework:** PennyLane (`lightning.qubit` simulator)
- **Qubits:** 5 (one per selected feature)
- **Feature map:** Custom circuit based on `StronglyEntanglingLayers`
  - Encoding: RX, RY, RZ rotation gates applied to each qubit
  - Entanglement: CNOT gates across qubits
  - Depth: Controlled by `n_layers` parameter
- **Why StronglyEntanglingLayers?** Generates highly entangled quantum states capable of capturing non-linear correlations in network traffic that classical kernels cannot represent. Successive layers of single-qubit rotations and CNOT gates allow richer exploration of the Hilbert space than shallower alternatives (ZZFeatureMap, PauliFeatureMap) for this task.
- **Quantum kernel:** Fidelity kernel — K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|² — computed via PennyLane's `kernel_matrix` function
- **QSVM training:** `sklearn.svm.SVC` with `kernel='precomputed'` using the precomputed quantum kernel matrix
- **Hyperparameter tuning:** `GridSearchCV` — same protocol as classical SVM for fair comparison

---

## Results

### Performance Comparison

| Metric | Classical SVM | QSVM |
|---|---|---|
| Test Accuracy (500 samples/class) | 97.3% | **98.0%** |
| Test Accuracy (100 samples/class) | 81.03% | **96.6%** |
| Average Accuracy (8 dataset evaluations) | 96.7% | **97.5%** |
| Precision (Macro Avg) | 0.97 | **0.98** |
| Recall (Macro Avg) | 0.97 | **0.98** |
| F1-Score (Macro Avg) | 0.97 | **0.98** |
| ROC AUC | 1.00 | 1.00 |

### Key Finding — Quantum Advantage Under Data Scarcity

The most significant result is the **15.57 percentage point gap** between QSVM (96.6%) and classical SVM (81.03%) on only 100 samples per class. Classical SVMs rely on dense training data to find an optimal separating hyperplane in the original feature space. With limited samples, generalization degrades sharply.

The QSVM's quantum feature map projects data into a high-dimensional Hilbert space where classes become more linearly separable even with fewer training points. The t-SNE projection of the quantum kernel matrix shows two clearly distinct, well-separated clusters (HTTPS and SSH), confirming that the quantum Hilbert space transformation provides clean class structure.

### Confusion Matrix — QSVM (500 samples/class test set)

|  | Predicted HTTPS | Predicted SSH |
|---|---|---|
| **Actual HTTPS** | 144 | 6 |
| **Actual SSH** | 0 | 150 |

---

## Implications for Side-Channel Cryptanalysis

Both models confirm that **encrypted traffic leaves exploitable statistical fingerprints** even without content access. The selected features (packet length, inter-arrival times, flow rates) are not arbitrary — they reflect protocol-specific behavioral signatures that persist through encryption, constituting a "metadata fingerprint."

This validates that content-level encryption alone is an incomplete defense. The QSVM's heightened performance on small datasets means this threat intensifies as quantum computing resources become more accessible to adversaries.

**Recommendations for quantum-resilient system design:**
- **Traffic obfuscation:** Adaptive packet padding, randomized inter-arrival times, dummy traffic insertion
- **Adaptive flow shaping:** Dynamically alter traffic parameters in real time to prevent predictable signatures
- **Protocol-level countermeasures:** Design future protocols with native metadata randomization mechanisms

---

## Repository Structure

```
traffic-pattern-qsvm-side-channel/
│
├── README.md
├── requirements.txt
│
└── src/
    ├── preprocess.py               # CICFlowMeter feature loading, cleaning, SMOTE, normalization
    ├── feature_selection.py        # Mutual Information scoring and top-k feature selection
    ├── classical_svm.py            # SVM baseline with GridSearchCV tuning
    ├── quantum_feature_map.py      # StronglyEntanglingLayers circuit (RX, RY, RZ + CNOT)
    ├── quantum_kernel.py           # Fidelity kernel matrix computation via PennyLane
    ├── train_qsvm.py               # QSVM training with precomputed kernel + GridSearchCV
    ├── evaluate.py                 # Accuracy, F1, confusion matrix, ROC/AUC across sample sizes
    ├── shap_analysis.py            # SHAP feature importance for classical SVM interpretability
    └── visualize.py                # Kernel matrix heatmap, t-SNE projection, ROC curves
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
imbalanced-learn>=0.11.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.43.0
scipy>=1.10.0
```

### Running the Pipeline

```bash
# Step 1: Preprocess CICIDS2017 data (assumes CICFlowMeter output CSVs)
python src/preprocess.py --data_path data/cicids2017/ --output data/processed/

# Step 2: Feature selection via Mutual Information
python src/feature_selection.py --input data/processed/features.csv --top_k 5

# Step 3: Train and evaluate classical SVM baseline
python src/classical_svm.py --input data/processed/selected.csv

# Step 4: Compute quantum kernel matrix
python src/quantum_kernel.py --input data/processed/selected.csv --n_layers 2

# Step 5: Train and evaluate QSVM
python src/train_qsvm.py --kernel data/processed/kernel_train.npy --kernel_test data/processed/kernel_test.npy

# Step 6: Run SHAP analysis on classical SVM
python src/shap_analysis.py

# Step 7: Generate all visualizations
python src/visualize.py
```

> **Data access:** CICIDS2017 is publicly available via IEEE Dataport: https://dx.doi.org/10.21227/akxq-9v09

---

## Key Concepts

**Side-Channel Attack:** An attack that infers sensitive information from observable physical or behavioral characteristics of a system — here, flow-level metadata — rather than attacking the cryptographic algorithm directly.

**StronglyEntanglingLayers:** A PennyLane ansatz applying alternating layers of single-qubit rotations (RX, RY, RZ) and two-qubit CNOT entangling gates. Produces highly expressive quantum states that explore the Hilbert space more thoroughly than shallower circuits.

**Quantum Kernel (Fidelity Kernel):** K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|² — the squared overlap between quantum states prepared by the feature map circuit. Used as a precomputed kernel matrix passed to the classical SVM optimizer.

**Quantum Advantage Under Data Scarcity:** In high-dimensional Hilbert space, data points from different classes may become linearly separable with fewer training examples than required in classical feature space — explaining the dramatic 15.57-point accuracy gap at 100 samples per class.

---

## Limitations

- Experiments run on a **simulated quantum device** (`lightning.qubit`) — does not account for NISQ hardware noise, decoherence, or gate errors
- Evaluated on binary classification (HTTPS vs. SSH); multi-class extension is future work
- Kernel matrix computation scales quadratically with training set size — tractability on larger datasets requires hardware acceleration or approximation methods

---

