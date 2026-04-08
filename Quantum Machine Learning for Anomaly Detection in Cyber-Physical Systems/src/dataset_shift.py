"""
dataset_shift.py
----------------
Implements and diagnoses the cross-year dataset shift artifact in the
SWaT hybrid dataset, and applies CORAL domain adaptation to mitigate it.

KEY FINDING (Section 3.2 of the paper):
  When classical supervised models are trained on a naive merge of
  SWaT 2026 normal data and SWaT 2015 attack data WITHOUT domain
  adaptation, all models achieve F1 = AUC = 1.0000. This is NOT
  genuine attack discrimination — supervised models exploit the
  year-specific distributional signature (cross-year shift artefact)
  rather than actual attack behaviour. The Isolation Forest, being
  unsupervised, is immune and achieves only F1 = 0.4726, confirming
  the signal is distributional, not behavioural.

CORAL (CORrelation ALignment) [Sun et al., AAAI 2016]:
  Aligns the second-order statistics (covariance) of the 2026 normal
  distribution toward the 2015 attack distribution by whitening the
  source covariance and re-colouring it with the target covariance.
  This mitigates cross-year distributional shift without requiring
  target labels.

The preprocessing pipeline then applies group-wise normalisation,
CORAL, ICA feature extraction, and angular scaling — in that order.

Usage:
    python dataset_shift.py --data_dir data/processed
                            --output_dir data/processed
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ── CORAL Domain Adaptation ────────────────────────────────────────────────────

class CORALAdapter:
    """
    CORrelation ALignment (CORAL) domain adaptation.

    Aligns the covariance structure of source distribution (2026 normal)
    to target distribution (2015 attack) so models learn genuine attack
    patterns rather than dataset-year artefacts.

    Reference: Sun et al. (2016). Return of frustratingly easy domain
    adaptation. AAAI 2016. arXiv:1511.05547.
    """

    def fit(self, X_src, X_tgt):
        self.src_mean = X_src.mean(axis=0)
        Cs = np.cov(X_src.T) + 1e-6 * np.eye(X_src.shape[1])
        Ct = np.cov(X_tgt.T) + 1e-6 * np.eye(X_tgt.shape[1])
        es, vs = np.linalg.eigh(Cs)
        et, vt = np.linalg.eigh(Ct)
        es = np.maximum(es, 1e-10)
        et = np.maximum(et, 1e-10)
        W       = vs @ np.diag(1.0 / np.sqrt(es)) @ vs.T   # whiten source
        C       = vt @ np.diag(np.sqrt(et))        @ vt.T   # colour to target
        self.T  = W @ C
        return self

    def transform(self, X):
        return (X - self.src_mean) @ self.T

    def fit_transform(self, X_src, X_tgt):
        self.fit(X_src, X_tgt)
        return self.transform(X_src), self.transform(X_tgt)


# ── Group-wise normalization ───────────────────────────────────────────────────

def groupwise_normalize(X, feature_cols):
    """
    Apply Z-score normalization independently within each sensor modality
    group (AIT chemical, FIT flow, LIT level, PIT pressure, Actuators).

    This prevents high-variance actuator binary states from dominating
    the covariance structure of the combined feature matrix.
    """
    sensor_groups = {
        'AIT (Chemical)': [c for c in feature_cols if c.startswith('AIT')],
        'FIT (Flow)':     [c for c in feature_cols if c.startswith('FIT')],
        'LIT (Level)':    [c for c in feature_cols if c.startswith('LIT')],
        'PIT (Pressure)': [c for c in feature_cols if c.startswith('PIT')],
        'Actuators':      [c for c in feature_cols
                           if not any(c.startswith(p)
                                      for p in ['AIT', 'FIT', 'LIT', 'PIT'])],
    }

    X_norm  = np.zeros_like(X, dtype=float)
    scalers = {}
    for gname, gcols in sensor_groups.items():
        idxs = [feature_cols.index(c) for c in gcols if c in feature_cols]
        if idxs:
            sc = StandardScaler()
            X_norm[:, idxs] = sc.fit_transform(X[:, idxs])
            scalers[gname]  = (idxs, sc)
            print(f"  {gname:<22}: {len(idxs)} sensors normalised")
    return X_norm, scalers


def groupwise_transform(X, scalers):
    X_norm = np.zeros_like(X, dtype=float)
    for _, (idxs, sc) in scalers.items():
        X_norm[:, idxs] = sc.transform(X[:, idxs])
    return X_norm


# ── ICA feature extraction ─────────────────────────────────────────────────────

def extract_ica_features(X_train_coral, X_test_coral, X_full_coral,
                         y_full, n_ica_total=15, n_qubits=6,
                         random_state=42):
    """
    FastICA extracts statistically independent source signals.

    Unlike PCA (maximises variance), ICA maximises statistical
    independence, aligning naturally with the ZZFeatureMap entanglement
    structure which creates correlations between qubits. Providing
    independent input features ensures each qubit encodes genuinely
    distinct physical information.

    Result: PC1 variance drops from 97.3% (naive PCA) to ~44.2% (ICA),
    confirming a semantically diverse quantum feature space.
    """
    ica        = FastICA(n_components=n_ica_total, random_state=random_state,
                         max_iter=2000, tol=0.0001)
    ica_scaler = StandardScaler()

    X_ica_full = ica_scaler.fit_transform(ica.fit_transform(X_full_coral))

    mi_ica     = mutual_info_classif(X_ica_full, y_full, random_state=random_state)
    top_idx    = np.argsort(mi_ica)[-n_qubits:][::-1]

    print(f"\n  Top {n_qubits} ICA components by MI with attack label:")
    for rank, idx in enumerate(top_idx):
        print(f"    IC{idx+1:2d}: MI = {mi_ica[idx]:.4f}")

    X_tr_ica_all = ica_scaler.transform(ica.transform(X_train_coral))
    X_te_ica_all = ica_scaler.transform(ica.transform(X_test_coral))

    X_tr_ica = X_tr_ica_all[:, top_idx]
    X_te_ica = X_te_ica_all[:, top_idx]

    # Diagnostics
    corr      = np.corrcoef(X_tr_ica.T)
    off_diag  = np.abs(corr - np.eye(n_qubits)).mean()
    pca_check = PCA(n_components=n_qubits, whiten=True, random_state=random_state)
    pca_check.fit(X_tr_ica)
    var_exp   = pca_check.explained_variance_ratio_

    print(f"\n  Off-diagonal correlation  : {off_diag:.4f}  (near 0 = independent)")
    print(f"  PCA variance distribution :")
    for i, v in enumerate(var_exp):
        print(f"    PC{i+1}: {v*100:5.1f}%")
    print(f"  Max PC1: {max(var_exp)*100:.1f}%  (naive PCA baseline: 97.3%)")

    return X_tr_ica, X_te_ica, top_idx, mi_ica, var_exp, off_diag


# ── Angular scaling ────────────────────────────────────────────────────────────

def angular_scale(X_train_ica, X_test_ica):
    """Scale ICA features to [0, pi] for quantum angle encoding (Ry gates)."""
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_tr   = scaler.fit_transform(X_train_ica)
    X_te   = scaler.transform(X_test_ica)
    return X_tr, X_te, scaler


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(data_dir, output_dir,
                               n_ica_total=15, n_qubits=6, random_state=42):
    """
    Full 4-stage preprocessing pipeline:
      1. Group-wise normalisation
      2. CORAL domain adaptation
      3. ICA feature extraction + MI selection
      4. Angular scaling to [0, pi]
    """
    print("=== Loading data ===")
    X_train_full = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test_full  = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train_full = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test_full  = np.load(os.path.join(data_dir, 'y_test.npy'))

    df       = pd.read_csv(os.path.join(data_dir, 'swat_final_labeled.csv'))
    X_full   = df[[c for c in df.columns if c != 'label']].values
    y_full   = df['label'].values
    feat_cols = [c for c in df.columns if c != 'label']

    print(f"  Train: {X_train_full.shape} | Test: {X_test_full.shape}")
    print(f"  Normal: {(y_full == 0).sum():,} | Attack: {(y_full == 1).sum():,}")

    print("\n=== Stage 1: Group-wise Sensor Normalization ===")
    X_gnorm_full,  group_scalers = groupwise_normalize(X_full,       feat_cols)
    X_train_gnorm               = groupwise_transform(X_train_full,  group_scalers)
    X_test_gnorm                = groupwise_transform(X_test_full,   group_scalers)

    print("\n=== Stage 2: CORAL Domain Adaptation ===")
    X_tr_n = X_train_gnorm[y_train_full == 0]
    X_tr_a = X_train_gnorm[y_train_full == 1]

    coral = CORALAdapter()
    X_tr_n_al, X_tr_a_al = coral.fit_transform(X_tr_n, X_tr_a)

    X_train_coral = np.zeros_like(X_train_gnorm)
    X_train_coral[y_train_full == 0] = X_tr_n_al
    X_train_coral[y_train_full == 1] = X_tr_a_al
    X_test_coral  = coral.transform(X_test_gnorm)

    coral_full = CORALAdapter()
    X_fn_al, X_fa_al = coral_full.fit_transform(
        X_gnorm_full[y_full == 0], X_gnorm_full[y_full == 1])
    X_full_coral = np.zeros_like(X_gnorm_full)
    X_full_coral[y_full == 0] = X_fn_al
    X_full_coral[y_full == 1] = X_fa_al
    print("  CORAL alignment applied.")

    print(f"\n=== Stage 3: ICA Feature Extraction ({n_ica_total} components -> top {n_qubits}) ===")
    X_tr_ica, X_te_ica, top_idx, mi_ica, var_exp, off_diag = extract_ica_features(
        X_train_coral, X_test_coral, X_full_coral,
        y_full, n_ica_total, n_qubits, random_state
    )

    print("\n=== Stage 4: Angular Scaling to [0, pi] ===")
    X_train_qml, X_test_qml, _ = angular_scale(X_tr_ica, X_te_ica)
    print(f"  Features scaled to [0, pi] for quantum angle encoding.")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train_qml.npy'), X_train_qml)
    np.save(os.path.join(output_dir, 'X_test_qml.npy'),  X_test_qml)
    np.save(os.path.join(output_dir, 'y_train_qml.npy'), y_train_full)
    np.save(os.path.join(output_dir, 'y_test_qml.npy'),  y_test_full)
    print(f"\nQML-ready arrays saved to: {output_dir}")
    print("  X_train_qml.npy / X_test_qml.npy / y_train_qml.npy / y_test_qml.npy")

    return X_train_qml, X_test_qml, y_train_full, y_test_full


def main():
    parser = argparse.ArgumentParser(
        description='CORAL + ICA preprocessing pipeline for CPS anomaly detection')
    parser.add_argument('--data_dir',    default='data/processed')
    parser.add_argument('--output_dir',  default='data/processed')
    parser.add_argument('--n_ica',       type=int, default=15,
                        help='Total ICA components to extract (default: 15)')
    parser.add_argument('--n_qubits',    type=int, default=6,
                        help='Number of ICA components to keep for QML (default: 6)')
    args = parser.parse_args()

    run_preprocessing_pipeline(
        args.data_dir, args.output_dir, args.n_ica, args.n_qubits)


if __name__ == '__main__':
    main()
