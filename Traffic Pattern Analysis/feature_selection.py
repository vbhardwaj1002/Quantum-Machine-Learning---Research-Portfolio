"""
feature_selection.py
--------------------
Performs Mutual Information (MI) based feature selection on the
CICIDS2017 dataset to identify the most discriminative flow-level
features for HTTPS vs SSH classification.

Also computes Kernel-Target Alignment (KTA) scores across feature
subset sizes (k=4..10) using the quantum feature map, to identify
the optimal quantum-compatible feature subset.

Usage:
    python feature_selection.py --data_path <path_to_combined_csv>
"""

import argparse
import heapq
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.kernels import kernel_matrix
from pennylane.templates import StronglyEntanglingLayers

RANDOM_STATE = 42
SAMPLE_SIZE_PER_CLASS = 100  # Small sample for KTA computation (quantum is slow)
TOP_K_MI = 20               # Number of top MI features to consider


# ── Mutual Information ────────────────────────────────────────────────────────

def compute_mi_scores(df, drop_cols):
    """Compute MI scores for all numeric features against the Target label."""
    X_all = df.drop(columns=drop_cols, errors='ignore')
    y_all = df['Target']

    X_all = X_all.replace([np.inf, -np.inf], np.nan).dropna()
    y_all = y_all.loc[X_all.index]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_all)

    mi_scores = mutual_info_classif(X_scaled, y_all, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=X_all.columns).sort_values(ascending=False)
    return mi_series


# ── Quantum Kernel & KTA ─────────────────────────────────────────────────────

def feature_map(x, wires, n_layers=3):
    """StronglyEntanglingLayers-based quantum feature map."""
    n_wires = len(wires)
    repeated_x = pnp.tile(x, (n_layers, 1))
    weights = pnp.stack([
        pnp.tile(repeated_x[i], 3)[:3 * n_wires].reshape(n_wires, 3)
        for i in range(n_layers)
    ])
    StronglyEntanglingLayers(weights, wires=wires)


def compute_kernel_matrix_qml(X_train, X_test, n_wires):
    """Compute quantum fidelity kernel matrix."""
    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        feature_map(x1, wires=range(n_wires))
        qml.adjoint(feature_map)(x2, wires=range(n_wires))
        return qml.expval(qml.Projector([0] * n_wires, wires=range(n_wires)))

    K_train = kernel_matrix(X_train, X_train, kernel=kernel_circuit)
    K_train += np.eye(len(K_train)) * 1e-4  # Numerical regularisation
    return K_train


def center_matrix(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def compute_kta(K, y):
    """Centered Kernel-Target Alignment score."""
    yyT = np.outer(y, y)
    K_c = center_matrix(K)
    yyT_c = center_matrix(yyT)
    return np.sum(K_c * yyT_c) / (np.linalg.norm(K_c) * np.linalg.norm(yyT_c))


def kta_feature_search(df, top_k_features, k_range=range(4, 11)):
    """Evaluate KTA for each feature subset size and return scores."""
    kta_scores = {}

    for k in k_range:
        features_k = top_k_features[:k]
        df_k = df[features_k + ['Target']].copy()

        df_0 = df_k[df_k['Target'] == 0].sample(SAMPLE_SIZE_PER_CLASS, random_state=RANDOM_STATE)
        df_1 = df_k[df_k['Target'] == 1].sample(SAMPLE_SIZE_PER_CLASS, random_state=RANDOM_STATE)
        df_bal = pd.concat([df_0, df_1])

        X_k = MinMaxScaler().fit_transform(df_bal[features_k])
        y_k = df_bal['Target'].values

        X_train, _, y_train, _ = train_test_split(
            X_k, y_k, test_size=0.3, stratify=y_k, random_state=RANDOM_STATE
        )

        n_wires = X_train.shape[1]
        K_train = compute_kernel_matrix_qml(X_train, None, n_wires)
        kta = compute_kta(K_train, 2 * y_train - 1)

        kta_scores[k] = (kta, features_k)
        print(f"  Top {k} features → KTA: {kta:.4f}")

    return kta_scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MI + KTA feature selection for QSVM traffic classification")
    parser.add_argument('--data_path', required=True, help="Path to combined CICIDS2017 CSV")
    args = parser.parse_args()

    print("=== Loading data ===")
    df = pd.read_csv(args.data_path)
    df.columns = df.columns.str.strip()
    df = df[df['Destination Port'].isin([443, 22])].copy()
    df['Target'] = df['Destination Port'].map({443: 0, 22: 1})

    non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    drop_cols = list(set(['Destination Port', 'Protocol_Type', 'Target']) | set(non_numeric))

    print("\n=== Computing Mutual Information scores ===")
    mi_series = compute_mi_scores(df, drop_cols)
    print(f"\nTop {TOP_K_MI} features by MI:\n{mi_series.head(TOP_K_MI)}")

    top_k_features = mi_series.head(TOP_K_MI).index.tolist()

    print("\n=== Computing KTA scores across feature subset sizes ===")
    kta_scores = kta_feature_search(df, top_k_features)

    # Display top 2 results by KTA
    top2 = heapq.nlargest(2, kta_scores.items(), key=lambda x: x[1][0])
    print("\n=== Best feature subsets by KTA ===")
    for rank, (k, (kta, features)) in enumerate(top2, 1):
        print(f"\nRank #{rank} — Top {k} features (KTA = {kta:.4f}):")
        print(features)


if __name__ == '__main__':
    main()
