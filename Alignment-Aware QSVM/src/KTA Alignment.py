"""
kta_alignment.py
----------------
Kernel-Target Alignment (KTA) computation and reporting.

KTA measures how well a quantum kernel matrix structure mirrors the
resistance label distribution — a higher score indicates the quantum
feature map creates a more informative embedding for the supervised
classification task.

Formula (centred KTA):
    KTA(K, y) = <Kc, yyTc>_F / (||Kc||_F * ||yyTc||_F)

where Kc and yyTc are the centred versions of K and the label outer
product, and <·,·>_F is the Frobenius inner product.

Usage:
    python kta_alignment.py --antibiotic AMP
                            --K_train data/processed/AMP_K_train.npy
                            --y_train data/processed/AMP_y_train.npy
"""

import argparse

import numpy as np


# ── KTA computation ────────────────────────────────────────────────────────────

def centre_matrix(M):
    """Double-centre a symmetric matrix: Mc = HMH where H = I - 11T/n."""
    n = M.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ M @ H


def compute_kta(K, y):
    """
    Centred Kernel-Target Alignment between kernel K and binary labels y.

    Parameters
    ----------
    K : np.ndarray, shape (n, n) — kernel matrix
    y : array-like, shape (n,)   — binary labels {0, 1}

    Returns
    -------
    float : KTA score in [-1, 1]; higher is better
    """
    y = np.asarray(y, dtype=float)
    y_pm = 2 * y - 1          # Map {0, 1} -> {-1, +1}
    yyT = np.outer(y_pm, y_pm)

    Kc   = centre_matrix(K)
    yyTc = centre_matrix(yyT)

    numerator   = np.sum(Kc * yyTc)
    denominator = np.linalg.norm(Kc) * np.linalg.norm(yyTc)
    return float(numerator / denominator)


def compute_raw_kta(K, y):
    """
    Un-centred (raw) KTA — included for diagnostic comparison.
    The centred version is the standard choice for kernel selection.
    """
    y = np.asarray(y, dtype=float)
    y_centered = y - np.mean(y)
    T = np.outer(y_centered, y_centered)
    numerator   = np.sum(K * T)
    denominator = np.sqrt(np.sum(K ** 2) * np.sum(T ** 2))
    return float(numerator / denominator)


def report_kta(K_train, y_train, antibiotic):
    """Print a formatted KTA report for one antibiotic."""
    kta_centred = compute_kta(K_train, y_train)
    kta_raw     = compute_raw_kta(K_train, y_train)

    print(f"\n=== KTA Report: {antibiotic} ===")
    print(f"  Centred KTA : {kta_centred:.4f}")
    print(f"  Raw KTA     : {kta_raw:.4f}")

    if kta_centred > 0.5:
        print("  Interpretation: HIGH alignment — quantum feature map encodes "
              "resistance-relevant structure effectively.")
    elif kta_centred > 0.2:
        print("  Interpretation: MODERATE alignment — kernel contains predictive "
              "signal; classification performance expected to be competitive.")
    else:
        print("  Interpretation: LOW alignment — consider revising the feature "
              "map or feature subset.")

    return kta_centred


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Compute KTA for AMR quantum kernel')
    parser.add_argument('--antibiotic', required=True, choices=['AMP', 'CIP', 'CTX'])
    parser.add_argument('--K_train',    required=True, help='Path to *_K_train.npy')
    parser.add_argument('--y_train',    required=True, help='Path to *_y_train.npy')
    args = parser.parse_args()

    K_train = np.load(args.K_train)
    y_train = np.load(args.y_train)

    report_kta(K_train, y_train, args.antibiotic)


if __name__ == '__main__':
    main()
