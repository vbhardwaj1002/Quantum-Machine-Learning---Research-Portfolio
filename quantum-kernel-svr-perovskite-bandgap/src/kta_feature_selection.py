"""
kta_feature_selection.py — Exhaustive KTA search over all 3-feature subsets.

Evaluates every C(n, 3) combination of the initial feature pool using
the centered Kernel-Target Alignment (KTA) metric and returns the
subset that maximizes alignment between the quantum kernel and the
regression target.
"""

from itertools import combinations
from pennylane import numpy as np
from numpy.linalg import norm

from src.preprocess import (
    load_dataset, split_and_scale, phase_normalize, INITIAL_FEATURES
)
from src.compute_kernel import compute_kernel_matrix


# ── KTA utilities ─────────────────────────────────────────────

def center_kernel(K):
    """Apply double-centering to kernel matrix K."""
    n = K.shape[0]
    one = np.ones((n, n)) / n
    return K - one @ K - K @ one + one @ K @ one


def kernel_target_alignment(K, y):
    """
    Centered Kernel-Target Alignment for regression.

    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Uncentered training kernel matrix.
    y : np.ndarray, shape (n,)
        Target values.

    Returns
    -------
    kta : float
        Alignment score in [−1, 1]; higher is better.
    """
    Kc = center_kernel(K)
    y_c = (y - np.mean(y)) / np.std(y)
    num = y_c.T @ Kc @ y_c
    den = norm(Kc, "fro") * (norm(y_c) ** 2)
    return float(num / den)


# ── Exhaustive search ─────────────────────────────────────────

def exhaustive_kta_search(
    dataset_path="data/perovskite_bandgap.csv",
    n_select=3,
    beta=0.40,
    layers=1,
):
    """
    Evaluate KTA for every C(len(INITIAL_FEATURES), n_select) feature subset.

    Parameters
    ----------
    dataset_path : str
    n_select     : int   — number of features per subset (default 3).
    beta         : float — RY scaling factor.
    layers       : int   — circuit depth.

    Returns
    -------
    results : list[dict]
        Sorted (descending KTA) list of {"features", "kta"} records.
    """
    df, y = load_dataset(dataset_path)
    combos = list(combinations(INITIAL_FEATURES, n_select))
    results = []

    print(f"Evaluating {len(combos)} feature subsets (n_select={n_select}) …\n")

    for idx, feat_set in enumerate(combos, 1):
        feat_list = list(feat_set)

        X_tr_s, _, y_tr, _ = split_and_scale(df, y, features=feat_list)
        X_tr_q, _ = phase_normalize(X_tr_s, X_tr_s)  # only need train for KTA

        K = compute_kernel_matrix(X_tr_q, X_tr_q, beta=beta, layers=layers)
        kta = kernel_target_alignment(K, y_tr)

        results.append({"features": feat_list, "kta": kta})
        print(f"  [{idx}/{len(combos)}] {feat_list} → KTA = {kta:.4f}")

    results.sort(key=lambda r: r["kta"], reverse=True)

    print(f"\nBest subset: {results[0]['features']}  (KTA = {results[0]['kta']:.4f})")
    return results


# ── CLI entry point ───────────────────────────────────────────

if __name__ == "__main__":
    exhaustive_kta_search()
