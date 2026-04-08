"""
compute_kernel.py — Quantum kernel matrix computation.

Builds the Gram matrix K(a, b) = |⟨φ(a)|φ(b)⟩|² by evaluating
the quantum feature map for every pair of input samples.
"""

from pennylane import numpy as np

from src.quantum_feature_map import feature_map


def compute_kernel_matrix(XA, XB, beta=1.0, layers=1):
    """
    Compute the quantum kernel (fidelity) matrix.

    K[i, j] = |⟨feature_map(xA_i) | feature_map(xB_j)⟩|²

    Parameters
    ----------
    XA     : np.ndarray, shape (m, d)
    XB     : np.ndarray, shape (n, d)
    beta   : float  — RY scaling factor.
    layers : int    — number of circuit layers.

    Returns
    -------
    K : np.ndarray, shape (m, n)
    """
    K = np.zeros((XA.shape[0], XB.shape[0]))

    for i, a in enumerate(XA):
        for j, b in enumerate(XB):
            s1 = feature_map(a, beta=beta, layers=layers)
            s2 = feature_map(b, beta=beta, layers=layers)
            K[i, j] = np.abs(np.vdot(s1, s2)) ** 2

    return K
