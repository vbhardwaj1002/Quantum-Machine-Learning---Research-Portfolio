"""
quantum_kernel.py
-----------------
Fidelity-based quantum kernel computation for AMR resistance prediction.

Feature map (AMP / CIP):
    AngleEmbedding (Ry rotations per qubit) followed by a single layer
    of CZ entangling gates between adjacent qubits.

Feature map (CTX):
    AngleEmbedding followed by one layer of StronglyEntanglingLayers
    (fixed random weights, seed=42) for higher expressivity needed by
    the larger 12-feature CTX circuit.

Kernel entry: K(x1, x2) = P(|00...0>) from the circuit
    U(x1) U†(x2) |0...0>

Kernel matrices are computed in parallel using joblib.

Usage:
    python quantum_kernel.py --antibiotic AMP
                             --X_train data/processed/AMP_X_train.npy
                             --X_test  data/processed/AMP_X_test.npy
                             --output_dir data/processed
"""

import argparse
import multiprocessing
import os

import numpy as np
import pennylane as qml
from joblib import Parallel, delayed
from pennylane import numpy as pnp

RANDOM_STATE = 42


# ── Feature maps ───────────────────────────────────────────────────────────────

def feature_map_angle_cz(x, n_qubits):
    """
    AngleEmbedding + CZ entanglement.
    Used for AMP (8 qubits) and CIP (6 qubits).
    """
    qml.AngleEmbedding(x, wires=range(n_qubits))
    for i in range(n_qubits - 1):
        qml.CZ(wires=[i, i + 1])


def feature_map_strongly_entangling(x, n_qubits, fixed_weights):
    """
    AngleEmbedding + StronglyEntanglingLayers (fixed weights).
    Used for CTX (12 qubits) to achieve higher circuit expressivity.
    Fixed random weights ensure the kernel is deterministic across runs.
    """
    qml.AngleEmbedding(x, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(fixed_weights, wires=range(n_qubits))


# ── Kernel circuit builder ─────────────────────────────────────────────────────

def build_kernel_circuit(n_qubits, antibiotic):
    """
    Return a QNode that computes one kernel entry K(x1, x2).

    For CTX we pre-generate fixed random weights once so parallel workers
    all use the same circuit (joblib serialises the closure).
    """
    dev = qml.device('lightning.qubit', wires=n_qubits)

    if antibiotic == 'CTX':
        np.random.seed(RANDOM_STATE)
        weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n_qubits)
        fixed_weights = pnp.array(np.random.random(weights_shape), requires_grad=False)

        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            feature_map_strongly_entangling(x1, n_qubits, fixed_weights)
            qml.adjoint(feature_map_strongly_entangling)(x2, n_qubits, fixed_weights)
            return qml.probs(wires=range(n_qubits))
    else:
        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            feature_map_angle_cz(x1, n_qubits)
            qml.adjoint(feature_map_angle_cz)(x2, n_qubits)
            return qml.probs(wires=range(n_qubits))

    def quantum_kernel(x1, x2):
        """K(x1, x2) = probability of measuring the all-zero state."""
        return float(kernel_circuit(x1, x2)[0])

    return quantum_kernel


# ── Parallel kernel matrix ─────────────────────────────────────────────────────

def _entry(i, j, XA, XB, kernel_fn):
    return i, j, kernel_fn(XA[i], XB[j])


def compute_kernel_matrix(XA, XB, kernel_fn, n_jobs=-1):
    """
    Compute the full kernel matrix K[i,j] = kernel_fn(XA[i], XB[j]) in parallel.

    Parameters
    ----------
    XA : np.ndarray, shape (nA, n_features)
    XB : np.ndarray, shape (nB, n_features)
    kernel_fn : callable — quantum kernel function
    n_jobs : int — number of parallel workers (-1 = all CPUs)

    Returns
    -------
    K : np.ndarray, shape (nA, nB)
    """
    nA, nB = len(XA), len(XB)
    n_workers = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f"  Computing kernel ({nA}x{nB}) using {n_workers} workers...")

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(_entry)(i, j, XA, XB, kernel_fn)
        for i in range(nA)
        for j in range(nB)
    )

    K = np.zeros((nA, nB))
    for i, j, val in results:
        K[i, j] = val
    return K


def print_kernel_diagnostics(K, name='K_train'):
    print(f"\n=== Kernel Diagnostics: {name} ===")
    print(f"  min={np.min(K):.4f}  max={np.max(K):.4f}  "
          f"mean={np.mean(K):.4f}  var={np.var(K):.4f}  "
          f"NaNs={np.isnan(K).any()}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Compute quantum kernel matrices for AMR QSVM')
    parser.add_argument('--antibiotic',  required=True, choices=['AMP', 'CIP', 'CTX'])
    parser.add_argument('--X_train',     required=True, help='Path to X_train .npy file')
    parser.add_argument('--X_test',      required=True, help='Path to X_test  .npy file')
    parser.add_argument('--output_dir',  default='data/processed')
    parser.add_argument('--n_jobs',      type=int, default=-1)
    args = parser.parse_args()

    X_train = np.load(args.X_train)
    X_test  = np.load(args.X_test)
    n_qubits = X_train.shape[1]

    print(f"\nAntibiotic : {args.antibiotic}")
    print(f"Qubits     : {n_qubits}")
    print(f"X_train    : {X_train.shape},  X_test : {X_test.shape}")

    kernel_fn = build_kernel_circuit(n_qubits, args.antibiotic)

    print("\nComputing K_train...")
    K_train = compute_kernel_matrix(X_train, X_train, kernel_fn, args.n_jobs)
    print_kernel_diagnostics(K_train, 'K_train')

    print("\nComputing K_test...")
    K_test = compute_kernel_matrix(X_test, X_train, kernel_fn, args.n_jobs)
    print_kernel_diagnostics(K_test, 'K_test')

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f'{args.antibiotic}_K_train.npy'), K_train)
    np.save(os.path.join(args.output_dir, f'{args.antibiotic}_K_test.npy'),  K_test)
    print(f"\nKernel matrices saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
