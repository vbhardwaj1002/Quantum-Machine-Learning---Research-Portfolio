"""
quantum_kernel.py
-----------------
Builds the quantum feature map and computes the fidelity kernel matrices
(train and test) used by the QSVM.

Feature map: StronglyEntanglingLayers with RX, RY, RZ rotations and
cyclic CNOT entanglement across 5 qubits (one per selected feature).

Kernel: K(x1, x2) = |<phi(x1)|phi(x2)>|^2  (fidelity / overlap)

Usage:
    python quantum_kernel.py --X_train data/processed/X_train.npy
                             --X_test  data/processed/X_test.npy
                             --output_dir data/processed
"""

import argparse
import multiprocessing
import os
import numpy as np

import pennylane as qml
from pennylane import numpy as pnp
from joblib import Parallel, delayed

N_LAYERS = 5   # Depth of the StronglyEntanglingLayers feature map


# ── Feature map ───────────────────────────────────────────────────────────────

def feature_map(x, wires, n_layers=N_LAYERS):
    """
    Encode input vector x into a quantum state using StronglyEntanglingLayers.

    Each qubit receives RX, RY, RZ rotations derived from the input features,
    followed by cyclic CNOT entanglement gates between adjacent qubits.
    """
    n_wires = len(wires)
    repeated_x = pnp.tile(x, (n_layers, 1))
    weights = pnp.stack([
        np.tile(repeated_x[i], 3)[:3 * n_wires].reshape(n_wires, 3)
        for i in range(n_layers)
    ])
    for layer in range(n_layers):
        for i in range(n_wires):
            qml.RX(weights[layer, i, 0], wires=wires[i])
            qml.RY(weights[layer, i, 1], wires=wires[i])
            qml.RZ(weights[layer, i, 2], wires=wires[i])
        # Cyclic entanglement
        for i in range(n_wires - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.CNOT(wires=[wires[-1], wires[0]])


# ── Kernel computation ────────────────────────────────────────────────────────

def _build_kernel_circuit(n_wires):
    """Return a PennyLane QNode that computes the fidelity kernel entry."""
    dev = qml.device('lightning.qubit', wires=n_wires)
    wire_list = list(range(n_wires))

    @qml.qnode(dev, diff_method='best', interface=None)
    def kernel_circuit(x1, x2):
        feature_map(x1, wires=wire_list)
        qml.adjoint(feature_map)(x2, wires=wire_list)
        return qml.expval(qml.Projector([0] * n_wires, wires=wire_list))

    return kernel_circuit


def _compute_row(x1, X2, kernel_circuit):
    """Compute one row of the kernel matrix (parallelisation helper)."""
    return [float(kernel_circuit(x1, x2)) for x2 in X2]


def compute_kernel_matrices(X_train, X_test, n_layers=N_LAYERS):
    """
    Compute the train and test quantum kernel matrices in parallel.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    X_test  : np.ndarray, shape (n_test,  n_features)
    n_layers : int, feature map depth

    Returns
    -------
    K_train : np.ndarray, shape (n_train, n_train)
    K_test  : np.ndarray, shape (n_test,  n_train)
    """
    n_wires = X_train.shape[1]
    X_train = pnp.array(X_train, dtype=np.float32)
    X_test = pnp.array(X_test, dtype=np.float32)

    kernel_circuit = _build_kernel_circuit(n_wires)
    n_jobs = multiprocessing.cpu_count()

    print("Computing training kernel matrix...")
    K_train = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_compute_row)(x1, X_train, kernel_circuit) for x1 in X_train
        )
    )
    K_train += np.eye(len(K_train)) * 1e-4   # Numerical regularisation

    print("Computing test kernel matrix...")
    K_test = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_compute_row)(x1, X_train, kernel_circuit) for x1 in X_test
        )
    )

    return K_train, K_test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute quantum kernel matrices for QSVM")
    parser.add_argument('--X_train', required=True, help="Path to X_train.npy")
    parser.add_argument('--X_test', required=True, help="Path to X_test.npy")
    parser.add_argument('--output_dir', default='data/processed',
                        help="Directory to save kernel matrices")
    args = parser.parse_args()

    X_train = np.load(args.X_train)
    X_test = np.load(args.X_test)
    print(f"X_train: {X_train.shape},  X_test: {X_test.shape}")

    K_train, K_test = compute_kernel_matrices(X_train, X_test)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'K_train.npy'), K_train)
    np.save(os.path.join(args.output_dir, 'K_test.npy'), K_test)
    print(f"\nKernel matrices saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
