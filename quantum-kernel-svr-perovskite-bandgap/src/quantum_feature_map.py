"""
quantum_feature_map.py — 3-qubit RY + T-gate + CZ ring-entangling circuit.

Implements a parameterized quantum feature map designed for high expressivity
in kernel-based regression. The T-gate introduces non-Clifford rotations
that increase the effective dimensionality of the feature Hilbert space.

Circuit structure per layer
---------------------------
    H ─ RY(β·xᵢ) ─ T ─ CZ ─
    H ─ RY(β·xᵢ) ─ T ─ CZ ─   (ring connectivity)
    H ─ RY(β·xᵢ) ─ T ─ CZ ─
"""

import pennylane as qml

# ── Circuit constants ─────────────────────────────────────────
N_QUBITS = 3

dev = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def feature_map(x, beta=1.0, layers=1):
    """
    Parameterized quantum feature map.

    Parameters
    ----------
    x      : array-like, shape (N_QUBITS,)
             Input features (phase-encoded in [0, π/2]).
    beta   : float
             Scaling factor applied to RY rotation angles.
    layers : int
             Number of repeated ansatz layers.

    Returns
    -------
    state : array
        Full 2^N_QUBITS statevector.
    """
    # Initial superposition
    for w in range(N_QUBITS):
        qml.Hadamard(wires=w)

    for _ in range(layers):
        # Single-qubit block: RY rotation + T-gate
        for i in range(N_QUBITS):
            qml.RY(beta * x[i], wires=i)
            qml.T(wires=i)

        # Ring-topology CZ entanglement
        for i in range(N_QUBITS - 1):
            qml.CZ(wires=[i, i + 1])
        qml.CZ(wires=[N_QUBITS - 1, 0])

    return qml.state()
