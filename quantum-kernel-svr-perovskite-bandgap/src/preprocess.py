"""
preprocess.py — Standardization and phase-friendly normalization for quantum encoding.

Handles dataset loading, train/test splitting, StandardScaler fitting,
and final normalization into [0, π/2] suitable for RY-gate angle encoding.
"""

import pandas as pd
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Dataset configuration ────────────────────────────────────────
INITIAL_FEATURES = ["Cs", "FA", "MA", "Cl", "Br", "I", "t"]
SELECTED_FEATURES = ["Cs", "I", "t"]
TARGET_COLUMN = "BG"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(path="data/perovskite_bandgap.csv"):
    """
    Load the perovskite bandgap CSV.

    Falls back to synthetic mock data when the file is absent,
    enabling the full pipeline to run without the real dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    df : pd.DataFrame
        Full dataframe containing at least INITIAL_FEATURES and TARGET_COLUMN.
    y  : np.ndarray
        Bandgap target values (eV).
    """
    try:
        df = pd.read_csv(path)
        y = df[TARGET_COLUMN].values
        print(f"Loaded dataset from '{path}' — {len(df)} samples.")
    except FileNotFoundError:
        print(f"Warning: '{path}' not found. Generating synthetic mock data.")
        n_samples = 124
        X_mock = np.random.rand(n_samples, len(INITIAL_FEATURES))
        df = pd.DataFrame(X_mock, columns=INITIAL_FEATURES)
        y = (
            1.5
            + 0.5 * df["Cs"]
            - 0.2 * df["MA"]
            + 0.8 * df["Cs"] * df["MA"]
            + np.random.randn(n_samples) * 0.05
        )
        y = np.clip(y, 1.5, 3.0)
        df[TARGET_COLUMN] = y

    return df, y


def split_and_scale(df, y, features=SELECTED_FEATURES):
    """
    Train/test split followed by StandardScaler fitting.

    Parameters
    ----------
    df       : pd.DataFrame
    y        : np.ndarray
    features : list[str]

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
        Standardized feature matrices.
    y_train, y_test : np.ndarray
    """
    X = df[features].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_scaled, X_test_scaled, y_train, y_test


def phase_normalize(X_train_scaled, X_test_scaled):
    """
    Normalize standardized features into [0, π/2] for quantum angle encoding.

    Maps the training-set min/max to the [0, π/2] interval and applies
    the same transformation to the test set.

    Parameters
    ----------
    X_train_scaled, X_test_scaled : np.ndarray

    Returns
    -------
    X_train_q, X_test_q : np.ndarray
        Phase-normalized feature matrices ready for the quantum circuit.
    """
    eps = 1e-8
    X_min = X_train_scaled.min(axis=0)
    X_max = X_train_scaled.max(axis=0)

    X_train_q = (X_train_scaled - X_min) / (X_max - X_min + eps) * np.pi / 2
    X_test_q = (X_test_scaled - X_min) / (X_max - X_min + eps) * np.pi / 2

    return X_train_q, X_test_q
