"""
train_qsvm.py
-------------
Trains the Quantum Support Vector Machine (QSVM) using a precomputed
quantum kernel matrix and evaluates it on the test set.

The quantum kernel is computed by quantum_kernel.py and stored as .npy
files. This script loads those matrices, runs GridSearchCV for the
classical SVC hyperparameter C, and produces evaluation metrics and plots.

Usage:
    python train_qsvm.py --K_train data/processed/K_train.npy
                         --K_test  data/processed/K_test.npy
                         --y_train data/processed/y_train.npy
                         --y_test  data/processed/y_test.npy
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_curve, auc
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# ── KTA helpers ───────────────────────────────────────────────────────────────

def _center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def compute_kta(K, y):
    """Centered Kernel-Target Alignment (KTA) score."""
    yyT = np.outer(y, y)
    Kc = _center(K)
    yyTc = _center(yyT)
    return np.sum(Kc * yyTc) / (np.linalg.norm(Kc) * np.linalg.norm(yyTc))


# ── Training ──────────────────────────────────────────────────────────────────

def train_qsvm(K_train, y_train):
    """GridSearchCV over C for SVC with a precomputed quantum kernel."""
    param_grid = {
        'C': np.linspace(500, 2000, 10),
        'class_weight': [None, 'balanced']
    }
    model = GridSearchCV(
        SVC(kernel='precomputed', probability=True),
        param_grid, cv=5, scoring='f1_weighted'
    )
    model.fit(K_train, y_train)
    print(f"Best parameters: {model.best_params_}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, K_train, y_train, K_test, y_test):
    """Evaluate on both train and test sets, print metrics."""
    y_train_pred = model.predict(K_train)
    y_test_pred  = model.predict(K_test)
    y_test_prob  = model.predict_proba(K_test)[:, 1]

    print("\n=== Train Set ===")
    print(f"Accuracy : {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"F1 Score : {f1_score(y_train, y_train_pred):.4f}")

    print("\n=== Test Set ===")
    print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_test_pred):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")

    return y_test_pred, y_test_prob


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_roc(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"QSVM (Test AUC = {roc_auc:.2f})",
             color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — QSVM (Test Set)")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_qsvm.png", dpi=150)
    plt.show()
    print(f"ROC AUC: {roc_auc:.4f}")


def plot_kernel_matrix(K, title="Quantum Kernel Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(K, cmap='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig("kernel_matrix.png", dpi=150)
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train QSVM with precomputed quantum kernel")
    parser.add_argument('--K_train', default='data/processed/K_train.npy')
    parser.add_argument('--K_test',  default='data/processed/K_test.npy')
    parser.add_argument('--y_train', default='data/processed/y_train.npy')
    parser.add_argument('--y_test',  default='data/processed/y_test.npy')
    args = parser.parse_args()

    K_train = np.load(args.K_train)
    K_test  = np.load(args.K_test)
    y_train = np.load(args.y_train)
    y_test  = np.load(args.y_test)

    # Report KTA before training
    kta_raw = compute_kta(K_train, y_train)
    kta_cen = compute_kta(K_train, 2 * y_train - 1)
    print(f"Raw KTA      : {kta_raw:.4f}")
    print(f"Centered KTA : {kta_cen:.4f}")

    print("\n=== Training QSVM ===")
    model = train_qsvm(K_train, y_train)

    y_pred, y_prob = evaluate(model, K_train, y_train, K_test, y_test)

    plot_kernel_matrix(K_train, "Quantum Kernel Matrix (Train)")
    plot_roc(y_test, y_prob)

    os.makedirs('reports', exist_ok=True)
    pd.DataFrame({'True': y_test, 'Predicted': y_pred, 'Score': y_prob})\
      .to_csv('reports/qsvm_predictions.csv', index=False)
    print("\nPredictions saved to reports/qsvm_predictions.csv")


if __name__ == '__main__':
    main()
