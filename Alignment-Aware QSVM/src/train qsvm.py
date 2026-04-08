"""
train_qsvm.py
-------------
Trains and evaluates the Quantum SVM for one antibiotic (AMP, CIP, or CTX)
using a precomputed quantum kernel matrix.

GridSearchCV over the regularisation parameter C selects the best model
via 5-fold cross-validation. The best estimator is then evaluated on the
held-out test set.

C search grids differ by antibiotic to reflect the scale of kernel values:
  AMP  : [0.1, 1, 10, 100]          (broader search, severe imbalance)
  CIP/CTX : [0.01, 0.05, 0.1, 0.5, 1.0]

Usage:
    python train_qsvm.py --antibiotic AMP
                         --K_train data/processed/AMP_K_train.npy
                         --K_test  data/processed/AMP_K_test.npy
                         --y_train data/processed/AMP_y_train.npy
                         --y_test  data/processed/AMP_y_test.npy
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from kta_alignment import report_kta

# Per-antibiotic C grids (reflect the paper's experimental setup)
C_GRIDS = {
    'AMP': {'C': [0.1, 1, 10, 100]},
    'CIP': {'C': [0.01, 0.05, 0.1, 0.5, 1.0]},
    'CTX': {'C': [0.01, 0.05, 0.1, 0.5, 1.0]},
}


# ── Training ───────────────────────────────────────────────────────────────────

def train_qsvm(K_train, y_train, antibiotic):
    """
    GridSearchCV over C for SVC with a precomputed quantum kernel.
    Returns the best fitted estimator.
    """
    param_grid = C_GRIDS.get(antibiotic, {'C': [0.1, 1, 10]})
    print(f"\nGridSearchCV — searching C in {param_grid['C']}...")

    grid = GridSearchCV(
        SVC(kernel='precomputed'),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
    )
    grid.fit(K_train, y_train)

    print(f"Best params : {grid.best_params_}")
    print(f"Best CV acc : {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_score_


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_qsvm(model, K_train, y_train, K_test, y_test, antibiotic, kta_score, cv_acc):
    """Evaluate model on train and test sets, print full summary."""
    y_train_pred = model.predict(K_train)
    y_test_pred  = model.predict(K_test)
    y_test_score = model.decision_function(K_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test,  y_test_pred)
    auc_score = roc_auc_score(y_test, y_test_score)

    print(f"\n{'='*50}")
    print(f"  QSVM Evaluation Summary: {antibiotic}")
    print(f"{'='*50}")
    print(f"  KTA score          : {kta_score:.4f}")
    print(f"  Best C             : {model.C}")
    print(f"  Cross-val accuracy : {cv_acc:.4f}")
    print(f"  Train accuracy     : {train_acc:.4f}")
    print(f"  Test  accuracy     : {test_acc:.4f}")
    print(f"  Test  AUC          : {auc_score:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")

    return y_test_pred, y_test_score, test_acc, auc_score


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train QSVM for AMR resistance prediction')
    parser.add_argument('--antibiotic', required=True, choices=['AMP', 'CIP', 'CTX'])
    parser.add_argument('--K_train',    required=True)
    parser.add_argument('--K_test',     required=True)
    parser.add_argument('--y_train',    required=True)
    parser.add_argument('--y_test',     required=True)
    parser.add_argument('--output_dir', default='reports')
    args = parser.parse_args()

    K_train = np.load(args.K_train)
    K_test  = np.load(args.K_test)
    y_train = np.load(args.y_train)
    y_test  = np.load(args.y_test)

    kta_score = report_kta(K_train, y_train, args.antibiotic)

    model, cv_acc = train_qsvm(K_train, y_train, args.antibiotic)

    y_pred, y_score, test_acc, auc_score = evaluate_qsvm(
        model, K_train, y_train, K_test, y_test,
        args.antibiotic, kta_score, cv_acc
    )

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, f'{args.antibiotic}_predictions.csv')
    pd.DataFrame({
        'True': y_test,
        'Predicted': y_pred,
        'Score': y_score
    }).to_csv(pred_path, index=False)
    print(f"\nPredictions saved: {pred_path}")


if __name__ == '__main__':
    main()
