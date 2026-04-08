"""
classical_svm.py
----------------
Trains a classical SVM (RBF kernel) baseline for HTTPS vs SSH traffic
classification and evaluates it with cross-validation and SHAP analysis.

Usage:
    python classical_svm.py --X_train data/processed/X_train.npy
                            --X_test  data/processed/X_test.npy
                            --y_train data/processed/y_train.npy
                            --y_test  data/processed/y_test.npy
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

FEATURE_NAMES = [
    'Init_Win_bytes_backward',
    'Flow IAT Max',
    'Fwd IAT Max',
    'Max Packet Length',
    'Bwd Packets/s'
]


# ── Training ──────────────────────────────────────────────────────────────────

def train_svm(X_train, y_train, X_test, y_test, c_values=(0.1, 1, 10, 100)):
    """Train SVM across candidate C values, return best model."""
    results = []
    for C in c_values:
        model = SVC(C=C, gamma='scale', kernel='rbf',
                    class_weight='balanced', probability=True)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        test_f1 = classification_report(
            y_test, model.predict(X_test), output_dict=True
        )['weighted avg']['f1-score']

        results.append({'C': C, 'Train Acc': train_acc,
                        'Test Acc': test_acc, 'Test F1': test_f1, 'model': model})
        print(f"  C={C:6} | Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    df_res = pd.DataFrame(results).sort_values('Test F1', ascending=False)
    print("\nBest configuration:")
    print(df_res[['C', 'Train Acc', 'Test Acc', 'Test F1']].iloc[0])
    return df_res.iloc[0]['model']


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(X, y, C=100, cv_folds=7):
    """Stratified k-fold cross-validation on the training data."""
    model = SVC(C=C, gamma='scale', class_weight='balanced',
                kernel='rbf', probability=True)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    print(f"\n{cv_folds}-fold CV — Accuracy: {acc.mean():.4f} ± {acc.std():.4f} | "
          f"F1 macro: {f1.mean():.4f} ± {f1.std():.4f}")
    return acc, f1


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    """Print accuracy, confusion matrix, and classification report."""
    y_pred = model.predict(X_test)
    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    return y_pred


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_roc(y_test, y_prob, label="Classical SVM"):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})", color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Classical SVM")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_classical_svm.png", dpi=150)
    plt.show()
    print(f"ROC AUC: {roc_auc:.4f}")


def explain_with_shap(model, X_train, X_test, feature_names,
                      n_train=100, n_test=50):
    """SHAP KernelExplainer summary plot for the classical SVM."""
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    X_bg = X_train_df.sample(n=min(n_train, len(X_train_df)), random_state=42)
    X_explain = X_test_df.sample(n=min(n_test, len(X_test_df)), random_state=42)

    explainer = shap.KernelExplainer(
        lambda X: model.predict_proba(X)[:, 1], X_bg
    )
    shap_values = explainer.shap_values(X_explain)

    print("\nSHAP Summary Plot (SSH class probability):")
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate classical SVM baseline")
    parser.add_argument('--X_train', default='data/processed/X_train.npy')
    parser.add_argument('--X_test',  default='data/processed/X_test.npy')
    parser.add_argument('--y_train', default='data/processed/y_train.npy')
    parser.add_argument('--y_test',  default='data/processed/y_test.npy')
    args = parser.parse_args()

    X_train = np.load(args.X_train)
    X_test  = np.load(args.X_test)
    y_train = np.load(args.y_train)
    y_test  = np.load(args.y_test)

    print("=== Training Classical SVM ===")
    model = train_svm(X_train, y_train, X_test, y_test)

    print("\n=== Cross-validation ===")
    cross_validate(X_train, y_train)

    y_pred = evaluate(model, X_test, y_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    plot_roc(y_test, y_prob)
    explain_with_shap(model, X_train, X_test, FEATURE_NAMES)

    os.makedirs('reports', exist_ok=True)
    pd.DataFrame({'True': y_test, 'Predicted': y_pred, 'Prob_SSH': y_prob})\
      .to_csv('reports/classical_svm_predictions.csv', index=False)
    print("\nPredictions saved to reports/classical_svm_predictions.csv")


if __name__ == '__main__':
    main()
