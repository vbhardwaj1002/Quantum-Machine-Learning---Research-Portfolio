"""
evaluate.py
-----------
Loads saved prediction CSVs and produces a comprehensive evaluation
report for each antibiotic (AMP, CIP, CTX):

  - Accuracy, Precision, Recall, F1 (macro and per-class)
  - ROC AUC score
  - Confusion matrix (printed and saved as PNG)
  - ROC curve (saved as PNG)
  - Cross-antibiotic summary table

Usage:
    python evaluate.py --reports_dir reports
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

ANTIBIOTICS = ['AMP', 'CIP', 'CTX']
CLASS_LABELS = ['Susceptible (0)', 'Resistant (1)']


# ── Per-antibiotic plots ───────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, antibiotic, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix — {antibiotic}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_roc_curve(y_true, y_score, antibiotic, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {antibiotic}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate_all(reports_dir):
    summary_rows = []

    for abx in ANTIBIOTICS:
        pred_path = os.path.join(reports_dir, f'{abx}_predictions.csv')
        if not os.path.exists(pred_path):
            print(f"Predictions not found for {abx}: {pred_path} — skipping")
            continue

        df = pd.read_csv(pred_path)
        y_true  = df['True'].values
        y_pred  = df['Predicted'].values
        y_score = df['Score'].values

        print(f"\n{'='*50}")
        print(f"  {abx}")
        print(f"{'='*50}")

        acc     = accuracy_score(y_true, y_pred)
        auc_val = roc_auc_score(y_true, y_score)
        report  = classification_report(y_true, y_pred, output_dict=True)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  AUC      : {auc_val:.4f}")
        print(f"\n{classification_report(y_true, y_pred)}")

        plot_confusion_matrix(
            y_true, y_pred, abx,
            out_path=os.path.join(reports_dir, f'confusion_matrix_{abx}.png')
        )
        plot_roc_curve(
            y_true, y_score, abx,
            out_path=os.path.join(reports_dir, f'roc_curve_{abx}.png')
        )

        summary_rows.append({
            'Antibiotic': abx,
            'Accuracy':   round(acc, 4),
            'AUC':        round(auc_val, 4),
            'Precision (macro)': round(report['macro avg']['precision'], 4),
            'Recall (macro)':    round(report['macro avg']['recall'],    4),
            'F1 (macro)':        round(report['macro avg']['f1-score'],  4),
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).set_index('Antibiotic')
        summary_path = os.path.join(reports_dir, 'evaluation_summary.csv')
        summary_df.to_csv(summary_path)
        print(f"\n{'='*50}")
        print("  Cross-Antibiotic Summary")
        print(f"{'='*50}")
        print(summary_df.to_string())
        print(f"\nSummary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate QSVM AMR predictions')
    parser.add_argument('--reports_dir', default='reports',
                        help='Directory containing *_predictions.csv files')
    args = parser.parse_args()
    evaluate_all(args.reports_dir)


if __name__ == '__main__':
    main()
