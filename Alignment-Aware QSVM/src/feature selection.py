"""
feature_selection.py
--------------------
Two-step feature selection pipeline for AMR resistance prediction:

  Step 1 — Mutual Information (MI) Ranking:
      Scores every genomic feature by its non-linear dependency with the
      binary resistance label. For CTX, the blaCTX-M-15 gene consistently
      achieves the highest MI score, confirming it as the primary resistance
      driver.

  Step 2 — Correlation Pruning:
      Removes redundant features with pairwise Pearson correlation above a
      threshold, always keeping the feature with the higher MI score.

Usage:
    python feature_selection.py --data_path data/processed/CTX_FeatureMatrix_Standardized.csv
                                --antibiotic CTX --top_k 12
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

DROP_COLS = ['Label', 'Antibiotic', 'BioSample', 'Isolate']
CORR_THRESHOLD = 0.9   # Pearson |r| above this triggers pruning
RANDOM_STATE = 42


# ── Step 1: Mutual Information ─────────────────────────────────────────────────

def compute_mi_scores(X, y, random_state=RANDOM_STATE):
    """Compute MI scores for all features vs. the resistance label."""
    mi = mutual_info_classif(X, y, random_state=random_state)
    return pd.Series(mi, index=X.columns, name='MI_Score').sort_values(ascending=False)


# ── Step 2: Correlation Pruning ────────────────────────────────────────────────

def prune_correlated(X, mi_scores, threshold=CORR_THRESHOLD):
    """
    Drop redundant features whose pairwise |Pearson r| > threshold.
    Between any correlated pair, the feature with lower MI score is dropped.
    """
    corr = X.corr().abs()
    to_drop = set()

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] > threshold:
                f1, f2 = corr.columns[i], corr.columns[j]
                # Keep the more informative feature
                drop = f1 if mi_scores.get(f1, 0) < mi_scores.get(f2, 0) else f2
                to_drop.add(drop)

    print(f"Features pruned due to correlation > {threshold}: {len(to_drop)}")
    return X.drop(columns=list(to_drop)), to_drop


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_correlation_heatmap(X, top_features, antibiotic, out_path=None):
    """Heatmap of pairwise correlations among the final selected features."""
    corr = X[top_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title(f'Correlation Heatmap — Top Features ({antibiotic})')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    plt.show()


def plot_mi_bar(mi_scores, top_k, antibiotic, out_path=None):
    """Bar chart of the top-k MI scores."""
    top = mi_scores.head(top_k)
    plt.figure(figsize=(10, 5))
    top.sort_values().plot(kind='barh', color='steelblue')
    plt.xlabel('Mutual Information Score')
    plt.title(f'Top-{top_k} Features by MI Score ({antibiotic})')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    plt.show()


# ── Class distribution donut ───────────────────────────────────────────────────

def plot_class_donut(y, antibiotic, out_path=None):
    """Donut chart of class distribution (Susceptible vs Resistant)."""
    counts = y.value_counts().sort_index()
    labels = ['Susceptible', 'Resistant']
    sizes  = [counts.get(0, 0), counts.get(1, 0)]
    colors = ['#a5d6a7', '#1b5e20']

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, pctdistance=1.35, labeldistance=1.05,
           wedgeprops=dict(width=0.4, edgecolor='white'))
    ax.set_title(f'Antibiotic: {antibiotic}')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def run_feature_selection(data_path, antibiotic, top_k=12, corr_threshold=CORR_THRESHOLD):
    """Full MI + correlation-pruning pipeline. Returns list of selected feature names."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df['Label'].astype(int)

    print(f"\n=== Feature Selection: {antibiotic} ===")
    print(f"Dataset shape: {X.shape},  class counts: {y.value_counts().to_dict()}")

    mi_scores = compute_mi_scores(X, y)
    print(f"\nTop 10 features by MI:\n{mi_scores.head(10).to_string()}")

    X_filtered, dropped = prune_correlated(X, mi_scores, corr_threshold)
    mi_filtered = mi_scores.drop(index=list(dropped), errors='ignore')

    top_features = mi_filtered.head(top_k).index.tolist()
    print(f"\nFinal selected features ({top_k}):\n{top_features}")

    plot_class_donut(y, antibiotic, out_path=f'donut_{antibiotic}.png')
    plot_mi_bar(mi_filtered, top_k, antibiotic, out_path=f'mi_scores_{antibiotic}.png')
    plot_correlation_heatmap(X, top_features, antibiotic,
                             out_path=f'corr_heatmap_{antibiotic}.png')
    return top_features


def main():
    parser = argparse.ArgumentParser(description='MI + correlation pruning feature selection for AMR QSVM')
    parser.add_argument('--data_path',   required=True, help='Path to *_FeatureMatrix_Standardized.csv')
    parser.add_argument('--antibiotic',  required=True, choices=['AMP', 'CIP', 'CTX'], help='Antibiotic name')
    parser.add_argument('--top_k',       type=int, default=12, help='Number of top features to select (default: 12)')
    parser.add_argument('--corr_threshold', type=float, default=CORR_THRESHOLD,
                        help='Pearson |r| threshold for pruning (default: 0.9)')
    args = parser.parse_args()

    run_feature_selection(
        args.data_path, args.antibiotic, args.top_k, args.corr_threshold
    )


if __name__ == '__main__':
    main()
