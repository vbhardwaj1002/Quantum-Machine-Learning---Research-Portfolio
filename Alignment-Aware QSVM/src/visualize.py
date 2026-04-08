"""
visualize.py
------------
Produces publication-ready visualisations for the AMR QSVM paper:

  1. Class distribution bar chart across all three antibiotics
  2. Quantum kernel matrix heatmap (K_train) per antibiotic
  3. KTA score comparison bar chart (AMP vs CIP vs CTX)
  4. Cross-antibiotic performance comparison bar chart

Usage:
    python visualize.py --data_dir  data/processed
                        --kernel_dir data/processed
                        --reports_dir reports
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ANTIBIOTICS = ['AMP', 'CIP', 'CTX']


# ── 1. Class distribution ──────────────────────────────────────────────────────

def plot_class_distribution(data_dir, out_path=None):
    """Grouped bar chart of susceptible / resistant counts per antibiotic."""
    frames = []
    for abx in ANTIBIOTICS:
        path = os.path.join(data_dir, f'{abx}_FeatureMatrix_Standardized.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=['Label'])
        df['Antibiotic'] = abx
        frames.append(df)

    if not frames:
        print("No standardised feature matrices found — skipping class distribution plot.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    plot_df = (
        df_all.groupby(['Antibiotic', 'Label'])
              .size()
              .reset_index(name='Count')
    )
    plot_df['Label'] = plot_df['Label'].map({0: 'Susceptible', 1: 'Resistant'})

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x='Antibiotic', y='Count', hue='Label',
                palette=['skyblue', 'salmon'])
    plt.title('Class Distribution per Antibiotic (after SMOTE+ENN)')
    plt.xlabel('Antibiotic')
    plt.ylabel('Number of Samples')
    plt.legend(title='Phenotype')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_or_show(out_path)


# ── 2. Kernel matrix heatmap ───────────────────────────────────────────────────

def plot_kernel_heatmap(kernel_dir, antibiotic, out_path=None):
    """Heatmap of the quantum training kernel matrix for one antibiotic."""
    k_path = os.path.join(kernel_dir, f'{antibiotic}_K_train.npy')
    if not os.path.exists(k_path):
        print(f"Kernel matrix not found: {k_path}")
        return

    K = np.load(k_path)
    plt.figure(figsize=(7, 5))
    sns.heatmap(K, cmap='viridis', vmin=0, vmax=1,
                xticklabels=False, yticklabels=False)
    plt.title(f'Quantum Kernel Matrix K_train — {antibiotic}')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    _save_or_show(out_path)


# ── 3. KTA score comparison ────────────────────────────────────────────────────

def plot_kta_comparison(kta_scores, out_path=None):
    """
    Bar chart comparing centred KTA scores across all three antibiotics.

    kta_scores : dict  e.g. {'AMP': 0.9434, 'CIP': 0.59, 'CTX': 0.60}
    """
    abx_list = list(kta_scores.keys())
    scores   = [kta_scores[a] for a in abx_list]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(abx_list, scores, color=['#1565C0', '#6A1B9A', '#2E7D32'],
                   width=0.5, edgecolor='black')
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01, f'{score:.4f}',
                 ha='center', va='bottom', fontsize=10)
    plt.ylim(0, max(scores) * 1.2)
    plt.xlabel('Antibiotic')
    plt.ylabel('Centred KTA Score')
    plt.title('Kernel-Target Alignment Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_or_show(out_path)


# ── 4. Performance comparison ──────────────────────────────────────────────────

def plot_performance_comparison(reports_dir, out_path=None):
    """Grouped bar chart of Accuracy and AUC across antibiotics."""
    summary_path = os.path.join(reports_dir, 'evaluation_summary.csv')
    if not os.path.exists(summary_path):
        print(f"Summary not found: {summary_path} — run evaluate.py first.")
        return

    df = pd.read_csv(summary_path, index_col='Antibiotic')
    metrics = ['Accuracy', 'AUC', 'F1 (macro)']
    df[metrics].plot(kind='bar', figsize=(8, 5), colormap='Set2', edgecolor='black')
    plt.ylim(0.5, 1.05)
    plt.title('QSVM Performance Comparison across Antibiotics')
    plt.xlabel('Antibiotic')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_or_show(out_path)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _save_or_show(path):
    if path:
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualisations for AMR QSVM experiment')
    parser.add_argument('--data_dir',    default='data/processed')
    parser.add_argument('--kernel_dir',  default='data/processed')
    parser.add_argument('--reports_dir', default='reports')
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)

    print("=== Class Distribution ===")
    plot_class_distribution(
        args.data_dir,
        out_path=os.path.join(args.reports_dir, 'class_distribution.png')
    )

    for abx in ANTIBIOTICS:
        print(f"\n=== Kernel Heatmap: {abx} ===")
        plot_kernel_heatmap(
            args.kernel_dir, abx,
            out_path=os.path.join(args.reports_dir, f'kernel_heatmap_{abx}.png')
        )

    # KTA scores from the paper (update with your computed values)
    print("\n=== KTA Comparison ===")
    kta_scores = {'AMP': 0.9434, 'CIP': 0.59, 'CTX': 0.60}
    plot_kta_comparison(
        kta_scores,
        out_path=os.path.join(args.reports_dir, 'kta_comparison.png')
    )

    print("\n=== Performance Comparison ===")
    plot_performance_comparison(
        args.reports_dir,
        out_path=os.path.join(args.reports_dir, 'performance_comparison.png')
    )


if __name__ == '__main__':
    main()
