"""
visualize.py
------------
Generates visualizations for the QSVM traffic classification experiment:

  1. t-SNE projection of the quantum kernel matrix (train set)
  2. Venn diagram of feature overlap between classical SVM and QSVM
  3. Comparison bar chart of MI scores (top features)

Usage:
    python visualize.py --K_train data/processed/K_train.npy
                        --y_train data/processed/y_train.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib_venn import venn2
from sklearn.manifold import TSNE


# ── t-SNE kernel projection ───────────────────────────────────────────────────

def plot_kernel_tsne(K_train, y_train, title="Quantum Kernel t-SNE Projection (Train Set)"):
    """Convert fidelity kernel to distance matrix and project with t-SNE."""
    diag = np.diag(K_train)
    dist_sq = np.add.outer(diag, diag) - 2 * K_train
    dist_sq = np.clip(dist_sq, 0, None)   # Numerical safety
    distance_matrix = np.sqrt(dist_sq)

    tsne = TSNE(
        n_components=2, metric='precomputed', perplexity=30,
        n_iter=1000, random_state=42, init='random', learning_rate='auto'
    )
    embedding = tsne.fit_transform(distance_matrix)

    label_names = np.array(["HTTPS (0)" if y == 0 else "SSH (1)" for y in y_train])

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=label_names, style=label_names,
        markers={"HTTPS (0)": "o", "SSH (1)": "X"},
        palette={"HTTPS (0)": "blue", "SSH (1)": "orange"},
        s=80
    )
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Class")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsne_kernel_projection.png", dpi=150)
    plt.show()


# ── Feature overlap Venn diagram ──────────────────────────────────────────────

def plot_feature_venn():
    """Venn diagram of features selected by classical SVM vs QSVM."""
    svm_features = {
        "Average Packet Size", "Packet Length Mean", "Max Packet Length",
        "Flow IAT Max", "Fwd IAT Max", "Fwd Header Length",
        "Fwd Header Length.1", "Subflow Fwd Bytes",
        "Total Length of Fwd Packets", "Init_Win_bytes_backward"
    }
    qsvm_features = {
        'Init_Win_bytes_backward', 'Bwd Packets/s',
        'Flow IAT Max', 'Fwd IAT Max', 'Max Packet Length'
    }

    plt.figure(figsize=(6, 6))
    venn2([svm_features, qsvm_features], set_labels=("Classical SVM", "Quantum SVM"))
    plt.title("Feature Overlap: Classical SVM vs QSVM")
    plt.tight_layout()
    plt.savefig("feature_venn.png", dpi=150)
    plt.show()

    common = svm_features & qsvm_features
    print(f"\nCommon features ({len(common)}): {sorted(common)}")
    print(f"SVM-only ({len(svm_features - qsvm_features)}): {sorted(svm_features - qsvm_features)}")
    print(f"QSVM-only ({len(qsvm_features - svm_features)}): {sorted(qsvm_features - svm_features)}")


# ── MI score bar chart ────────────────────────────────────────────────────────

def plot_mi_scores():
    """Bar chart of top-5 MI scores for classical SVM and QSVM features."""
    features = [
        'Init_Win_bytes_backward', 'Bwd Packets/s',
        'Flow IAT Max', 'Fwd IAT Max', 'Max Packet Length'
    ]
    mi_classical = [0.139356, 0.112138, 0.105325, 0.105387, 0.104704]
    mi_quantum   = [0.139164, 0.112141, 0.105055, 0.105095, 0.104788]

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, mi_classical, width, label='Classical SVM', color='steelblue')
    ax.bar(x + width / 2, mi_quantum,   width, label='Quantum SVM',   color='darkorange')

    ax.set_xlabel("Feature")
    ax.set_ylabel("Mutual Information Score")
    ax.set_title("Top-5 Feature MI Scores: Classical SVM vs QSVM")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=20, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig("mi_scores_comparison.png", dpi=150)
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualizations for QSVM traffic classification")
    parser.add_argument('--K_train', default='data/processed/K_train.npy',
                        help="Path to the quantum training kernel matrix (.npy)")
    parser.add_argument('--y_train', default='data/processed/y_train.npy',
                        help="Path to training labels (.npy)")
    args = parser.parse_args()

    K_train = np.load(args.K_train)
    y_train = np.load(args.y_train)

    print("=== t-SNE Kernel Projection ===")
    plot_kernel_tsne(K_train, y_train)

    print("\n=== Feature Overlap Venn Diagram ===")
    plot_feature_venn()

    print("\n=== MI Score Comparison ===")
    plot_mi_scores()


if __name__ == '__main__':
    main()
