"""
visualize.py — Correlation heatmap, KTA bar chart, circuit schematic,
               target distribution, and prediction CSV export.

Generates the remaining publication figures that are not part of
the core evaluation diagnostics (see evaluate.py for parity plot
and error distribution).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pennylane as qml

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.sans-serif"] = ["Inter", "Arial", "sans-serif"]


# ── Individual plot functions ─────────────────────────────────

def plot_correlation_heatmap(df, features, target="BG", save_dir="results/plots"):
    """
    Feature-target Pearson correlation heatmap.

    Parameters
    ----------
    df       : pd.DataFrame
    features : list[str]   — columns to include.
    target   : str         — target column name.
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 7))
    corr = df[features + [target]].corr()
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "Pearson Correlation"})
    plt.title("Feature Correlation Heatmap")

    path = os.path.join(save_dir, "correlation_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {path}")


def plot_target_distribution(df, target="BG", save_dir="results/plots"):
    """Histogram + KDE of the bandgap target distribution."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.histplot(df[target], kde=True, bins=15, color="teal")
    plt.title("Bandgap Distribution in Dataset")
    plt.xlabel("Bandgap (eV)")
    plt.ylabel("Frequency")

    path = os.path.join(save_dir, "bandgap_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {path}")


def plot_circuit_schematic(feature_map_fn, sample_x, beta, layers,
                           save_dir="results/plots"):
    """
    Render the quantum feature map using PennyLane's matplotlib drawer.

    Falls back to ASCII circuit drawing if matplotlib rendering fails.
    """
    os.makedirs(save_dir, exist_ok=True)

    try:
        fig, ax = qml.draw_mpl(feature_map_fn)(sample_x, beta=beta, layers=layers)
        ax.set_title(r"Quantum Feature Map ($\phi_{\mathrm{QKE}}$)", fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, "quantum_circuit_schematic.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Saved: {path}")
    except Exception as e:
        print(f"  Warning: matplotlib circuit draw failed ({e}). ASCII fallback:")
        print(qml.draw(feature_map_fn)(sample_x, beta=beta, layers=layers))


def plot_kta_comparison(kta_optimal, kta_baseline, save_dir="results/plots"):
    """
    Bar chart comparing KTA scores for the optimal vs. MI-selected feature sets.

    Parameters
    ----------
    kta_optimal  : float — KTA for the best subset (Cs, I, t).
    kta_baseline : float — KTA for the MI-selected subset (Cl, Br, I).
    save_dir     : str
    """
    os.makedirs(save_dir, exist_ok=True)

    data = pd.DataFrame({
        "Feature Set": ["Optimal (Cs, I, t)", "MI-Selected (Cl, Br, I)"],
        "KTA Score": [kta_optimal, kta_baseline],
    })

    plt.figure(figsize=(6, 4))
    sns.barplot(x="Feature Set", y="KTA Score", data=data,
                palette=["#1f77b4", "#ff7f0e"])
    plt.title("Kernel-Target Alignment (KTA) Comparison")
    plt.ylim(0.70, 0.90)
    plt.ylabel("Centered KTA Score")

    path = os.path.join(save_dir, "kta_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {path}")


def export_predictions(y_test, y_pred, save_dir="results/predictions"):
    """Save actual vs. predicted bandgaps and errors to CSV."""
    os.makedirs(save_dir, exist_ok=True)

    results = pd.DataFrame({
        "Actual_Bandgap_eV": y_test,
        "Predicted_Bandgap_eV": y_pred,
        "Error_eV": y_pred - y_test,
    })

    path = os.path.join(save_dir, "bandgap_predictions.csv")
    results.to_csv(path, index=False)
    print(f"  -> Saved: {path}")
    return results
