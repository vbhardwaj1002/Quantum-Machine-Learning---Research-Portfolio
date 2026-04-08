"""
evaluate.py — Model evaluation: R², RMSE, parity plot, and error distribution.

Computes regression metrics and generates the two key diagnostic plots
for the test set (parity plot and residual histogram).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.sans-serif"] = ["Inter", "Arial", "sans-serif"]


def compute_metrics(y_test, y_pred):
    """
    Compute R² and RMSE.

    Returns
    -------
    metrics : dict  —  {"r2": float, "rmse": float}
    """
    return {
        "r2": r2_score(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


def plot_parity(y_test, y_pred, r2, rmse, save_dir="results/plots"):
    """
    Actual-vs-predicted parity plot with ideal y = x reference line.

    Parameters
    ----------
    y_test, y_pred : np.ndarray
    r2, rmse       : float
    save_dir       : str
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, color="#34A853", alpha=0.8,
                edgecolors="w", linewidths=0.5)

    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="Ideal (y = x)")

    plt.xlabel("Actual Bandgap (eV)", fontsize=12)
    plt.ylabel("Predicted Bandgap (eV)", fontsize=12)
    plt.title("Parity Plot: Actual vs. Predicted Bandgap")
    plt.text(lo, hi * 0.95, f"$R^2 = {r2:.4f}$", fontsize=12, va="top")
    plt.text(lo, hi * 0.90, f"RMSE = {rmse:.4f} eV", fontsize=12, va="top")
    plt.legend()
    plt.axis("square")

    path = os.path.join(save_dir, "parity_plot.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {path}")


def plot_error_distribution(y_test, y_pred, save_dir="results/plots"):
    """
    Histogram of prediction residuals with mean-error marker.

    Parameters
    ----------
    y_test, y_pred : np.ndarray
    save_dir       : str
    """
    os.makedirs(save_dir, exist_ok=True)
    errors = y_pred - y_test

    plt.figure(figsize=(6, 4))
    sns.histplot(errors, kde=True, bins=15, color="orange")
    plt.title("Prediction Error Distribution (Test Set)")
    plt.xlabel("Error (Predicted − Actual, eV)")
    plt.ylabel("Frequency")
    plt.axvline(errors.mean(), color="r", linestyle="--",
                label=f"Mean Error: {errors.mean():.4f}")
    plt.legend()

    path = os.path.join(save_dir, "error_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {path}")
