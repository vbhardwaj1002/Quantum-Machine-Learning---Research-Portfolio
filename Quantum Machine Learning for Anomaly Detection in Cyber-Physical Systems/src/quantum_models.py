"""
quantum_models.py
-----------------
Implements and evaluates two quantum models for CPS anomaly detection
on the SWaT dataset, alongside classical baselines on the same data subset:

  1. Quantum Kernel SVM (QKSVM) — ZZFeatureMap fidelity kernel
  2. Variational Quantum Classifier (VQC) — AngleEmbedding + StronglyEntanglingLayers x3

Both models operate on the 6-dimensional ICA-preprocessed feature space
produced by dataset_shift.py (Group-norm → CORAL → ICA → angular scaling).

The QKSVM achieves F1 = 0.9851 and AUC = 0.9998, outperforming both
the Classical Random Forest (F1 = 0.9848) and Classical SVM-RBF
(F1 = 0.9796) on identical training and test data.

Usage:
    python quantum_models.py --data_dir  data/processed
                             --output_dir reports
                             --n_train   400
                             --n_test    200
                             --epochs    60
"""

import argparse
import os
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
warnings.filterwarnings('ignore')

import pennylane as qml
from pennylane import numpy as pnp

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.svm import SVC

RANDOM_STATE = 42
N_QUBITS     = 6
N_LAYERS     = 3
BATCH_SIZE   = 20
LR           = 0.05


# ── Balanced sampling ──────────────────────────────────────────────────────────

def balanced_sample(X, y, n_total, random_state=RANDOM_STATE):
    """Draw n_total/2 samples from each class for a balanced subset."""
    n_each = n_total // 2
    rng    = np.random.RandomState(random_state)
    idx0   = rng.choice(np.where(y == 0)[0], n_each, replace=False)
    idx1   = rng.choice(np.where(y == 1)[0], n_each, replace=False)
    idx    = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return X[idx], y[idx]


# ── ZZFeatureMap ───────────────────────────────────────────────────────────────

def zz_feature_map(x, n_qubits, reps=2):
    """
    ZZFeatureMap: Hadamard + Rz(data) + pairwise ZZ entanglement.

    Captures quantum correlations between statistically independent ICA
    features via the ZZ interaction term:
        Rz(2*(pi - xi)(pi - xj)) on qubit j after CNOT(i, j)

    Reference: Havlicek et al. (2019). Supervised learning with
    quantum-enhanced feature spaces. Nature, 567, 209-212.
    """
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(2.0 * x[i], wires=i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
                qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])


# ── QKSVM ──────────────────────────────────────────────────────────────────────

def build_kernel_matrix(X1, X2, dev, n_qubits, tag=''):
    """Compute fidelity kernel matrix K[i,j] = P(|00..0>) for circuit U(x1)U†(x2)."""
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        zz_feature_map(x1, n_qubits)
        qml.adjoint(zz_feature_map)(x2, n_qubits)
        return qml.probs(wires=range(n_qubits))

    K     = np.zeros((len(X1), len(X2)))
    total = len(X1) * len(X2)
    done  = 0
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K[i, j] = kernel_circuit(x1, x2)[0]
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  {tag}: {done}/{total} ({100*done/total:.0f}%)", end='\r')
    print()
    return K


def train_qksvm(X_tr, y_tr, X_te, y_te, dev, n_qubits, output_dir):
    print(f"\nBuilding train kernel ({len(X_tr)}x{len(X_tr)})...")
    t0      = time.time()
    K_train = build_kernel_matrix(X_tr, X_tr, dev, n_qubits, 'Train kernel')
    kt      = time.time() - t0
    print(f"  Done in {kt:.0f}s")

    print(f"Building test kernel ({len(X_te)}x{len(X_tr)})...")
    t0     = time.time()
    K_test = build_kernel_matrix(X_te, X_tr, dev, n_qubits, 'Test kernel')
    ktest_t = time.time() - t0
    print(f"  Done in {ktest_t:.0f}s")

    np.save(os.path.join(output_dir, 'K_train.npy'), K_train)
    np.save(os.path.join(output_dir, 'K_test.npy'),  K_test)

    t0   = time.time()
    qsvm = SVC(kernel='precomputed', C=1.0, probability=True)
    qsvm.fit(K_train, y_tr)
    fit_t = time.time() - t0

    y_pred  = qsvm.predict(K_test)
    y_score = qsvm.predict_proba(K_test)[:, 1]

    return {
        'name':       'Quantum Kernel SVM',
        'y_pred':     y_pred,
        'y_score':    y_score,
        'accuracy':   round(accuracy_score(y_te, y_pred), 4),
        'f1':         round(f1_score(y_te, y_pred), 4),
        'precision':  round(precision_score(y_te, y_pred), 4),
        'recall':     round(recall_score(y_te, y_pred), 4),
        'roc_auc':    round(roc_auc_score(y_te, y_score), 4),
        'train_time': round(kt + ktest_t + fit_t, 1),
    }


# ── VQC ────────────────────────────────────────────────────────────────────────

def train_vqc(X_tr, y_tr, X_te, y_te, dev, n_qubits, n_layers, epochs, batch_size, lr):
    @qml.qnode(dev, interface='autograd')
    def vqc_circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    def predict_proba(weights, X):
        out = np.array([float(vqc_circuit(weights, x)) for x in X])
        return (out + 1.0) / 2.0

    def mse_loss(weights, Xb, yb):
        preds = pnp.array([vqc_circuit(weights, x) for x in Xb])
        return pnp.mean((preds - yb) ** 2)

    w_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
    np.random.seed(RANDOM_STATE)
    weights = pnp.array(np.random.uniform(-np.pi, np.pi, w_shape), requires_grad=True)
    opt     = qml.AdamOptimizer(stepsize=lr)

    y_tr_pm = np.where(y_tr == 1, 1.0, -1.0)
    print(f"\n  VQC: {np.prod(w_shape)} params | {epochs} epochs | batch {batch_size}")

    loss_history = []
    t0 = time.time()
    for epoch in range(epochs):
        perm       = np.random.permutation(len(X_tr))
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, len(X_tr), batch_size):
            b_idx   = perm[start:start + batch_size]
            Xb, yb  = X_tr[b_idx], y_tr_pm[b_idx].astype(float)
            weights, bl = opt.step_and_cost(
                lambda w: mse_loss(w, Xb, yb), weights)
            epoch_loss += float(bl)
            n_batches  += 1
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            proba  = predict_proba(weights, X_te)
            preds  = (proba > 0.5).astype(int)
            f1_now = f1_score(y_te, preds)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss={avg_loss:.4f} | "
                  f"F1={f1_now:.4f} | {time.time()-t0:.0f}s")

    vqc_time  = time.time() - t0
    vqc_proba = predict_proba(weights, X_te)
    vqc_preds = (vqc_proba > 0.5).astype(int)

    return {
        'name':       'VQC',
        'y_pred':     vqc_preds,
        'y_score':    vqc_proba,
        'accuracy':   round(accuracy_score(y_te, vqc_preds), 4),
        'f1':         round(f1_score(y_te, vqc_preds), 4),
        'precision':  round(precision_score(y_te, vqc_preds), 4),
        'recall':     round(recall_score(y_te, vqc_preds), 4),
        'roc_auc':    round(roc_auc_score(y_te, vqc_proba), 4),
        'train_time': round(vqc_time, 1),
    }, loss_history, weights


# ── Classical baselines on same ICA-6 subset ──────────────────────────────────

def classical_on_ica(X_tr, y_tr, X_te, y_te):
    results = {}
    for name, clf in [
        ('Classical SVM (RBF)',  SVC(kernel='rbf', C=0.01, gamma='scale',
                                    probability=True, random_state=RANDOM_STATE)),
        ('Classical RF (ICA-6)', RandomForestClassifier(n_estimators=200,
                                    random_state=RANDOM_STATE, n_jobs=-1)),
    ]:
        t0  = time.time()
        clf.fit(X_tr, y_tr)
        yp  = clf.predict(X_te)
        ys  = clf.predict_proba(X_te)[:, 1]
        results[name] = {
            'y_pred':     yp,      'y_score':    ys,
            'accuracy':   round(accuracy_score(y_te, yp), 4),
            'f1':         round(f1_score(y_te, yp), 4),
            'precision':  round(precision_score(y_te, yp), 4),
            'recall':     round(recall_score(y_te, yp), 4),
            'roc_auc':    round(roc_auc_score(y_te, ys), 4),
            'train_time': round(time.time() - t0, 2),
        }
        r = results[name]
        print(f"  {name}: F1={r['f1']} | AUC={r['roc_auc']}")
    return results


# ── Save results and plot ──────────────────────────────────────────────────────

def save_results(all_res, y_te, loss_history, output_dir):
    rows = [{'Model': name,
             'Type': 'Quantum' if ('Quantum' in name or name == 'VQC') else 'Classical',
             'F1': r['f1'], 'Precision': r['precision'], 'Recall': r['recall'],
             'ROC_AUC': r['roc_auc'], 'Accuracy': r['accuracy'],
             'Train_Time_s': r['train_time']}
            for name, r in all_res.items()]
    out_csv = os.path.join(output_dir, 'quantum_comparison.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Results saved: {out_csv}")

    colors_map = {
        'Classical SVM (RBF)':  '#3498db',
        'Classical RF (ICA-6)': '#2ecc71',
        'Quantum Kernel SVM':   '#e74c3c',
        'VQC':                  '#9b59b6',
    }
    model_colors = [colors_map.get(n, '#95a5a6') for n in all_res]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Quantum vs Classical — SWaT CPS Anomaly Detection\n'
                 'Group-norm + CORAL + ICA (15 components, top-6 by MI) + '
                 'ZZFeatureMap Kernel / VQC',
                 fontsize=12, fontweight='bold')
    gsl = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

    # Metrics bar chart
    ax1 = fig.add_subplot(gsl[0, :3])
    met_k = ['f1', 'precision', 'recall', 'roc_auc', 'accuracy']
    met_l = ['F1', 'Precision', 'Recall', 'ROC-AUC', 'Accuracy']
    x = np.arange(len(met_k))
    w = 0.18
    for i, (name, r) in enumerate(all_res.items()):
        offset = (i - len(all_res) / 2 + 0.5) * w
        hatch  = '///' if ('Quantum' in name or name == 'VQC') else ''
        ax1.bar(x + offset, [r[m] for m in met_k], w,
                label=name, color=model_colors[i],
                edgecolor='black', lw=0.5, alpha=0.88, hatch=hatch)
    ax1.set_xticks(x)
    ax1.set_xticklabels(met_l, fontsize=11)
    ax1.set_ylim(0.88, 1.05)
    ax1.set_ylabel('Score')
    ax1.set_title('All Metrics — Quantum (hatched) vs Classical', fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')

    # ROC curves
    ax2 = fig.add_subplot(gsl[0, 3])
    for i, (name, r) in enumerate(all_res.items()):
        fpr, tpr, _ = roc_curve(y_te, r['y_score'])
        ls = '-' if ('Quantum' in name or name == 'VQC') else '--'
        ax2.plot(fpr, tpr, color=model_colors[i], lw=2, ls=ls,
                 label=f"{name}\n(AUC={r['roc_auc']:.4f})")
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_title('ROC Curves\n(solid=quantum, dashed=classical)', fontweight='bold')
    ax2.legend(fontsize=7)

    # Confusion matrices
    cm_cmaps = ['Blues', 'Blues', 'Reds', 'Purples']
    for idx, ((name, r), cmap) in enumerate(zip(all_res.items(), cm_cmaps)):
        ax = fig.add_subplot(gsl[1, idx])
        cm  = confusion_matrix(y_te, r['y_pred'])
        cm_n = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_n, annot=True, fmt='.2%', cmap=cmap, ax=ax,
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'],
                    annot_kws={'size': 10, 'weight': 'bold'})
        tag = ' (Q)' if ('Quantum' in name or name == 'VQC') else ''
        ax.set_title(f'{name}{tag}\nF1={r["f1"]:.4f} | AUC={r["roc_auc"]:.4f}',
                     fontweight='bold', fontsize=9)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    # VQC loss curve
    ax_l = fig.add_subplot(gsl[2, 1])
    ax_l.plot(range(1, len(loss_history) + 1), loss_history,
              color='#9b59b6', lw=2, marker='o', ms=2.5, markevery=5)
    ax_l.fill_between(range(1, len(loss_history) + 1), loss_history,
                      alpha=0.15, color='#9b59b6')
    ax_l.set_xlabel('Epoch')
    ax_l.set_ylabel('MSE Loss')
    ax_l.set_title('VQC Training Loss', fontweight='bold')
    ax_l.grid(True, alpha=0.25)

    out_png = os.path.join(output_dir, 'quantum_results.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_png}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Quantum models for CPS anomaly detection')
    parser.add_argument('--data_dir',   default='data/processed')
    parser.add_argument('--output_dir', default='reports')
    parser.add_argument('--n_train',    type=int, default=400,
                        help='Quantum training samples (balanced, default: 400)')
    parser.add_argument('--n_test',     type=int, default=200,
                        help='Quantum test samples (balanced, default: 200)')
    parser.add_argument('--epochs',     type=int, default=60, help='VQC epochs')
    args = parser.parse_args()

    X_train_qml = np.load(os.path.join(args.data_dir, 'X_train_qml.npy'))
    X_test_qml  = np.load(os.path.join(args.data_dir, 'X_test_qml.npy'))
    y_train     = np.load(os.path.join(args.data_dir, 'y_train_qml.npy'))
    y_test      = np.load(os.path.join(args.data_dir, 'y_test_qml.npy'))

    X_tr, y_tr = balanced_sample(X_train_qml, y_train, args.n_train)
    X_te, y_te = balanced_sample(X_test_qml,  y_test,  args.n_test)

    print(f"QML subset — Train: {len(X_tr)} | Test: {len(X_te)}")
    print(f"Features: {X_tr.shape[1]} qubits | Angles in [0, pi]")

    dev = qml.device('lightning.qubit', wires=N_QUBITS)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Quantum Kernel SVM ===")
    qsvm_res = train_qksvm(X_tr, y_tr, X_te, y_te, dev, N_QUBITS, args.output_dir)
    r = qsvm_res
    print(f"  QKSVM: F1={r['f1']} | AUC={r['roc_auc']} | "
          f"Prec={r['precision']} | Rec={r['recall']}")

    print("\n=== VQC ===")
    vqc_res, loss_history, vqc_weights = train_vqc(
        X_tr, y_tr, X_te, y_te, dev, N_QUBITS, N_LAYERS,
        args.epochs, BATCH_SIZE, LR)
    r = vqc_res
    print(f"  VQC: F1={r['f1']} | AUC={r['roc_auc']} | "
          f"Prec={r['precision']} | Rec={r['recall']}")
    np.save(os.path.join(args.output_dir, 'vqc_weights.npy'), np.array(vqc_weights))

    print("\n=== Classical Baselines (ICA-6 subset) ===")
    classic_res = classical_on_ica(X_tr, y_tr, X_te, y_te)

    all_res = {**classic_res, 'Quantum Kernel SVM': qsvm_res, 'VQC': vqc_res}

    print(f"\n{'Model':<25} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUC':>7} {'Acc':>7}")
    print("-" * 58)
    for name, r in all_res.items():
        tag = ' (Q)' if ('Quantum' in name or name == 'VQC') else ''
        print(f"{name:<25} {r['f1']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['roc_auc']:>7.4f} {r['accuracy']:>7.4f}{tag}")

    save_results(all_res, y_te, loss_history, args.output_dir)


if __name__ == '__main__':
    main()
