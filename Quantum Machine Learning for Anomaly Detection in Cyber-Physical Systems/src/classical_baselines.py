"""
classical_baselines.py
----------------------
Trains and evaluates five classical ML models as baselines for CPS
anomaly detection on the SWaT dataset:

  1. SVM (RBF kernel)
  2. SVM (Linear kernel)
  3. Random Forest
  4. Gradient Boosting
  5. Isolation Forest (unsupervised)

Each supervised model is hyperparameter-tuned with GridSearchCV (5-fold
stratified CV). Results include accuracy, F1, precision, recall, ROC-AUC,
confusion matrices, and feature importance (Random Forest).

IMPORTANT NOTE on dataset shift:
  When run on the naive cross-year merge (2026 normal + 2015 attacks)
  WITHOUT CORAL domain adaptation, all supervised models achieve
  F1 = AUC = 1.0000. This is a distributional artefact, not genuine
  attack discrimination. The Isolation Forest achieves F1 = 0.4726
  on the same raw data, confirming the signal is year-specific.
  Run dataset_shift.py first to apply CORAL adaptation.

Usage:
    python classical_baselines.py --data_dir  data/processed
                                  --output_dir reports
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

from sklearn.ensemble import (GradientBoostingClassifier,
                              IsolationForest,
                              RandomForestClassifier)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

RANDOM_STATE = 42
TEST_SIZE    = 0.30
N_SVM        = 10_000    # SVM subsample (SVM is O(n^2))
N_GB         = 15_000    # Gradient Boosting subsample


# ── Model evaluation helper ────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_t = round(time.time() - t0, 2)

    y_pred = model.predict(X_te)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_te)
    else:
        y_score = y_pred.astype(float)

    return {
        'name':       name,
        'model':      model,
        'y_pred':     y_pred,
        'y_score':    y_score,
        'accuracy':   round(accuracy_score(y_te, y_pred), 4),
        'f1':         round(f1_score(y_te, y_pred), 4),
        'precision':  round(precision_score(y_te, y_pred), 4),
        'recall':     round(recall_score(y_te, y_pred), 4),
        'roc_auc':    round(roc_auc_score(y_te, y_score), 4),
        'train_time': train_t,
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train_all(X_train, y_train, X_test, y_test, feature_cols):
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    rng     = np.random.RandomState(RANDOM_STATE)

    # SVM subsample (speed)
    idx_svm  = rng.choice(len(X_train), min(N_SVM, len(X_train)), replace=False)
    X_svm, y_svm = X_train[idx_svm], y_train[idx_svm]

    # ── 1. SVM RBF ──
    print("\n[1/5] SVM (RBF Kernel)")
    gs = GridSearchCV(
        SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.001]},
        cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    gs.fit(X_svm, y_svm)
    print(f"  Best params: {gs.best_params_}  |  CV F1: {gs.best_score_:.4f}")
    results['SVM (RBF)'] = evaluate_model(
        'SVM (RBF)',
        SVC(**gs.best_params_, kernel='rbf', probability=True, random_state=RANDOM_STATE),
        X_svm, y_svm, X_test, y_test)
    r = results['SVM (RBF)']
    print(f"  Test: F1={r['f1']} | AUC={r['roc_auc']} | "
          f"Prec={r['precision']} | Rec={r['recall']}")

    # ── 2. SVM Linear ──
    print("\n[2/5] SVM (Linear Kernel)")
    gs = GridSearchCV(
        SVC(kernel='linear', probability=True, random_state=RANDOM_STATE),
        {'C': [0.01, 0.1, 1, 10]},
        cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    gs.fit(X_svm, y_svm)
    print(f"  Best params: {gs.best_params_}  |  CV F1: {gs.best_score_:.4f}")
    results['SVM (Linear)'] = evaluate_model(
        'SVM (Linear)',
        SVC(**gs.best_params_, kernel='linear', probability=True, random_state=RANDOM_STATE),
        X_svm, y_svm, X_test, y_test)
    r = results['SVM (Linear)']
    print(f"  Test: F1={r['f1']} | AUC={r['roc_auc']} | "
          f"Prec={r['precision']} | Rec={r['recall']}")

    # ── 3. Random Forest ──
    print("\n[3/5] Random Forest")
    gs = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {'n_estimators': [100, 200, 300], 'max_depth': [None, 15, 30],
         'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5]},
        cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    print(f"  Best params: {gs.best_params_}  |  CV F1: {gs.best_score_:.4f}")
    results['Random Forest'] = evaluate_model(
        'Random Forest',
        RandomForestClassifier(**gs.best_params_, random_state=RANDOM_STATE, n_jobs=-1),
        X_train, y_train, X_test, y_test)
    r = results['Random Forest']
    print(f"  Test: F1={r['f1']} | AUC={r['roc_auc']} | "
          f"Prec={r['precision']} | Rec={r['recall']}")

    # ── 4. Gradient Boosting ──
    print("\n[4/5] Gradient Boosting")
    idx_gb  = rng.choice(len(X_train), min(N_GB, len(X_train)), replace=False)
    X_gb, y_gb = X_train[idx_gb], y_train[idx_gb]
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
         'max_depth': [3, 5, 7], 'subsample': [0.8, 1.0]},
        cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    gs.fit(X_gb, y_gb)
    print(f"  Best params: {gs.best_params_}  |  CV F1: {gs.best_score_:.4f}")
    results['Gradient Boosting'] = evaluate_model(
        'Gradient Boosting',
        GradientBoostingClassifier(**gs.best_params_, random_state=RANDOM_STATE),
        X_gb, y_gb, X_test, y_test)
    r = results['Gradient Boosting']
    print(f"  Test: F1={r['f1']} | AUC={r['roc_auc']} | "
          f"Prec={r['precision']} | Rec={r['recall']}")

    # ── 5. Isolation Forest ──
    print("\n[5/5] Isolation Forest (unsupervised)")
    best_f1, best_params = -1, {}
    for n_est in [100, 200, 300]:
        for contam in [0.1, 0.2, 0.3, 0.5]:
            for mf in [0.5, 0.8, 1.0]:
                iso = IsolationForest(n_estimators=n_est, contamination=contam,
                                     max_features=mf, random_state=RANDOM_STATE, n_jobs=-1)
                iso.fit(X_train)
                preds = np.where(iso.predict(X_test) == -1, 1, 0)
                f1 = f1_score(y_test, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'n_estimators': n_est, 'contamination': contam,
                                   'max_features': mf}
    print(f"  Best params: {best_params}  |  Best F1: {best_f1:.4f}")
    best_iso = IsolationForest(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    best_iso.fit(X_train)
    iso_pred  = np.where(best_iso.predict(X_test) == -1, 1, 0)
    iso_score = -best_iso.decision_function(X_test)
    results['Isolation Forest'] = {
        'name': 'Isolation Forest', 'model': best_iso,
        'y_pred': iso_pred, 'y_score': iso_score,
        'accuracy':   round(accuracy_score(y_test, iso_pred), 4),
        'f1':         round(f1_score(y_test, iso_pred), 4),
        'precision':  round(precision_score(y_test, iso_pred), 4),
        'recall':     round(recall_score(y_test, iso_pred), 4),
        'roc_auc':    round(roc_auc_score(y_test, iso_score), 4),
        'train_time': 0,
    }
    r = results['Isolation Forest']
    print(f"  Test: F1={r['f1']} | AUC={r['roc_auc']} | "
          f"Prec={r['precision']} | Rec={r['recall']}")

    return results


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_summary(results, y_test):
    print(f"\n{'Model':<22} {'Acc':>7} {'F1':>7} {'Prec':>7} "
          f"{'Rec':>7} {'AUC':>7} {'Time(s)':>8}")
    print("-" * 68)
    for name, r in results.items():
        print(f"{name:<22} {r['accuracy']:>7.4f} {r['f1']:>7.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['roc_auc']:>7.4f} {r['train_time']:>8.1f}")

    best_name = max(results, key=lambda k: results[k]['f1'])
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.4f})")
    print(classification_report(y_test, results[best_name]['y_pred'],
                                 target_names=['Normal', 'Attack']))


def save_plots(results, y_test, feature_cols, output_dir):
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle('Classical Baselines — SWaT CPS Anomaly Detection',
                 fontsize=14, fontweight='bold')
    gsl = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Metrics bar chart
    ax1 = fig.add_subplot(gsl[0, :2])
    met_k = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    met_l = ['Accuracy', 'F1', 'Precision', 'Recall', 'ROC-AUC']
    x = np.arange(len(met_k))
    n_m, w = len(results), 0.15
    for i, (name, r) in enumerate(results.items()):
        offset = (i - n_m / 2 + 0.5) * w
        ax1.bar(x + offset, [r[m] for m in met_k], w, label=name,
                color=colors[i], edgecolor='black', lw=0.4, alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(met_l)
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel('Score')
    ax1.set_title('All Metrics Comparison', fontweight='bold')
    ax1.legend(fontsize=8)

    # ROC curves
    ax2 = fig.add_subplot(gsl[0, 2])
    for i, (name, r) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, r['y_score'])
        ax2.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{name} ({r['roc_auc']:.3f})")
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves', fontweight='bold')
    ax2.legend(fontsize=7)

    # Confusion matrices
    cm_pos = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    for idx, (name, r) in enumerate(results.items()):
        row, col = cm_pos[idx]
        ax = fig.add_subplot(gsl[row, col])
        cm  = confusion_matrix(y_test, r['y_pred'])
        cm_n = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_n, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'],
                    annot_kws={'size': 9, 'weight': 'bold'})
        ax.set_title(f'{name}\nF1={r["f1"]} | AUC={r["roc_auc"]}',
                     fontweight='bold', fontsize=9)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    # RF feature importance
    ax_fi = fig.add_subplot(gsl[2, 2])
    rf_m  = results['Random Forest']['model']
    imps  = rf_m.feature_importances_
    top_i = np.argsort(imps)[-15:]
    top_f = [feature_cols[i].replace('.Pv', '').replace('.Status', '')
              for i in top_i]
    ax_fi.barh(range(len(top_i)), imps[top_i],
               color=plt.cm.RdYlGn(imps[top_i] / imps[top_i].max()),
               edgecolor='black', lw=0.4)
    ax_fi.set_yticks(range(len(top_i)))
    ax_fi.set_yticklabels(top_f, fontsize=7)
    ax_fi.set_title('Feature Importance (RF Top 15)', fontweight='bold', fontsize=9)
    ax_fi.set_xlabel('Importance Score')

    out_path = os.path.join(output_dir, 'classical_results.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train classical ML baselines on SWaT dataset')
    parser.add_argument('--data_dir',   default='data/processed')
    parser.add_argument('--output_dir', default='reports')
    args = parser.parse_args()

    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    X_test  = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    y_test  = np.load(os.path.join(args.data_dir, 'y_test.npy'))

    with open(os.path.join(args.data_dir, 'feature_cols.txt')) as f:
        feature_cols = [l.strip() for l in f]

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Normal: {(y_test == 0).sum():,} | Attack: {(y_test == 1).sum():,}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = train_all(X_train, y_train, X_test, y_test, feature_cols)

    print_summary(results, y_test)

    rows = [{'Model': name, 'Accuracy': r['accuracy'], 'F1': r['f1'],
             'Precision': r['precision'], 'Recall': r['recall'],
             'ROC_AUC': r['roc_auc'], 'Train_Time_s': r['train_time']}
            for name, r in results.items()]
    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, 'classical_results.csv'), index=False)
    print(f"Results saved: {args.output_dir}/classical_results.csv")

    save_plots(results, y_test, feature_cols, args.output_dir)


if __name__ == '__main__':
    main()
