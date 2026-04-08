"""
train_qksvr.py — SVR training with a precomputed quantum kernel.

Wraps scikit-learn's SVR(kernel='precomputed') with the optimal
hyperparameters identified during the grid search.
"""

from sklearn.svm import SVR


# ── Optimal hyperparameters ───────────────────────────────────
BEST_C = 2000.0
BEST_EPSILON = 0.010


def train_qksvr(K_train, y_train, C=BEST_C, epsilon=BEST_EPSILON):
    """
    Fit an SVR model on a precomputed quantum kernel matrix.

    Parameters
    ----------
    K_train : np.ndarray, shape (n, n)
        Precomputed training kernel matrix.
    y_train : np.ndarray, shape (n,)
        Training target values.
    C       : float
        Regularization parameter.
    epsilon : float
        ε-insensitive tube width.

    Returns
    -------
    model : sklearn.svm.SVR
        Fitted SVR model.
    """
    model = SVR(C=C, epsilon=epsilon, kernel="precomputed")
    model.fit(K_train, y_train)
    return model
