from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class Fit:
    theta: np.ndarray
    gamma: float
    loglik: float


def neg_log_likelihood(theta, X, y):
    logits = X @ theta
    m = logits.max(axis=1, keepdims=True)
    log_norm = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    chosen = logits[np.arange(len(y)), y]
    return float((log_norm - chosen).sum())


def _softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def simulate_reports(X, theta_true, rng):
    probs = _softmax(X @ theta_true)
    n_concepts = X.shape[1]
    return np.array([rng.choice(n_concepts, p=probs[t]) for t in range(len(probs))])


def fit(X, y):
    n_features = X.shape[2]
    result = minimize(neg_log_likelihood, np.zeros(n_features), args=(X, y),
                      method="L-BFGS-B")
    theta = result.x
    return Fit(theta=theta, gamma=float(theta[-1]), loglik=-float(result.fun))


def gamma_ci(X, y, n_boot=200, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    n_trials = len(y)
    gammas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_trials, size=n_trials)
        gammas.append(fit(X[idx], y[idx]).gamma)
    lo, hi = np.percentile(gammas, [2.5, 97.5])
    return float(lo), float(hi)


def gamma_difference_ci(X_a, y_a, X_b, y_b, n_boot=200, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(y_a), size=len(y_a))
        ib = rng.integers(0, len(y_b), size=len(y_b))
        diffs.append(fit(X_a[ia], y_a[ia]).gamma - fit(X_b[ib], y_b[ib]).gamma)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)
