import numpy as np


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
