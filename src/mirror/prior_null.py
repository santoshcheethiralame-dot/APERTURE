import numpy as np


def neg_log_likelihood(theta, X, y):
    logits = X @ theta
    m = logits.max(axis=1, keepdims=True)
    log_norm = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    chosen = logits[np.arange(len(y)), y]
    return float((log_norm - chosen).sum())
