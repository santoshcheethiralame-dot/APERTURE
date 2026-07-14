from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbeResult:
    accuracy: float
    control_accuracy: float
    n_classes: int


def _fit_eval(acts, labels, train_mask, test_mask):
    model = LogisticRegression(max_iter=1000)
    model.fit(acts[train_mask], labels[train_mask])
    return model.score(acts[test_mask], labels[test_mask])


def train_probe(acts, labels, groups, seed=0):
    acts = np.asarray(acts)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    test_mask = groups == groups.max()
    train_mask = ~test_mask
    accuracy = _fit_eval(acts, labels, train_mask, test_mask)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(labels)
    control = _fit_eval(acts, shuffled, train_mask, test_mask)
    return ProbeResult(accuracy=float(accuracy), control_accuracy=float(control),
                       n_classes=int(len(np.unique(labels))))


def prg(probe_accuracy, report_accuracy):
    return float(probe_accuracy - report_accuracy)
