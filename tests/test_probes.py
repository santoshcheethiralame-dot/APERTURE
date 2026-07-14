import numpy as np

from mirror.probes import prg, train_probe


def _separable(n_classes, per_class, groups, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, 8)) * 5
    acts, labels, grp = [], [], []
    for g in range(groups):
        for c in range(n_classes):
            for _ in range(per_class):
                acts.append(centers[c] + rng.normal(size=8) * 0.3)
                labels.append(c)
                grp.append(g)
    return np.array(acts), np.array(labels), np.array(grp)


def test_probe_recovers_separable_classes():
    acts, labels, groups = _separable(4, 5, groups=3, seed=0)
    result = train_probe(acts, labels, groups)
    assert result.accuracy > 0.8
    assert result.n_classes == 4


def test_probe_control_is_chance():
    acts, labels, groups = _separable(4, 5, groups=3, seed=1)
    result = train_probe(acts, labels, groups)
    assert result.control_accuracy < 2 / 4


def test_prg_is_difference():
    assert abs(prg(0.7, 0.1) - 0.6) < 1e-9
