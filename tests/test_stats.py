import numpy as np

from aperture.stats import bootstrap_ci


def test_ci_contains_known_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, size=200)
    lo, hi = bootstrap_ci(values, rng=np.random.default_rng(1))
    assert lo < 5.0 < hi


def test_ci_collapses_on_identical_values():
    lo, hi = bootstrap_ci([3.0] * 20, rng=np.random.default_rng(2))
    assert abs(lo - 3.0) < 1e-9
    assert abs(hi - 3.0) < 1e-9


def test_ci_on_proportion_excludes_chance():
    hits = [1] * 11 + [0] * 5
    lo, hi = bootstrap_ci(hits, rng=np.random.default_rng(3))
    assert lo > 0.0625
    assert lo < 0.688 < hi


def test_ci_is_deterministic_under_seed():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = bootstrap_ci(values, rng=np.random.default_rng(4))
    b = bootstrap_ci(values, rng=np.random.default_rng(4))
    assert a == b
