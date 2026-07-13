import numpy as np

from mirror.prior_null import neg_log_likelihood, simulate_reports


def test_nll_matches_hand_computation():
    X = np.array([[[0.0], [1.0]]])
    y = np.array([1])
    theta = np.array([2.0])
    logits = np.array([0.0, 2.0])
    expected = -(logits[1] - np.log(np.exp(logits).sum()))
    assert np.isclose(neg_log_likelihood(theta, X, y), expected)


def test_nll_uniform_at_zero_theta():
    X = np.zeros((1, 4, 1))
    y = np.array([0])
    theta = np.array([0.0])
    assert np.isclose(neg_log_likelihood(theta, X, y), np.log(4))


def test_simulate_prefers_high_logit_concept():
    n_trials = 2000
    X = np.zeros((n_trials, 3, 1))
    X[:, 2, 0] = 1.0
    theta = np.array([5.0])
    rng = np.random.default_rng(0)
    y = simulate_reports(X, theta, rng)
    assert (y == 2).mean() > 0.9


def test_simulate_shape_and_range():
    X = np.zeros((10, 4, 2))
    rng = np.random.default_rng(1)
    y = simulate_reports(X, np.zeros(2), rng)
    assert y.shape == (10,)
    assert set(np.unique(y)).issubset({0, 1, 2, 3})
