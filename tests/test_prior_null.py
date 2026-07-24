import numpy as np

from aperture.prior_null import fit, gamma_ci, neg_log_likelihood, simulate_reports


def _sim_dataset(theta_true, n_trials, n_concepts, n_features, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_trials, n_concepts, n_features))
    injected = rng.integers(0, n_concepts, size=n_trials)
    X[:, :, -1] = 0.0
    X[np.arange(n_trials), injected, -1] = 1.0
    y = simulate_reports(X, theta_true, rng)
    return X, y


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


def test_fit_recovers_zero_gamma():
    theta_true = np.array([1.0, 0.0])
    X, y = _sim_dataset(theta_true, 4000, 6, 2, seed=0)
    result = fit(X, y)
    assert abs(result.gamma) < 0.3


def test_fit_recovers_signal_gamma():
    theta_true = np.array([0.5, 2.0])
    X, y = _sim_dataset(theta_true, 4000, 6, 2, seed=1)
    result = fit(X, y)
    assert abs(result.gamma - 2.0) < 0.4


def test_fit_recovers_frequency_coefficient():
    theta_true = np.array([1.5, 0.0])
    X, y = _sim_dataset(theta_true, 4000, 6, 2, seed=2)
    result = fit(X, y)
    assert abs(result.theta[0] - 1.5) < 0.4


def test_ci_excludes_zero_under_signal():
    theta_true = np.array([0.5, 2.0])
    X, y = _sim_dataset(theta_true, 3000, 6, 2, seed=3)
    lo, hi = gamma_ci(X, y, n_boot=60, rng=np.random.default_rng(3))
    assert lo > 0.0


def test_ci_includes_zero_under_pure_prior():
    theta_true = np.array([1.0, 0.0])
    X, y = _sim_dataset(theta_true, 3000, 6, 2, seed=4)
    lo, hi = gamma_ci(X, y, n_boot=60, rng=np.random.default_rng(4))
    assert lo < 0.0 < hi


def test_gamma_difference_detects_a_real_gap():
    from aperture.prior_null import gamma_difference_ci
    Xa, ya = _sim_dataset(np.array([0.5, 2.0]), 1500, 6, 2, seed=10)
    Xb, yb = _sim_dataset(np.array([0.5, 0.0]), 1500, 6, 2, seed=11)
    lo, hi = gamma_difference_ci(Xa, ya, Xb, yb, n_boot=40,
                                 rng=np.random.default_rng(12))
    assert lo > 0.0


def test_gamma_difference_is_null_for_matched_conditions():
    from aperture.prior_null import gamma_difference_ci
    Xa, ya = _sim_dataset(np.array([0.5, 1.0]), 1500, 6, 2, seed=13)
    Xb, yb = _sim_dataset(np.array([0.5, 1.0]), 1500, 6, 2, seed=14)
    lo, hi = gamma_difference_ci(Xa, ya, Xb, yb, n_boot=40,
                                 rng=np.random.default_rng(15))
    assert lo < 0.0 < hi
