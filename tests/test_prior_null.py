import numpy as np

from mirror.prior_null import neg_log_likelihood


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
