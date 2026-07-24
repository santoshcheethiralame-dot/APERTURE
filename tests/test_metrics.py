from aperture.metrics import kl_meter

PROMPT = "The weather today is"


def test_kl_zero_at_alpha_zero(model, vec):
    assert abs(kl_meter(model, PROMPT, vec, 0.0)) < 1e-4


def test_kl_positive_under_injection(model, vec):
    assert kl_meter(model, PROMPT, vec, 8.0) > 0.01
