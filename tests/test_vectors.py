import torch


def test_direction_is_unit_norm(vec, model):
    assert vec.direction.shape == (model.cfg.d_model,)
    assert torch.isclose(vec.direction.norm(), torch.tensor(1.0), atol=1e-5)


def test_sigma_positive(vec):
    assert vec.sigma > 0


def test_metadata(vec):
    assert vec.concept == "elephant"
    assert vec.layer == 3
