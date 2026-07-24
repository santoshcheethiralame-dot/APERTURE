import torch


def test_direction_is_unit_norm(vec, model):
    assert vec.direction.shape == (model.cfg.d_model,)
    assert torch.isclose(vec.direction.norm(), torch.tensor(1.0), atol=1e-5)


def test_sigma_positive(vec):
    assert vec.sigma > 0


def test_metadata(vec):
    assert vec.concept == "elephant"
    assert vec.layer == 3


def test_flags_present_and_boolean(vec):
    assert set(vec.flags) == {"steering", "probe", "stability"}
    assert all(isinstance(v, bool) for v in vec.flags.values())


def test_probe_split_shares_no_prompts(bank):
    from aperture.vectors import split_pairs

    pairs = bank.pairs(bank.get("elephant"), n_pairs=20)
    train, test = split_pairs(pairs, len(bank.templates))
    assert train and test
    train_prompts = {p for pair in train for p in pair}
    test_prompts = {p for pair in test for p in pair}
    assert not train_prompts & test_prompts
