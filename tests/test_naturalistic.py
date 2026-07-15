import torch

from mirror.naturalistic import nearest_concept


def test_nearest_concept_picks_aligned_direction():
    directions = {
        "elephant": torch.tensor([1.0, 0.0, 0.0]),
        "volcano": torch.tensor([0.0, 1.0, 0.0]),
        "joy": torch.tensor([0.0, 0.0, 1.0]),
    }
    assert nearest_concept(torch.tensor([0.0, 5.0, 0.0]), directions) == "volcano"


def test_nearest_concept_handles_mixed_activation():
    directions = {
        "elephant": torch.tensor([1.0, 0.0]),
        "volcano": torch.tensor([0.0, 1.0]),
    }
    assert nearest_concept(torch.tensor([3.0, 1.0]), directions) == "elephant"


def test_last_activation_shape(hf_model, hf_tok):
    from mirror.naturalistic import last_activation_hf
    act = last_activation_hf(hf_model, hf_tok, "the ground trembled", 1)
    assert act.shape == (hf_model.config.hidden_size,)


def test_last_activation_leaves_no_hook(hf_model, hf_tok):
    from mirror.hf_model import hf_layer
    from mirror.naturalistic import last_activation_hf
    last_activation_hf(hf_model, hf_tok, "the ground trembled", 1)
    assert len(hf_layer(hf_model, 1)._forward_hooks) == 0


def test_contexts_cover_dev_bank():
    from mirror.concepts import load_bank
    from mirror.naturalistic import load_contexts
    contexts = load_contexts()
    bank = load_bank("data/concepts/dev_bank.yaml")
    for concept in bank.concepts:
        assert concept.name in contexts
        assert len(contexts[concept.name]) > 40


def test_contexts_never_name_their_concept():
    from mirror.naturalistic import load_contexts
    for name, passage in load_contexts().items():
        assert name not in passage.lower()


def test_load_contexts_rejects_leaked_concept(tmp_path):
    import pytest

    from mirror.naturalistic import load_contexts
    bad = tmp_path / "contexts.yaml"
    bad.write_text("volcano: The volcano erupted loudly over the valley below.\n")
    with pytest.raises(ValueError):
        load_contexts(bad)
