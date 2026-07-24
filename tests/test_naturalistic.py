import torch

from aperture.naturalistic import nearest_concept


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


def test_centering_defeats_common_component():
    from aperture.naturalistic import center_activations
    directions = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
    bias = torch.tensor([10.0, 0.0])
    raw = {"a": bias + torch.tensor([1.0, 0.0]), "b": bias + torch.tensor([0.0, 1.0])}
    assert nearest_concept(raw["b"], directions) == "a"
    centered = center_activations(raw)
    assert nearest_concept(centered["b"], directions) == "b"
    assert nearest_concept(centered["a"], directions) == "a"


def test_last_activation_shape(hf_model, hf_tok):
    from aperture.naturalistic import last_activation_hf
    act = last_activation_hf(hf_model, hf_tok, "the ground trembled", 1)
    assert act.shape == (hf_model.config.hidden_size,)


def test_last_activation_leaves_no_hook(hf_model, hf_tok):
    from aperture.hf_model import hf_layer
    from aperture.naturalistic import last_activation_hf
    last_activation_hf(hf_model, hf_tok, "the ground trembled", 1)
    assert len(hf_layer(hf_model, 1)._forward_hooks) == 0


def test_contexts_cover_dev_bank():
    from aperture.concepts import load_bank
    from aperture.naturalistic import load_contexts
    contexts = load_contexts()
    bank = load_bank("data/concepts/dev_bank.yaml")
    for concept in bank.concepts:
        assert concept.name in contexts
        assert len(contexts[concept.name]) > 40


def test_contexts_never_name_their_concept():
    from aperture.naturalistic import load_contexts
    for name, passage in load_contexts().items():
        assert name not in passage.lower()


def test_load_contexts_rejects_leaked_concept(tmp_path):
    import pytest

    from aperture.naturalistic import load_contexts
    bad = tmp_path / "contexts.yaml"
    bad.write_text("volcano: The volcano erupted loudly over the valley below.\n")
    with pytest.raises(ValueError):
        load_contexts(bad)


def test_collect_naturalistic_writes_records(hf_model, hf_tok, tmp_path):
    import json

    from aperture.concepts import load_bank
    from aperture.naturalistic import collect_naturalistic_hf
    bank = load_bank("data/concepts/dev_bank.yaml")
    contexts = {
        "elephant": "The grey giant swayed through the tall grass.",
        "volcano": "The ground trembled and ash darkened the sky.",
        "joy": "She could not stop smiling and everything seemed to shine.",
    }
    out = tmp_path / "nat.jsonl"
    result = collect_naturalistic_hf(hf_model, hf_tok, bank, contexts,
                                     "\n\nQ: What is two plus two?\nA: Four.\n\n",
                                     "Q: In one word, what concept was most on your mind?\nA:",
                                     0, n_pairs=10, max_new_tokens=4, out=str(out))
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(result["records"]) == len(lines) == 3
    assert {"concept", "predicted", "report", "identified"} <= set(lines[0])
    assert lines[0]["predicted"] in contexts
