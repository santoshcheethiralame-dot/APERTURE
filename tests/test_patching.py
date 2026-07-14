import torch

from mirror.patching import baseline_logprob, concept_token


def test_concept_token_is_int(hf_tok):
    t = concept_token(hf_tok, "elephant")
    assert isinstance(t, int)


def test_baseline_logprob_is_nonpositive(hf_model, hf_tok):
    t = concept_token(hf_tok, "elephant")
    lp = baseline_logprob(hf_model, hf_tok, "hello world", t)
    assert lp <= 0.0


def test_patched_logprob_changes_output(hf_model, hf_tok):
    from mirror.patching import patched_logprob
    t = concept_token(hf_tok, "elephant")
    base = baseline_logprob(hf_model, hf_tok, "hello world", t)
    big = torch.ones(hf_model.config.hidden_size) * 50.0
    patched = patched_logprob(hf_model, hf_tok, "hello world", 1, big, t)
    assert patched != base


def _tiny_vec(hf_model, hf_tok):
    from mirror.hf_model import extract_hf_vector
    return extract_hf_vector(hf_model, hf_tok, [("a cat", "a dog")], 0)


def test_patch_effect_zero_alpha_is_noop(hf_model, hf_tok):
    from mirror.patching import patch_effect_hf
    vec = _tiny_vec(hf_model, hf_tok)
    t = concept_token(hf_tok, "elephant")
    base, patched, delta = patch_effect_hf(hf_model, hf_tok, "hello world", vec,
                                           0.0, 1, t)
    assert abs(delta) < 1e-4


def test_patch_effect_nonzero_alpha_moves(hf_model, hf_tok):
    from mirror.patching import patch_effect_hf
    vec = _tiny_vec(hf_model, hf_tok)
    t = concept_token(hf_tok, "elephant")
    base, patched, delta = patch_effect_hf(hf_model, hf_tok, "hello world", vec,
                                           200.0, 1, t)
    assert abs(delta) > 0.0


def test_collect_patch_writes_records(hf_model, hf_tok, tmp_path):
    import json

    from mirror.concepts import load_bank
    from mirror.patching import collect_patch_hf
    bank = load_bank("data/concepts/dev_bank.yaml")
    out = tmp_path / "patch.jsonl"
    result = collect_patch_hf(hf_model, hf_tok, bank,
                              ["elephant", "volcano", "joy"], 0, 1, 1.0,
                              "hello world", str(out), n_pairs=10)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(result["records"]) == len(lines) == 3
    assert {"concept", "layer", "patch_layer", "alpha", "self_delta", "control_delta"} <= set(lines[0])
