import torch

from mirror.hf_model import hf_layer, raw_direction_hf, resid_stats_hf


def test_hf_layer_returns_decoder_block(hf_model):
    layer = hf_layer(hf_model, 0)
    assert layer is hf_model.model.layers[0]


def test_resid_stats_shape(hf_model, hf_tok):
    mean_resid, median_norm = resid_stats_hf(hf_model, hf_tok, "hello world", 0)
    assert mean_resid.shape == (hf_model.config.hidden_size,)
    assert float(median_norm) >= 0.0


def test_raw_direction_shape(hf_model, hf_tok):
    pairs = [("a cat", "a dog"), ("the cat", "the dog")]
    vector, sigma = raw_direction_hf(hf_model, hf_tok, pairs, 0)
    assert vector.shape == (hf_model.config.hidden_size,)
    assert isinstance(sigma, float)


def _tiny_vec(hf_model, hf_tok):
    from mirror.hf_model import extract_hf_vector
    return extract_hf_vector(hf_model, hf_tok, [("a cat", "a dog")], 0)


def test_generate_alpha_zero_is_golden(hf_model, hf_tok):
    from mirror.hf_model import generate_hf
    vec = _tiny_vec(hf_model, hf_tok)
    clean = generate_hf(hf_model, hf_tok, "hello", seed=0, max_new_tokens=8)
    zeroed = generate_hf(hf_model, hf_tok, "hello", vec, alpha=0.0, seed=0, max_new_tokens=8)
    assert clean == zeroed


def test_generate_huge_alpha_changes(hf_model, hf_tok):
    from mirror.hf_model import generate_hf
    vec = _tiny_vec(hf_model, hf_tok)
    clean = generate_hf(hf_model, hf_tok, "hello", seed=0, max_new_tokens=8)
    injected = generate_hf(hf_model, hf_tok, "hello", vec, alpha=500.0, seed=0, max_new_tokens=8)
    assert clean != injected


def test_injection_only_last_position(hf_model, hf_tok):
    from mirror.hf_model import inject_hook
    vec = _tiny_vec(hf_model, hf_tok)
    ids = hf_tok("hello world", return_tensors="pt").input_ids
    captured = {}

    def grab(module, inputs, output):
        captured.setdefault("clean", _hidden_out(output).detach().clone())

    def _hidden_out(output):
        return output[0] if isinstance(output, tuple) else output

    h = hf_layer(hf_model, 0).register_forward_hook(grab)
    with torch.no_grad():
        hf_model(ids)
    h.remove()

    captured2 = {}

    def grab2(module, inputs, output):
        captured2["inj"] = _hidden_out(output).detach().clone()
        return output

    inj = hf_layer(hf_model, 0).register_forward_hook(inject_hook(vec, 4.0, "response"))
    grabh = hf_layer(hf_model, 0).register_forward_hook(grab2)
    with torch.no_grad():
        hf_model(ids)
    inj.remove()
    grabh.remove()

    clean, inj_resid = captured["clean"][0], captured2["inj"][0]
    assert torch.allclose(clean[:-1], inj_resid[:-1], atol=1e-5)
    expected = clean[-1] + 4.0 * vec.sigma * vec.direction
    assert torch.allclose(inj_resid[-1], expected, atol=1e-4)


def test_extract_hf_flags(hf_model, hf_tok):
    from mirror.concepts import load_bank
    from mirror.hf_model import extract_hf
    bank = load_bank("data/concepts/dev_bank.yaml")
    vec = extract_hf(hf_model, hf_tok, bank, bank.get("elephant"), 0, n_pairs=10)
    assert vec.concept == "elephant"
    assert vec.layer == 0
    assert torch.isclose(vec.direction.norm(), torch.tensor(1.0), atol=1e-5)
    assert set(vec.flags) == {"steering", "probe", "stability"}


def test_kl_hf_zero_at_alpha_zero(hf_model, hf_tok):
    from mirror.hf_model import kl_meter_hf
    vec = _tiny_vec(hf_model, hf_tok)
    assert abs(kl_meter_hf(hf_model, hf_tok, "hello world", vec, 0.0)) < 1e-4


def test_kl_hf_positive_under_injection(hf_model, hf_tok):
    from mirror.hf_model import kl_meter_hf
    vec = _tiny_vec(hf_model, hf_tok)
    assert kl_meter_hf(hf_model, hf_tok, "hello world", vec, 50.0) > 0.0


def test_run_hf_writes_records(hf_model, hf_tok, tmp_path):
    import json

    from mirror.hf_model import run_hf
    cfg = {
        "model": {"name": "tiny"},
        "injection": {"layer": 0, "alphas": [0, 4], "span": "response"},
        "concepts": {"bank": "data/concepts/dev_bank.yaml", "names": ["elephant"], "n_pairs": 10},
        "run": {"seeds": [0], "max_new_tokens": 6, "prompt": "hello", "out": str(tmp_path / "out.jsonl")},
    }
    records = run_hf(hf_model, hf_tok, cfg)
    lines = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(records) == len(lines) == 2
    expected = {"config", "concept", "layer", "alpha", "span", "seed", "kl", "flags", "clean", "report"}
    assert expected <= set(lines[0])
    assert {r["alpha"] for r in records} == {0, 4}
