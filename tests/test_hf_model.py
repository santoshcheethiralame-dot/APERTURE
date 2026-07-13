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
