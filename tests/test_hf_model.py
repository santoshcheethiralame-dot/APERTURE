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
