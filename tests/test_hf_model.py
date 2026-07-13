from mirror.hf_model import hf_layer


def test_hf_layer_returns_decoder_block(hf_model):
    layer = hf_layer(hf_model, 0)
    assert layer is hf_model.model.layers[0]
