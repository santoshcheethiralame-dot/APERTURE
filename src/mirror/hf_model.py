import torch


def hf_layer(model, layer):
    return model.model.layers[layer]


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def resid_stats_hf(model, tok, prompt, layer):
    captured = {}

    def hook(module, inputs, output):
        captured["resid"] = _hidden(output).detach()

    handle = hf_layer(model, layer).register_forward_hook(hook)
    try:
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    resid = captured["resid"][0]
    return resid.mean(0), resid.norm(dim=-1).median()


def raw_direction_hf(model, tok, pairs, layer):
    positives, negatives, norms = [], [], []
    for positive, negative in pairs:
        p_mean, p_norm = resid_stats_hf(model, tok, positive, layer)
        n_mean, n_norm = resid_stats_hf(model, tok, negative, layer)
        positives.append(p_mean)
        negatives.append(n_mean)
        norms += [p_norm, n_norm]
    vector = torch.stack(positives).mean(0) - torch.stack(negatives).mean(0)
    return vector, torch.stack(norms).median().item()
