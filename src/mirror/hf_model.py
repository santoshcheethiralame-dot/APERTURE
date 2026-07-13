import torch

from mirror.vectors import ConceptVector


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


def extract_hf_vector(model, tok, pairs, layer):
    vector, sigma = raw_direction_hf(model, tok, pairs, layer)
    return ConceptVector(layer=layer, concept="", direction=vector / vector.norm(),
                         sigma=sigma, flags={})


def inject_hook(vec, alpha, span):
    state = {"calls": 0}

    def hook(module, inputs, output):
        if span == "response" or state["calls"] == 0:
            hidden = _hidden(output)
            hidden[:, -1:] += (alpha * vec.sigma * vec.direction).to(hidden.device)
        state["calls"] += 1
        return output

    return hook


def generate_hf(model, tok, prompt, vec=None, alpha=0.0, span="response",
                max_new_tokens=64, seed=0):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    handle = None
    if vec is not None:
        handle = hf_layer(model, vec.layer).register_forward_hook(inject_hook(vec, alpha, span))
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        if handle is not None:
            handle.remove()
    return tok.decode(out[0], skip_special_tokens=True)
