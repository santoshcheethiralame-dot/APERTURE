from dataclasses import dataclass, field

import torch


@dataclass
class ConceptVector:
    concept: str
    layer: int
    direction: torch.Tensor
    sigma: float
    flags: dict = field(default_factory=dict)


def hook_name(layer):
    return f"blocks.{layer}.hook_resid_post"


def resid_stats(model, prompt, layer):
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt, names_filter=hook_name(layer))
    resid = cache[hook_name(layer)][0]
    return resid.mean(0), resid.norm(dim=-1).median()


def raw_direction(model, pairs, layer):
    positives, negatives, norms = [], [], []
    for positive, negative in pairs:
        p_mean, p_norm = resid_stats(model, positive, layer)
        n_mean, n_norm = resid_stats(model, negative, layer)
        positives.append(p_mean)
        negatives.append(n_mean)
        norms += [p_norm, n_norm]
    vector = torch.stack(positives).mean(0) - torch.stack(negatives).mean(0)
    return vector, torch.stack(norms).median().item()


def steering_check(model, concept, direction, sigma, layer, alpha=8.0):
    token = model.to_tokens(" " + concept.name, prepend_bos=False)[0, 0]
    prompt = "I am thinking about"

    def hook(resid, hook):
        resid += alpha * sigma * direction
        return resid

    with torch.no_grad():
        clean = model(prompt)[0, -1].log_softmax(-1)[token]
        with model.hooks(fwd_hooks=[(hook_name(layer), hook)]):
            steered = model(prompt)[0, -1].log_softmax(-1)[token]
    return bool(steered > clean)


def probe_check(model, pairs, direction, layer, threshold=0.9):
    wins = 0
    for positive, negative in pairs:
        p_mean, _ = resid_stats(model, positive, layer)
        n_mean, _ = resid_stats(model, negative, layer)
        wins += int(p_mean @ direction > n_mean @ direction)
    return wins / len(pairs) >= threshold


def stability_check(model, pairs, layer, threshold=0.8):
    half = len(pairs) // 2
    a, _ = raw_direction(model, pairs[:half], layer)
    b, _ = raw_direction(model, pairs[half:], layer)
    return bool(torch.cosine_similarity(a, b, dim=0) >= threshold)


def split_pairs(pairs, n_templates, held_out_templates=2):
    cut = n_templates - held_out_templates
    train = [p for i, p in enumerate(pairs) if i % n_templates < cut]
    test = [p for i, p in enumerate(pairs) if i % n_templates >= cut]
    return train, test


def extract(model, bank, concept, layer, n_pairs=40, seed=0):
    pairs = bank.pairs(concept, n_pairs, seed)
    train, test = split_pairs(pairs, len(bank.templates))
    vector, sigma = raw_direction(model, train, layer)
    direction = vector / vector.norm()
    flags = {
        "steering": steering_check(model, concept, direction, sigma, layer),
        "probe": probe_check(model, test, direction, layer),
        "stability": stability_check(model, train, layer),
    }
    return ConceptVector(concept.name, layer, direction, sigma, flags)
