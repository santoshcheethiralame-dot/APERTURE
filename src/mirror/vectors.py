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


def extract(model, bank, concept, layer, n_pairs=40, seed=0):
    pairs = bank.pairs(concept, n_pairs, seed)
    vector, sigma = raw_direction(model, pairs, layer)
    return ConceptVector(concept.name, layer, vector / vector.norm(), sigma)
