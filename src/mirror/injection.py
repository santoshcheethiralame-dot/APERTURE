import torch

from mirror.vectors import hook_name


def make_hook(vec, alpha, span):
    state = {"calls": 0}

    def hook(resid, hook):
        if span == "response" or state["calls"] == 0:
            resid[:, -1:] += alpha * vec.sigma * vec.direction
        state["calls"] += 1
        return resid

    return hook


def generate(model, prompt, vec=None, alpha=0.0, span="response",
             max_new_tokens=64, seed=0):
    torch.manual_seed(seed)
    hooks = []
    if vec is not None:
        hooks = [(hook_name(vec.layer), make_hook(vec, alpha, span))]
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        return model.generate(prompt, max_new_tokens=max_new_tokens, verbose=False)
