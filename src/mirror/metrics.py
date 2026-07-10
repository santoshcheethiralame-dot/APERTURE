import torch

from mirror.injection import make_hook
from mirror.vectors import hook_name


def kl_meter(model, prompt, vec, alpha, span="response"):
    with torch.no_grad():
        clean = model(prompt)[0, -1].log_softmax(-1)
        hook = make_hook(vec, alpha, span)
        with model.hooks(fwd_hooks=[(hook_name(vec.layer), hook)]):
            injected = model(prompt)[0, -1].log_softmax(-1)
    return torch.sum(injected.exp() * (injected - clean)).item()
