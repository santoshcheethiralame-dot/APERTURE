from pathlib import Path

import torch
import yaml

from mirror.hf_model import _hidden, hf_layer


def load_contexts(path="data/concepts/contexts.yaml"):
    raw = yaml.safe_load(Path(path).read_text())
    for name, passage in raw.items():
        if name in passage.lower():
            raise ValueError(f"context for {name} names its own concept")
    return raw


def last_activation_hf(model, tok, prompt, layer):
    captured = {}

    def grab(module, inputs, output):
        captured["resid"] = _hidden(output).detach()

    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    handle = hf_layer(model, layer).register_forward_hook(grab)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return captured["resid"][0, -1]


def nearest_concept(activation, directions):
    scores = {name: float(activation.float() @ d.float().to(activation.device))
              for name, d in directions.items()}
    return max(scores, key=scores.get)
