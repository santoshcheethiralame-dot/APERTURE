import json
from pathlib import Path

import torch

from aperture.hf_model import _hidden, extract_hf, hf_layer, probe_activation_hf


def concept_token(tok, name):
    return int(tok(" " + name, add_special_tokens=False).input_ids[0])


def baseline_logprob(model, tok, prompt, target_token):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    return float(logits.log_softmax(-1)[target_token])


def patched_logprob(model, tok, prompt, patch_layer, patch_resid, target_token):
    def hook(module, inputs, output):
        hidden = _hidden(output)
        hidden[:, -1] = patch_resid.to(hidden.device)
        return output

    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    handle = hf_layer(model, patch_layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
    finally:
        handle.remove()
    return float(logits.log_softmax(-1)[target_token])


def patch_effect_hf(model, tok, prompt, vec, alpha, patch_layer, target_token):
    patch_resid = probe_activation_hf(model, tok, prompt, vec, alpha, patch_layer)
    baseline_lp = baseline_logprob(model, tok, prompt, target_token)
    patched_lp = patched_logprob(model, tok, prompt, patch_layer, patch_resid,
                                 target_token)
    return baseline_lp, patched_lp, patched_lp - baseline_lp


def collect_patch_hf(model, tok, bank, names, layer, patch_layer, alpha, prompt,
                     out, n_pairs=12):
    vecs = {name: extract_hf(model, tok, bank, bank.get(name), layer, n_pairs)
            for name in names}
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a") as f:
        for i, name in enumerate(names):
            target = concept_token(tok, name)
            other = names[(i + 1) % len(names)]
            _, _, self_delta = patch_effect_hf(model, tok, prompt, vecs[name],
                                               alpha, patch_layer, target)
            _, _, control_delta = patch_effect_hf(model, tok, prompt, vecs[other],
                                                  alpha, patch_layer, target)
            record = {
                "concept": name,
                "layer": layer,
                "patch_layer": patch_layer,
                "alpha": alpha,
                "self_delta": self_delta,
                "control_delta": control_delta,
            }
            f.write(json.dumps(record) + "\n")
            records.append(record)
    return {"records": records}
