import json
from pathlib import Path

import torch

from mirror.concepts import load_bank
from mirror.runner import config_hash
from mirror.vectors import ConceptVector, split_pairs


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


def _steering_hf(model, tok, concept, direction, sigma, layer, alpha=8.0):
    token = tok(" " + concept, add_special_tokens=False).input_ids[0]
    vec = ConceptVector(layer=layer, concept=concept, direction=direction,
                        sigma=sigma, flags={})
    ids = tok("I am thinking about", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        clean = model(ids).logits[0, -1].log_softmax(-1)[token]
        h = hf_layer(model, layer).register_forward_hook(inject_hook(vec, alpha, "response"))
        steered = model(ids).logits[0, -1].log_softmax(-1)[token]
        h.remove()
    return bool(steered > clean)


def _probe_hf(model, tok, pairs, direction, layer, threshold=0.9):
    wins = 0
    for positive, negative in pairs:
        p_mean, _ = resid_stats_hf(model, tok, positive, layer)
        n_mean, _ = resid_stats_hf(model, tok, negative, layer)
        wins += int(p_mean @ direction > n_mean @ direction)
    return wins / len(pairs) >= threshold


def _stability_hf(model, tok, pairs, layer, threshold=0.8):
    half = len(pairs) // 2
    a, _ = raw_direction_hf(model, tok, pairs[:half], layer)
    b, _ = raw_direction_hf(model, tok, pairs[half:], layer)
    return bool(torch.cosine_similarity(a, b, dim=0) >= threshold)


def extract_hf(model, tok, bank, concept, layer, n_pairs=40, seed=0):
    pairs = bank.pairs(concept, n_pairs, seed)
    train, test = split_pairs(pairs, len(bank.templates))
    vector, sigma = raw_direction_hf(model, tok, train, layer)
    direction = vector / vector.norm()
    flags = {
        "steering": _steering_hf(model, tok, concept.name, direction, sigma, layer),
        "probe": _probe_hf(model, tok, test, direction, layer),
        "stability": _stability_hf(model, tok, train, layer),
    }
    return ConceptVector(layer=layer, concept=concept.name, direction=direction,
                         sigma=sigma, flags=flags)


def kl_meter_hf(model, tok, prompt, vec, alpha, span="response"):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        clean = model(ids).logits[0, -1].log_softmax(-1)
        h = hf_layer(model, vec.layer).register_forward_hook(inject_hook(vec, alpha, span))
        injected = model(ids).logits[0, -1].log_softmax(-1)
        h.remove()
    return float(torch.sum(injected.exp() * (injected - clean)))


def run_hf(model, tok, cfg):
    bank = load_bank(cfg["concepts"]["bank"])
    injection_cfg, run_cfg = cfg["injection"], cfg["run"]
    out = Path(run_cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".config.json").write_text(json.dumps(cfg, indent=2))
    records = []
    with out.open("a") as f:
        for name in cfg["concepts"]["names"]:
            vec = extract_hf(model, tok, bank, bank.get(name),
                             injection_cfg["layer"], cfg["concepts"]["n_pairs"])
            for alpha in injection_cfg["alphas"]:
                kl = kl_meter_hf(model, tok, run_cfg["prompt"], vec, alpha,
                                 injection_cfg["span"])
                for seed in run_cfg["seeds"]:
                    clean = generate_hf(model, tok, run_cfg["prompt"], seed=seed,
                                        max_new_tokens=run_cfg["max_new_tokens"])
                    report = generate_hf(model, tok, run_cfg["prompt"], vec, alpha,
                                         injection_cfg["span"],
                                         run_cfg["max_new_tokens"], seed)
                    record = {
                        "config": config_hash(cfg),
                        "concept": name,
                        "layer": injection_cfg["layer"],
                        "alpha": alpha,
                        "span": injection_cfg["span"],
                        "seed": seed,
                        "kl": kl,
                        "flags": vec.flags,
                        "clean": clean,
                        "report": report,
                    }
                    f.write(json.dumps(record) + "\n")
                    records.append(record)
    return records


def load_hf(name, load_in_8bit=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=quant, device_map="auto", torch_dtype="auto")
    model.eval()
    return model, AutoTokenizer.from_pretrained(name)
