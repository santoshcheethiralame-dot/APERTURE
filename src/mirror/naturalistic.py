import json
from pathlib import Path

import torch
import yaml

from mirror.grading import RulesGrader
from mirror.hf_model import _hidden, extract_hf, generate_hf, hf_layer


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


def collect_naturalistic_hf(model, tok, bank, contexts, distractor, report_suffix,
                            layer, n_pairs=12, max_new_tokens=16,
                            out="naturalistic.jsonl"):
    directions = {name: extract_hf(model, tok, bank, bank.get(name), layer,
                                   n_pairs).direction
                  for name in contexts}
    grader = RulesGrader()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a") as f:
        for name, passage in contexts.items():
            print(f"[{len(records) + 1}/{len(contexts)}] {name}", flush=True)
            act = last_activation_hf(model, tok, passage, layer)
            predicted = nearest_concept(act, directions)
            prompt = passage + distractor + report_suffix
            report = generate_hf(model, tok, prompt, max_new_tokens=max_new_tokens)
            answer = report.rpartition("A:")[2]
            record = {
                "concept": name,
                "predicted": predicted,
                "report": report,
                "identified": grader.grade(name, answer)["identified"],
            }
            f.write(json.dumps(record) + "\n")
            records.append(record)
    return {"records": records}
