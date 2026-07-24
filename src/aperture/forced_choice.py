import json
import re
from pathlib import Path

import numpy as np
from wordfreq import zipf_frequency

from aperture.hf_model import extract_hf, generate_hf


def option_prompt(names, order, template):
    listed = ", ".join(names[i] for i in order)
    return template.format(options=listed)


def parse_choice(answer, names):
    for token in re.findall(r"[a-z]+", answer.lower()):
        if token in names:
            return token
    return None


def report_hit_rate(records):
    parsed = [r for r in records if r["chosen"] is not None]
    if not parsed:
        return 0.0
    return sum(r["chosen"] == r["concept"] for r in parsed) / len(parsed)


def concept_frequencies(names):
    return {name: float(zipf_frequency(name, "en")) for name in names}


def concept_abstractness(bank, names):
    return {name: (1.0 if bank.get(name).category == "emotions" else 0.0)
            for name in names}


def build_features(names, records, freqs, abstract):
    usable = [r for r in records if r["chosen"] is not None]
    rows, targets = [], []
    for record in usable:
        rows.append([[freqs[name], abstract[name],
                      1.0 if name == record["concept"] else 0.0]
                     for name in names])
        targets.append(names.index(record["chosen"]))
    return np.array(rows, dtype="float32"), np.array(targets)


def collect_forced_choice_hf(model, tok, bank, names, layer, alpha, template,
                             answer_marker, n_orders=6, n_pairs=12,
                             max_new_tokens=8, out="forced.jsonl", seed=0):
    rng = np.random.default_rng(seed)
    vecs = {name: extract_hf(model, tok, bank, bank.get(name), layer, n_pairs)
            for name in names}
    orders = [list(rng.permutation(len(names))) for _ in range(n_orders)]
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a") as f:
        for i, name in enumerate(names):
            print(f"[{i + 1}/{len(names)}] {name}", flush=True)
            for order_index, order in enumerate(orders):
                prompt = option_prompt(names, order, template)
                report = generate_hf(model, tok, prompt, vecs[name], alpha,
                                     "response", max_new_tokens, 0)
                answer = report.rpartition(answer_marker)[2]
                record = {
                    "concept": name,
                    "order_index": order_index,
                    "chosen": parse_choice(answer, names),
                    "report": report,
                }
                f.write(json.dumps(record) + "\n")
                records.append(record)
    return {"records": records}
