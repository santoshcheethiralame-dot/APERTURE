import hashlib
import json
from pathlib import Path

import yaml

from mirror.concepts import load_bank
from mirror.injection import generate
from mirror.metrics import kl_meter
from mirror.vectors import extract


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def config_hash(cfg):
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:12]


def run(model, cfg):
    bank = load_bank(cfg["concepts"]["bank"])
    injection_cfg, run_cfg = cfg["injection"], cfg["run"]
    out = Path(run_cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".config.json").write_text(json.dumps(cfg, indent=2))
    records = []
    with out.open("a") as f:
        for name in cfg["concepts"]["names"]:
            vec = extract(model, bank, bank.get(name),
                          injection_cfg["layer"], cfg["concepts"]["n_pairs"])
            for alpha in injection_cfg["alphas"]:
                kl = kl_meter(model, run_cfg["prompt"], vec, alpha,
                              injection_cfg["span"])
                for seed in run_cfg["seeds"]:
                    clean = generate(model, run_cfg["prompt"], seed=seed,
                                     max_new_tokens=run_cfg["max_new_tokens"])
                    report = generate(model, run_cfg["prompt"], vec, alpha,
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
