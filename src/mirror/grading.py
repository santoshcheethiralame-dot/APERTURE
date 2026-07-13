import re
from pathlib import Path

import yaml


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def load_synonyms(path="data/concepts/synonyms.yaml"):
    raw = yaml.safe_load(Path(path).read_text())
    from mirror.concepts import load_bank
    bank = load_bank("data/concepts/dev_bank.yaml")
    for concept in bank.concepts:
        if concept.name not in raw:
            raise ValueError(f"missing synonyms for concept: {concept.name}")
    return raw
