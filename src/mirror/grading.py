import re
from abc import ABC, abstractmethod
from pathlib import Path

import yaml


def words(text):
    return re.findall(r"[a-z]+", text.lower())


class Grader(ABC):
    @abstractmethod
    def grade(self, concept, report):
        ...


class RulesGrader(Grader):
    def __init__(self, synonyms=None):
        self.synonyms = synonyms if synonyms is not None else load_synonyms()

    def grade(self, concept, report):
        toks = words(report)
        token_set = set(toks)
        entry = self.synonyms[concept]
        exact = [t for t in entry["exact"] if t in token_set]
        related = [t for t in entry["related"] if t in token_set]
        if exact:
            identified, matched = "exact", exact
        elif related:
            identified, matched = "related", related
        else:
            identified, matched = "no", []
        detected = toks[0] if toks and toks[0] in ("yes", "no") else None
        return {"detected": detected, "identified": identified, "matched": matched}


def load_synonyms(path="data/concepts/synonyms.yaml"):
    raw = yaml.safe_load(Path(path).read_text())
    from mirror.concepts import load_bank
    bank = load_bank("data/concepts/dev_bank.yaml")
    for concept in bank.concepts:
        if concept.name not in raw:
            raise ValueError(f"missing synonyms for concept: {concept.name}")
    return raw
