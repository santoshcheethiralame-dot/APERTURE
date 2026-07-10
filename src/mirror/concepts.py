import random
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Concept:
    name: str
    category: str


@dataclass(frozen=True)
class Bank:
    concepts: tuple
    templates: tuple

    def pairs(self, concept, n_pairs=40, seed=0):
        rng = random.Random(seed)
        negatives = [
            c for c in self.concepts
            if c.category == concept.category and c.name != concept.name
        ]
        out = []
        for i in range(n_pairs):
            template = self.templates[i % len(self.templates)]
            negative = rng.choice(negatives)
            out.append((
                template.format(concept=concept.name),
                template.format(concept=negative.name),
            ))
        return out

    def get(self, name):
        return next(c for c in self.concepts if c.name == name)


def load_bank(path):
    raw = yaml.safe_load(Path(path).read_text())
    concepts = tuple(Concept(**c) for c in raw["concepts"])
    templates = tuple(raw["templates"])
    if any("{concept}" not in t for t in templates):
        raise ValueError("every template needs a {concept} slot")
    counts = {}
    for c in concepts:
        counts[c.category] = counts.get(c.category, 0) + 1
    thin = [category for category, n in counts.items() if n < 2]
    if thin:
        raise ValueError(f"categories need at least 2 concepts: {thin}")
    return Bank(concepts, templates)
