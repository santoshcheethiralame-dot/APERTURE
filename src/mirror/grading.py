import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
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


def strip_prompt(report):
    for marker in ("<start_of_turn>model\n", "model\n"):
        if marker in report:
            return report.split(marker)[-1]
    return report


def grade_file(in_path, out_path, grader=None):
    grader = grader if grader is not None else RulesGrader()
    records = []
    for line in Path(in_path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        grade = grader.grade(record["concept"], strip_prompt(record["report"]))
        record.update(grade)
        records.append(record)
    Path(out_path).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return records


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["concept"], r["alpha"])].append(r)
    rows = []
    for (concept, alpha), items in groups.items():
        kls = [i["kl"] for i in items]
        rows.append({
            "concept": concept,
            "alpha": alpha,
            "n": len(items),
            "exact": sum(i["identified"] == "exact" for i in items),
            "related": sum(i["identified"] == "related" for i in items),
            "no": sum(i["identified"] == "no" for i in items),
            "detected_yes": sum(i["detected"] == "yes" for i in items),
            "mean_kl": sum(kls) / len(kls),
        })
    rows.sort(key=lambda r: (r["concept"], r["alpha"]))
    return rows


def load_synonyms(path="data/concepts/synonyms.yaml"):
    raw = yaml.safe_load(Path(path).read_text())
    from mirror.concepts import load_bank
    bank = load_bank("data/concepts/dev_bank.yaml")
    for concept in bank.concepts:
        if concept.name not in raw:
            raise ValueError(f"missing synonyms for concept: {concept.name}")
    return raw
