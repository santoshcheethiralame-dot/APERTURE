import re

import numpy as np
from wordfreq import zipf_frequency


def option_prompt(names, order, template):
    listed = ", ".join(names[i] for i in order)
    return template.format(options=listed)


def parse_choice(answer, names):
    for token in re.findall(r"[a-z]+", answer.lower()):
        if token in names:
            return token
    return None


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
