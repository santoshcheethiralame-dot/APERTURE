import numpy as np

from mirror.concepts import load_bank
from mirror.forced_choice import (
    build_features,
    concept_abstractness,
    concept_frequencies,
    option_prompt,
    parse_choice,
)

TEMPLATE = "Pick one: {options}\nAnswer:"
NAMES = ["elephant", "volcano", "joy"]
FREQS = {"elephant": 4.0, "volcano": 3.5, "joy": 5.0}
ABSTRACT = {"elephant": 0.0, "volcano": 0.0, "joy": 1.0}


def test_frequencies_are_positive_floats():
    freqs = concept_frequencies(["elephant", "joy"])
    assert freqs["elephant"] > 0
    assert freqs["joy"] > 0


def test_frequent_word_scores_higher():
    freqs = concept_frequencies(["joy", "jealousy"])
    assert freqs["joy"] > freqs["jealousy"]


def test_abstractness_flags_emotions():
    bank = load_bank("data/concepts/dev_bank.yaml")
    abstract = concept_abstractness(bank, ["joy", "elephant"])
    assert abstract["joy"] == 1.0
    assert abstract["elephant"] == 0.0


def test_option_prompt_lists_all_names_in_order():
    prompt = option_prompt(["elephant", "volcano", "joy"], [2, 0, 1], TEMPLATE)
    assert "joy, elephant, volcano" in prompt
    assert prompt.startswith("Pick one:")


def test_parse_choice_finds_named_concept():
    assert parse_choice("Volcano.", ["elephant", "volcano", "joy"]) == "volcano"


def test_parse_choice_returns_none_when_absent():
    assert parse_choice("I refuse to answer.", ["elephant", "volcano"]) is None


def test_build_features_shapes_and_injected_column():
    records = [
        {"concept": "volcano", "chosen": "joy"},
        {"concept": "joy", "chosen": "joy"},
    ]
    X, y = build_features(NAMES, records, FREQS, ABSTRACT)
    assert X.shape == (2, 3, 3)
    assert y.tolist() == [2, 2]
    assert X[0, :, -1].tolist() == [0.0, 1.0, 0.0]
    assert X[1, :, -1].tolist() == [0.0, 0.0, 1.0]


def test_build_features_drops_unparsed():
    records = [
        {"concept": "volcano", "chosen": None},
        {"concept": "joy", "chosen": "elephant"},
    ]
    X, y = build_features(NAMES, records, FREQS, ABSTRACT)
    assert X.shape == (1, 3, 3)
    assert y.tolist() == [0]


def test_build_features_carries_covariates():
    records = [{"concept": "joy", "chosen": "joy"}]
    X, y = build_features(NAMES, records, FREQS, ABSTRACT)
    assert X[0, 2, 0] == 5.0
    assert X[0, 2, 1] == 1.0


def _many_names(n):
    return [f"c{i}" for i in range(n)]


def test_gamma_positive_when_chooser_is_perfect():
    from mirror.prior_null import fit
    names = _many_names(8)
    freqs = {n: 3.0 for n in names}
    abstract = {n: 0.0 for n in names}
    rng = np.random.default_rng(0)
    records = []
    for _ in range(300):
        injected = names[rng.integers(0, len(names))]
        records.append({"concept": injected, "chosen": injected})
    X, y = build_features(names, records, freqs, abstract)
    assert fit(X, y).gamma > 2.0


def test_gamma_near_zero_when_chooser_is_random():
    from mirror.prior_null import fit
    names = _many_names(8)
    freqs = {n: 3.0 for n in names}
    abstract = {n: 0.0 for n in names}
    rng = np.random.default_rng(1)
    records = []
    for _ in range(300):
        injected = names[rng.integers(0, len(names))]
        chosen = names[rng.integers(0, len(names))]
        records.append({"concept": injected, "chosen": chosen})
    X, y = build_features(names, records, freqs, abstract)
    assert abs(fit(X, y).gamma) < 0.6


def test_collect_forced_choice_writes_records(hf_model, hf_tok, tmp_path):
    import json

    from mirror.forced_choice import collect_forced_choice_hf
    bank = load_bank("data/concepts/dev_bank.yaml")
    out = tmp_path / "forced.jsonl"
    result = collect_forced_choice_hf(hf_model, hf_tok, bank,
                                      ["elephant", "volcano", "joy"], 0, 1.0,
                                      TEMPLATE, "Answer:", n_orders=2, n_pairs=10,
                                      max_new_tokens=4, out=str(out))
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(result["records"]) == len(lines) == 6
    assert {"concept", "order_index", "chosen", "report"} <= set(lines[0])
    assert all(r["chosen"] is None or r["chosen"] in ["elephant", "volcano", "joy"]
               for r in result["records"])


def test_hit_rate_counts_matches_over_parsed():
    from mirror.forced_choice import report_hit_rate
    records = [
        {"concept": "joy", "chosen": "joy"},
        {"concept": "joy", "chosen": "fear"},
        {"concept": "volcano", "chosen": None},
        {"concept": "volcano", "chosen": "volcano"},
    ]
    assert report_hit_rate(records) == 2 / 3


def test_hit_rate_zero_when_none_parsed():
    from mirror.forced_choice import report_hit_rate
    records = [{"concept": "joy", "chosen": None}]
    assert report_hit_rate(records) == 0.0


def test_collect_forced_choice_parses_only_the_answer(hf_model, hf_tok, tmp_path):
    from mirror.forced_choice import collect_forced_choice_hf
    bank = load_bank("data/concepts/dev_bank.yaml")
    out = tmp_path / "order.jsonl"
    result = collect_forced_choice_hf(hf_model, hf_tok, bank,
                                      ["elephant", "volcano", "joy"], 0, 1.0,
                                      TEMPLATE, "Answer:", n_orders=3, n_pairs=10,
                                      max_new_tokens=4, out=str(out))
    chosen = [r["chosen"] for r in result["records"]]
    assert not all(c == "elephant" for c in chosen)
