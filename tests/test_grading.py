import json

from mirror.grading import (
    RulesGrader,
    grade_file,
    load_synonyms,
    strip_prompt,
    summarize,
    words,
)


def test_synonyms_cover_dev_bank():
    from mirror.concepts import load_bank
    syn = load_synonyms()
    bank = load_bank("data/concepts/dev_bank.yaml")
    for concept in bank.concepts:
        assert concept.name in syn
        assert syn[concept.name]["exact"]
        assert concept.name in syn[concept.name]["exact"]


def test_synonyms_lowercase():
    syn = load_synonyms()
    for entry in syn.values():
        for term in entry["exact"] + entry["related"]:
            assert term == term.lower()


def test_words_splits_and_lowercases():
    assert words("YES, Elephant!") == ["yes", "elephant"]


def test_words_respects_boundaries():
    toks = words("I really enjoy astronomy")
    assert "joy" not in toks
    assert "enjoy" in toks
    assert words("telescope") == ["telescope"]
    assert "scope" not in words("telescope")


def test_identifies_exact():
    result = RulesGrader().grade("elephant", "It is an elephant, clearly.")
    assert result["identified"] == "exact"
    assert "elephant" in result["matched"]


def test_identifies_related():
    result = RulesGrader().grade("volcano", "I sense lava and an eruption.")
    assert result["identified"] == "related"
    assert set(result["matched"]) & {"lava", "eruption"}


def test_identifies_none():
    result = RulesGrader().grade("telescope", "As an AI I have no thoughts.")
    assert result["identified"] == "no"
    assert result["matched"] == []


def test_exact_beats_related():
    result = RulesGrader().grade("volcano", "The volcano spewed lava.")
    assert result["identified"] == "exact"


def test_detects_yes():
    result = RulesGrader().grade("elephant", "YES, elephant.")
    assert result["detected"] == "yes"


def test_detects_no():
    result = RulesGrader().grade("elephant", "NO, nothing unusual.")
    assert result["detected"] == "no"


def test_detects_none_when_open_ended():
    result = RulesGrader().grade("elephant", "As an AI, I don't have thoughts.")
    assert result["detected"] is None


def test_strip_prompt():
    text = "user\nq<start_of_turn>model\n YES, elephant."
    assert strip_prompt(text).strip() == "YES, elephant."


def test_grade_file(tmp_path):
    src = tmp_path / "run.jsonl"
    records = [
        {"concept": "elephant", "alpha": 1, "seed": 0, "kl": 5.0,
         "report": "x<start_of_turn>model\nYES, elephant."},
        {"concept": "volcano", "alpha": 0, "seed": 0, "kl": 0.0,
         "report": "x<start_of_turn>model\nNO, nothing."},
    ]
    src.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    out = tmp_path / "graded.jsonl"
    graded = grade_file(src, out)
    assert graded[0]["identified"] == "exact"
    assert graded[0]["detected"] == "yes"
    assert graded[1]["identified"] == "no"
    assert graded[1]["detected"] == "no"
    lines = out.read_text().splitlines()
    assert json.loads(lines[0])["matched"] == ["elephant"]


def test_summarize_groups_and_counts():
    records = [
        {"concept": "joy", "alpha": 1, "kl": 6.0, "identified": "exact", "detected": "yes"},
        {"concept": "joy", "alpha": 1, "kl": 4.0, "identified": "no", "detected": "no"},
        {"concept": "joy", "alpha": 0, "kl": 0.0, "identified": "no", "detected": "no"},
    ]
    rows = summarize(records)
    a1 = next(r for r in rows if r["concept"] == "joy" and r["alpha"] == 1)
    assert a1["n"] == 2
    assert a1["exact"] == 1
    assert a1["no"] == 1
    assert a1["detected_yes"] == 1
    assert a1["mean_kl"] == 5.0
    assert [r["alpha"] for r in rows] == [0, 1]
