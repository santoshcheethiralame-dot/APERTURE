from mirror.grading import RulesGrader, load_synonyms, words


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
