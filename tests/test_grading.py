from mirror.grading import load_synonyms


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
