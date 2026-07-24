import pytest

from aperture.concepts import load_bank


def test_bank_loads(bank):
    assert len(bank.concepts) == 16
    assert len({c.category for c in bank.concepts}) == 4


def test_pairs_are_category_matched(bank):
    pairs = bank.pairs(bank.get("elephant"), n_pairs=12)
    assert len(pairs) == 12
    categories = {c.name: c.category for c in bank.concepts}
    for positive, negative in pairs:
        assert "elephant" in positive
        negative_name = next(n for n in categories if n in negative)
        assert categories[negative_name] == "animals"
        assert negative_name != "elephant"


def test_pairs_deterministic(bank):
    concept = bank.get("joy")
    assert bank.pairs(concept, seed=3) == bank.pairs(concept, seed=3)


def test_bad_template_rejected(tmp_path):
    bad = tmp_path / "bank.yaml"
    bad.write_text(
        "concepts:\n"
        "- {name: a, category: x}\n"
        "- {name: b, category: x}\n"
        "templates:\n"
        "- no slot here\n"
    )
    with pytest.raises(ValueError):
        load_bank(bad)


def test_thin_category_rejected(tmp_path):
    bad = tmp_path / "bank.yaml"
    bad.write_text(
        "concepts:\n"
        "- {name: a, category: x}\n"
        "templates:\n"
        "- about {concept}\n"
    )
    with pytest.raises(ValueError):
        load_bank(bad)
