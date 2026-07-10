import pytest


@pytest.fixture(scope="session")
def bank():
    from mirror.concepts import load_bank
    return load_bank("data/concepts/dev_bank.yaml")


@pytest.fixture(scope="session")
def model():
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained("pythia-70m")


@pytest.fixture(scope="session")
def vec(model, bank):
    from mirror.vectors import extract
    return extract(model, bank, bank.get("elephant"), layer=3, n_pairs=10)
