import pytest


@pytest.fixture(scope="session")
def bank():
    from mirror.concepts import load_bank
    return load_bank("data/concepts/dev_bank.yaml")
