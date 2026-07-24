import json

from aperture.runner import config_hash, run


def make_cfg(tmp_path):
    return {
        "model": {"name": "pythia-70m"},
        "injection": {"layer": 3, "alphas": [0, 8], "span": "response"},
        "concepts": {
            "bank": "data/concepts/dev_bank.yaml",
            "names": ["elephant"],
            "n_pairs": 10,
        },
        "run": {
            "seeds": [0],
            "max_new_tokens": 8,
            "prompt": "Anything odd?",
            "out": str(tmp_path / "out.jsonl"),
        },
    }


def test_config_hash_deterministic(tmp_path):
    assert config_hash(make_cfg(tmp_path)) == config_hash(make_cfg(tmp_path))
    assert len(config_hash(make_cfg(tmp_path))) == 12


def test_run_writes_records(model, tmp_path):
    cfg = make_cfg(tmp_path)
    records = run(model, cfg)
    lines = [json.loads(line) for line in
             (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(lines) == len(records) == 2
    expected = {"config", "concept", "layer", "alpha", "span", "seed",
                "kl", "flags", "clean", "report"}
    assert expected <= set(lines[0])
    assert {r["alpha"] for r in records} == {0, 8}
    assert json.loads((tmp_path / "out.config.json").read_text()) == cfg
