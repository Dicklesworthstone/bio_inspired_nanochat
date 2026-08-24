"""
Results corpus + experiment registry — bead hm4.1.

Locks: provenance-stamped, schema-valid RunRecords; JSONL append/read round-trip; best-by-metric
respecting the schema's optimization direction; the query CLI. Reuses checkpoint_manager
provenance + the metrics_schema (hm4.2) validation.

Run:  pytest tests/test_results_registry.py -v
"""

from __future__ import annotations

import json

import pytest

from bio_inspired_nanochat.metrics_schema import UnknownMetricError
from bio_inspired_nanochat.results_registry import (
    DEFAULT_REGISTRY,
    RunRecord,
    _main,
    append_record,
    best_record,
    make_record,
    read_records,
    summarize,
)
from bio_inspired_nanochat.synaptic import SynapticConfig


@pytest.mark.unit
def test_make_record_stamps_provenance_and_validates_metrics():
    cfg = SynapticConfig(tau_c=7.0)
    rec = make_record(
        "train", {"train_loss": 4.5, "val_bpb": 1.2},
        run_id="r1", syn_cfg=cfg, seed=42, dataset_shards=["shard0"], timestamp=1000.0,
    )
    assert rec.harness == "train" and rec.run_id == "r1" and rec.seed == 42
    assert rec.metrics == {"train_loss": 4.5, "val_bpb": 1.2}
    assert rec.config_hash is not None and len(rec.config_hash) == 16
    assert rec.hardware and rec.timestamp == 1000.0
    assert rec.dataset_shards == ["shard0"]
    assert rec.git_sha is None or len(rec.git_sha) == 40  # SHA in a repo, else None


@pytest.mark.unit
def test_make_record_rejects_unknown_metric():
    with pytest.raises(UnknownMetricError):
        make_record("train", {"made_up_metric": 1.0}, run_id="r")


@pytest.mark.unit
def test_make_record_rejects_unknown_harness():
    with pytest.raises(ValueError, match="unknown harness"):
        make_record("nope", {"train_loss": 4.5}, run_id="r")


@pytest.mark.unit
def test_default_registry_is_a_committable_results_path():
    assert DEFAULT_REGISTRY == "results/registry.jsonl"


@pytest.mark.unit
def test_append_and_read_roundtrip(tmp_path):
    path = str(tmp_path / "registry.jsonl")
    append_record(make_record("train", {"val_bpb": 1.5}, run_id="a", timestamp=1.0), path)
    append_record(make_record("eval", {"eval_bpb": 1.1}, run_id="b", timestamp=2.0), path)
    recs = read_records(path)
    assert [r.run_id for r in recs] == ["a", "b"]
    assert recs[0].metrics == {"val_bpb": 1.5} and recs[1].harness == "eval"


@pytest.mark.unit
def test_read_missing_registry_is_empty(tmp_path):
    assert read_records(str(tmp_path / "nope.jsonl")) == []


@pytest.mark.unit
def test_read_registry_reports_corrupt_line(tmp_path):
    path = tmp_path / "registry.jsonl"
    path.write_text(
        '{"run_id":"ok","harness":"eval","metrics":{}}\n{broken}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"registry\.jsonl:2"):
        read_records(str(path))


@pytest.mark.unit
def test_best_record_respects_optimization_direction():
    recs = [
        make_record("eval", {"val_bpb": 1.5}, run_id="hi", timestamp=1.0),
        make_record("eval", {"val_bpb": 1.1}, run_id="lo", timestamp=2.0),       # lower better
        make_record("eval", {"eval_accuracy": 0.7}, run_id="a", timestamp=3.0),
        make_record("eval", {"eval_accuracy": 0.9}, run_id="b", timestamp=4.0),  # higher better
    ]
    assert best_record(recs, "val_bpb").run_id == "lo"
    assert best_record(recs, "eval_accuracy").run_id == "b"
    assert best_record([], "val_bpb") is None
    with pytest.raises(KeyError):
        best_record(recs, "not_a_metric")


@pytest.mark.unit
def test_record_json_roundtrip():
    rec = make_record("tune", {"tune_objective": 1.2, "tune_generation": 3}, run_id="t", timestamp=5.0)
    rec2 = RunRecord.from_json(json.loads(json.dumps(rec.to_json())))
    assert rec2 == rec


@pytest.mark.unit
def test_summarize_and_cli(tmp_path, capsys):
    path = str(tmp_path / "registry.jsonl")
    append_record(make_record("train", {"val_bpb": 1.3}, run_id="x", timestamp=1.0), path)
    assert "val_bpb" in summarize(read_records(path))
    assert summarize([]) == "(no runs in the registry)"

    assert _main(["list", "--path", path]) == 0
    assert "val_bpb" in capsys.readouterr().out
    assert _main(["best", "--path", path, "--metric", "val_bpb"]) == 0
    assert "best by val_bpb" in capsys.readouterr().out
