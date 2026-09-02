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
    TRACKED_REGISTRY,
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
def test_direct_record_construction_enforces_schema_and_finiteness():
    with pytest.raises(ValueError, match="not finite"):
        RunRecord("nan", "eval", {"val_bpb": float("nan")})
    with pytest.raises(UnknownMetricError):
        RunRecord("unknown", "eval", {"made_up_metric": 1.0})
    with pytest.raises(ValueError, match="unknown harness"):
        RunRecord("bad-harness", "other", {"val_bpb": 1.0})
    with pytest.raises(ValueError, match="run_id"):
        RunRecord(" ", "eval", {"val_bpb": 1.0})


@pytest.mark.unit
def test_make_record_rejects_unknown_harness():
    with pytest.raises(ValueError, match="unknown harness"):
        make_record("nope", {"train_loss": 4.5}, run_id="r")


@pytest.mark.unit
def test_make_record_hashes_full_mapping_config():
    first = make_record(
        "eval",
        {"eval_bpb": 1.2},
        run_id="first",
        config={"model": {"depth": 4}, "data": "fineweb"},
    )
    same = make_record(
        "eval",
        {"eval_bpb": 1.1},
        run_id="same",
        config={"data": "fineweb", "model": {"depth": 4}},
    )
    changed = make_record(
        "eval",
        {"eval_bpb": 1.1},
        run_id="changed",
        config={"model": {"depth": 8}, "data": "fineweb"},
    )
    assert first.config_hash == same.config_hash
    assert first.config_hash != changed.config_hash


@pytest.mark.unit
def test_make_record_rejects_ambiguous_config_and_empty_run_id():
    with pytest.raises(ValueError, match="either config or syn_cfg"):
        make_record(
            "train",
            {"train_loss": 1.0},
            run_id="r",
            config={"depth": 4},
            syn_cfg=SynapticConfig(),
        )
    with pytest.raises(ValueError, match="run_id must be non-empty"):
        make_record("train", {"train_loss": 1.0}, run_id="  ")
    with pytest.raises(ValueError, match="verdict must be"):
        make_record("eval", {"eval_bpb": 1.0}, run_id="bad", verdict="maybe")
    with pytest.raises(ValueError, match="cannot be eligible"):
        make_record("eval", {"eval_bpb": 1.0}, run_id="null", verdict="null")


@pytest.mark.unit
def test_default_registry_is_a_committable_results_path():
    # Outside the test suite the harnesses append to the tracked corpus; under pytest the
    # conftest redirects the default (BIO_RESULTS_REGISTRY) so tests never pollute it.
    assert TRACKED_REGISTRY == "results/registry.jsonl"
    assert DEFAULT_REGISTRY != TRACKED_REGISTRY
    assert DEFAULT_REGISTRY.endswith("registry.jsonl")


@pytest.mark.unit
def test_append_and_read_roundtrip(tmp_path):
    path = str(tmp_path / "registry.jsonl")
    append_record(make_record("train", {"val_bpb": 1.5}, run_id="a", timestamp=1.0), path)
    append_record(make_record("eval", {"eval_bpb": 1.1}, run_id="b", timestamp=2.0), path)
    recs = read_records(path)
    assert [r.run_id for r in recs] == ["a", "b"]
    assert recs[0].metrics == {"val_bpb": 1.5} and recs[1].harness == "eval"


@pytest.mark.unit
def test_append_revalidates_mutated_record_before_any_io(tmp_path):
    path = tmp_path / "nested" / "registry.jsonl"
    record = make_record("eval", {"val_bpb": 1.0}, run_id="mutated")
    record.metrics["val_bpb"] = float("nan")

    with pytest.raises(ValueError, match="not finite"):
        append_record(record, str(path))

    assert not path.exists()


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
def test_read_registry_rejects_unknown_fields_with_line_context(tmp_path):
    path = tmp_path / "registry.jsonl"
    path.write_text(
        json.dumps(
            {
                "run_id": "typo",
                "harness": "eval",
                "metrics": {"val_bpb": 1.0},
                "eligible_for_bset": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"registry\.jsonl:1:.*eligible_for_bset",
    ):
        read_records(str(path))


@pytest.mark.unit
def test_read_registry_rejects_nonstandard_json_constant_with_line_context(tmp_path):
    path = tmp_path / "registry.jsonl"
    path.write_text(
        '{"run_id":"bad","harness":"eval","metrics":{"val_bpb":NaN}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"registry\.jsonl:1:.*non-standard JSON constant"):
        read_records(str(path))


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        '{"run_id":"first","run_id":"second","harness":"eval","metrics":{}}',
        '{"run_id":"nested","harness":"eval","metrics":{"val_bpb":1,"val_bpb":2}}',
    ],
)
def test_read_registry_rejects_duplicate_json_fields_with_line_context(tmp_path, line):
    path = tmp_path / "registry.jsonl"
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"registry\.jsonl:1:.*duplicate JSON field"):
        read_records(str(path))


@pytest.mark.unit
def test_best_record_respects_optimization_direction():
    recs = [
        make_record("eval", {"val_bpb": 1.5}, run_id="hi", timestamp=1.0),
        make_record("eval", {"val_bpb": 1.1}, run_id="lo", timestamp=2.0),       # lower better
        make_record("eval", {"eval_accuracy": 0.7}, run_id="a", timestamp=3.0),
        make_record("eval", {"eval_accuracy": 0.9}, run_id="b", timestamp=4.0),  # higher better
    ]
    best_bpb = best_record(recs, "val_bpb")
    best_accuracy = best_record(recs, "eval_accuracy")
    assert best_bpb is not None and best_bpb.run_id == "lo"
    assert best_accuracy is not None and best_accuracy.run_id == "b"
    assert best_record([], "val_bpb") is None
    with pytest.raises(KeyError):
        best_record(recs, "not_a_metric")


@pytest.mark.unit
def test_best_record_rejects_neutral_metrics():
    records = [make_record("train", {"step": 1}, run_id="one")]

    with pytest.raises(ValueError, match="neutral direction"):
        best_record(records, "step")


@pytest.mark.unit
def test_performance_registry_metrics_are_schema_valid():
    record = RunRecord(
        "perf",
        "eval",
        {"tok_per_sec": 100.0, "latency_ms": 2.0, "memory_mb": 10.0},
    )

    assert record.metrics == {
        "tok_per_sec": 100.0,
        "latency_ms": 2.0,
        "memory_mb": 10.0,
    }


@pytest.mark.unit
def test_best_record_excludes_ineligible_falsification_results():
    invalidated = make_record(
        "eval",
        {"eval_accuracy": 1.0},
        run_id="invalidated",
        verdict="invalidated",
        eligible_for_best=False,
    )
    positive = make_record(
        "eval",
        {"eval_accuracy": 0.8},
        run_id="positive",
        verdict="positive",
    )

    best = best_record([invalidated, positive], "eval_accuracy")
    assert best is not None and best.run_id == "positive"


@pytest.mark.unit
def test_legacy_free_text_verdict_is_ineligible_until_migrated():
    legacy = RunRecord.from_json(
        {
            "run_id": "legacy",
            "harness": "eval",
            "metrics": {"eval_accuracy": 1.0},
            "notes": "experiment=old; verdict=positive",
        }
    )
    ordinary = RunRecord.from_json(
        {"run_id": "ordinary", "harness": "eval", "metrics": {"eval_accuracy": 0.8}}
    )
    null = RunRecord.from_json(
        {
            "run_id": "null",
            "harness": "eval",
            "metrics": {"eval_accuracy": 1.0},
            "verdict": "null",
        }
    )

    assert not legacy.eligible_for_best
    assert not null.eligible_for_best
    assert ordinary.eligible_for_best
    assert best_record([legacy, null, ordinary], "eval_accuracy") == ordinary


@pytest.mark.unit
def test_persisted_verdict_eligibility_invariant_fails_closed():
    base = {"run_id": "bad", "harness": "eval", "metrics": {"eval_accuracy": 1.0}}

    with pytest.raises(ValueError, match="cannot be eligible"):
        RunRecord.from_json(
            {**base, "verdict": "invalidated", "eligible_for_best": True}
        )
    with pytest.raises(TypeError, match="must be a bool"):
        RunRecord.from_json(
            {**base, "verdict": "positive", "eligible_for_best": "true"}
        )
    with pytest.raises(ValueError, match="verdict must be"):
        RunRecord.from_json(
            {**base, "verdict": "unknown", "eligible_for_best": False}
        )


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


@pytest.mark.unit
def test_summarize_revalidates_mutated_records():
    record = make_record("eval", {"val_bpb": 1.0}, run_id="mutated")
    record.metrics["val_bpb"] = float("nan")

    with pytest.raises(ValueError, match="not finite"):
        summarize([record])
