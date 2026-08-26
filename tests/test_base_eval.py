import csv
import json
import os
import pytest
import torch
import yaml

from scripts.base_eval import _validate_eval_bundle_dir, evaluate_model, ModelWrapper


class _MockTokenizer:
    def get_bos_token_id(self):
        return 1

    def __call__(self, texts, prepend=None):
        return [[1, 2, 3, 4] for _ in texts]


class _MockModel(torch.nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = 512

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        logits = torch.zeros((bsz, seq_len, self.vocab_size), dtype=torch.float32)
        # Give gold choice a high score
        logits[:, :, 2] = 5.0
        return logits


def _setup_mock_eval_bundle(bundle_dir: str):
    os.makedirs(bundle_dir, exist_ok=True)
    eval_data_dir = os.path.join(bundle_dir, "eval_data")
    os.makedirs(eval_data_dir, exist_ok=True)

    # 1. Create eval_data task file
    task_file = os.path.join(eval_data_dir, "mock_mc.jsonl")
    with open(task_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query": "Question 1?", "choices": ["A", "B"], "gold": 0}) + "\n")
        f.write(json.dumps({"query": "Question 2?", "choices": ["A", "B"], "gold": 1}) + "\n")

    # 2. Create core.yaml
    core_yaml_path = os.path.join(bundle_dir, "core.yaml")
    core_config = {
        "icl_tasks": [
            {
                "label": "mock_mc",
                "icl_task_type": "multiple_choice",
                "dataset_uri": "mock_mc.jsonl",
                "num_fewshot": [0],
                "continuation_delimiter": " ",
            }
        ]
    }
    with open(core_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(core_config, f)

    # 3. Create eval_meta_data.csv
    meta_csv_path = os.path.join(bundle_dir, "eval_meta_data.csv")
    with open(meta_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Eval Task", "Random baseline"])
        writer.writeheader()
        writer.writerow({"Eval Task": "mock_mc", "Random baseline": "50.0"})


def test_validate_eval_bundle_dir(tmp_path):
    bundle_dir = str(tmp_path / "bundle")
    _setup_mock_eval_bundle(bundle_dir)
    # Should not raise
    _validate_eval_bundle_dir(bundle_dir)

    # Missing file should raise FileNotFoundError
    os.remove(os.path.join(bundle_dir, "core.yaml"))
    with pytest.raises(FileNotFoundError):
        _validate_eval_bundle_dir(bundle_dir)


def test_evaluate_model_with_bundle(tmp_path):
    bundle_dir = str(tmp_path / "bundle")
    _setup_mock_eval_bundle(bundle_dir)

    model = _MockModel()
    tokenizer = _MockTokenizer()
    device = torch.device("cpu")

    out = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_per_task=2,
        eval_bundle_dir=bundle_dir,
    )
    assert "results" in out
    assert "centered_results" in out
    assert "core_metric" in out
    assert "mock_mc" in out["results"]
    assert isinstance(out["core_metric"], float)


def test_model_wrapper():
    class TupleReturningModel(torch.nn.Module):
        def forward(self, x):
            return (torch.randn(2, 4, 10), None)

    wrapped = ModelWrapper(TupleReturningModel(), max_seq_len=64)
    res = wrapped(torch.zeros(2, 4, dtype=torch.long))
    assert hasattr(res, "logits")
    assert res.logits.shape == (2, 4, 10)
