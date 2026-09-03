"""Selective decoding is reachable from Engine.generate (bead wmel.1).

README §Calibrated uncertainty promised abstention from predictive uncertainty; the policy object
(``adaptive_compute.UncertaintyDecodingConfig``) existed but no serving path could reach it. Now
``Engine.generate(selective=...)`` measures the predictive entropy of every step's logits and, when it
exceeds the configured threshold, ends the row with ``<|assistant_end|>`` instead of emitting the
token, reporting the event in that step's metrics. Locked here:

* selective off (``None``) and a never-triggering threshold give a byte-identical token stream;
* a threshold of zero makes the first step abstain: the row ends, the mask marks the token as
  forced, and the metrics carry the event with the measured entropy;
* a bad ``selective`` type is refused.

Calibration quality is not claimed; the committed calibration numbers are toy-scale.

Run:  pytest tests/test_selective_decoding.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bio_testkit import make_tiny_synaptic  # noqa: E402

from bio_inspired_nanochat.adaptive_compute import UncertaintyDecodingConfig  # noqa: E402
from bio_inspired_nanochat.engine import Engine, predictive_entropy_nats  # noqa: E402

pytestmark = pytest.mark.unit


class _Tokenizer:
    _special = {
        "<|python_start|>": 90,
        "<|python_end|>": 91,
        "<|output_start|>": 92,
        "<|output_end|>": 93,
        "<|assistant_end|>": 94,
    }

    def encode_special(self, token):
        return self._special[token]

    def get_bos_token_id(self):
        return 96

    def decode(self, _tokens):
        return ""

    def encode(self, _text):
        return []


ASSISTANT_END = 94
PROMPT = [1, 2, 3, 4]


def _engine():
    return Engine(make_tiny_synaptic(seed=0), _Tokenizer())


def _stream(engine, **kw):
    return [tuple(step) for step in engine.generate(PROMPT, max_tokens=6, temperature=1.0, seed=7, **kw)]


def test_entropy_helper_is_shannon_entropy_in_nats():
    logits = torch.zeros(1, 8)  # uniform over 8 -> ln 8
    assert float(predictive_entropy_nats(logits)[0]) == pytest.approx(torch.log(torch.tensor(8.0)).item(), abs=1e-6)
    peaked = torch.tensor([[50.0, 0.0, 0.0]])
    assert float(predictive_entropy_nats(peaked)[0]) < 1e-6


def test_selective_off_or_never_triggering_is_byte_identical():
    baseline = _stream(_engine())
    never = _stream(_engine(), selective={"max_predictive_entropy_nats": 1e9})
    assert never == baseline
    assert baseline and all(mask == [1] for _, mask in baseline)


def test_zero_threshold_abstains_on_the_first_step_and_reports_it():
    engine = _engine()
    steps = list(engine.generate(PROMPT, max_tokens=6, seed=7, yield_metrics=True,
                                 selective=UncertaintyDecodingConfig(enabled=True, max_predictive_entropy_nats=0.0)))
    assert len(steps) == 1, "the row must end on the abstention"
    token_column, token_masks, metrics = steps[0]
    assert token_column == [ASSISTANT_END] and token_masks == [0]
    sel = metrics["selective"]
    assert sel["threshold_nats"] == 0.0 and len(sel["entropy_nats"]) == 1 and sel["entropy_nats"][0] > 0.0
    assert sel["events"] == [{"row": 0, "action": "abstain", "entropy_nats": sel["entropy_nats"][0], "threshold_nats": 0.0}]


def test_clarify_action_and_disabled_config_and_bad_type():
    engine = _engine()
    steps = list(engine.generate(PROMPT, max_tokens=3, seed=7, yield_metrics=True,
                                 selective={"max_predictive_entropy_nats": 0.0, "terminal_action": "clarify"}))
    assert steps[0][2]["selective"]["events"][0]["action"] == "clarify"
    # An explicitly disabled config is the same as None.
    off = _stream(_engine(), selective=UncertaintyDecodingConfig(enabled=False))
    assert off == _stream(_engine())
    with pytest.raises(TypeError, match="selective must be"):
        list(_engine().generate(PROMPT, max_tokens=1, selective=3.5))
