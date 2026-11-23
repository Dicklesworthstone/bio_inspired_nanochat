import math
from collections import Counter

import numpy as np
import pytest

from scripts.e2e.episodic_write_policy import (
    ARMS, BLOCK, CAPACITY, MASS, Memory, complete_documents, score_document, write_gain,
)


def test_gain_is_the_expected_change_of_a_normalized_distribution():
    p = (0.1, 0.3, 0.6)
    mem = Memory()
    for target in (0, 1, 1):
        mem.write((0, 5, target))
    seen = (2, 1, 2)
    q = [(n + MASS * prob) / (sum(seen) + MASS) for n, prob in zip(seen, p, strict=True)]
    before = [mem.predict((0, 5, j), v) for j, v in enumerate(p)]
    mem.write((0, 5, 0))
    after = [mem.predict((0, 5, j), v) for j, v in enumerate(p)]
    assert sum(before) == pytest.approx(1) and sum(after) == pytest.approx(1)
    exact = sum(w * math.log2(b / a) for w, a, b in zip(q, before, after, strict=True))
    assert write_gain(p[0], seen[0], sum(seen), 1, 3) == pytest.approx(sum(seen) * exact)


@pytest.mark.parametrize("order2", [False, True])
def test_eviction_and_context_read_match_independent_fifo(order2):
    events = [(i % 3, i % 2, (i * 7) % 5) for i in range(CAPACITY * 2 + 1)]
    mem = Memory(order2)
    for e in events:
        mem.write(e)
    assert list(mem.slots) == events[-CAPACITY:]
    pairs = Counter((x, y) for _, x, y in events[-CAPACITY:])
    triples = Counter(events[-CAPACITY:])
    for x in range(2):
        observed = []
        for y in range(5):
            p = (pairs[x, y] + MASS / 5) / (sum(v for (k, _), v in pairs.items() if k == x) + MASS)
            if order2:
                n = sum(v for (z, k, _), v in triples.items() if (z, k) == (1, x))
                p = (triples[1, x, y] + MASS * p) / (n + MASS)
            observed.append(mem.predict((1, x, y), 0.2))
            assert observed[-1] == pytest.approx(p)
        assert sum(observed) == pytest.approx(1)


def test_first_block_has_no_future_writes_and_next_target_uses_only_committed_past():
    events = [(1, 1, 2)] * BLOCK
    bits, writes = score_document(events, [0.1] * BLOCK, [0.2] * BLOCK, 4)
    assert all(s == pytest.approx(-BLOCK * math.log2(0.1)) for s in bits.values())
    assert all(n == (BLOCK if arm == "dense" else 4) for arm, n in writes.items())
    for y in (2, 3):
        after, _ = score_document([*events, (1, 1, y)], [0.1] * (BLOCK + 1), [0.2] * (BLOCK + 1), 4)
        p = ((4 if y == 2 else 0) + MASS * 0.1) / (4 + MASS)
        assert after["gain"] - bits["gain"] == pytest.approx(-math.log2(p))
    assert set(bits) == set(ARMS)


def test_missing_stream_transition_and_partial_documents_are_excluded():
    # First and last documents are truncated. A dropped edge splits the middle document.
    pairs = [(1, 9), (9, 2), (2, 3), (3, 9), (9, 2), (4, 9), (9, 1), (1, 9), (9, 5)]
    raw = np.array([10 * x + y for x, y in pairs])
    assert list(complete_documents(raw, 10, 9)) == [(1, 3), (6, 7)]
