import torch

from bio_inspired_nanochat.synaptic import SynapticLinear
from scripts.e2e.trained_hebbian_binding import CHUNK, DOCUMENTS, SEEDS, TraceRead, decision, model_for, modes


@torch.no_grad()
def test_both_trace_factors_use_the_same_within_lane_permutation():
    model = model_for(31, 67, torch.device("cpu")).eval()
    sites = [site for site in model.modules() if isinstance(site, SynapticLinear)]
    originals = [site._update_hebb_traces for site in sites]
    rng = torch.Generator().manual_seed(12)
    indices = torch.stack([torch.randperm(5, generator=rng) for _ in range(2)])
    assert not torch.equal(indices[0], indices[1])
    with TraceRead(model, 12) as intervention:
        for site in sites:
            x = torch.randn(2, 5, site.w_slow.size(0))
            y = torch.randn(2, 5, site.w_slow.size(1))
            permuted = y.gather(1, indices.unsqueeze(-1).expand_as(y))
            torch.testing.assert_close(y.sort(dim=1).values, permuted.sort(dim=1).values, atol=0, rtol=0)
            moment = x.flatten(0, 1).T @ permuted.flatten(0, 1) / 10
            site._update_hebb_traces(x, y, None)
            assert site.u_buf is not None and site.v_buf is not None
            assert site.proj_in is not None and site.proj_out is not None
            torch.testing.assert_close(site.u_buf, .05 * (moment @ site.proj_out))
            torch.testing.assert_close(site.v_buf, .05 * (site.proj_in.T @ moment))
        assert intervention.calls == len(sites)
    assert [site._update_hebb_traces for site in sites] == originals


def test_a_permuted_write_is_visible_only_to_a_later_block():
    model = model_for(31, 67, torch.device("cpu")).eval()
    rng = torch.Generator().manual_seed(13)
    inputs = torch.randint(0, 67, (1, CHUNK + 4), generator=rng)
    targets = torch.randint(0, 67, inputs.shape, generator=rng)
    result = modes(model, inputs, targets, 337)
    identity, frozen, shuffle, _, _, _, writes = result
    assert writes == 2 * sum(isinstance(site, SynapticLinear) for site in model.modules())
    torch.testing.assert_close(shuffle[:, :, :CHUNK], identity.expand(4, -1, -1)[:, :, :CHUNK], atol=0, rtol=0)
    assert float((identity[:, CHUNK:] - frozen[:, CHUNK:]).abs().max()) > 1e-5
    assert float((identity[:, CHUNK:] - shuffle[:, :, CHUNK:]).abs().max()) > 1e-5


def test_decision_requires_improvement_over_initialization_and_matched_training():
    rows = []
    for seed in SEEDS:
        for index in range(DOCUMENTS):
            for arm in ("initial", "paired", "shuffle"):
                rows.append({"seed": seed, "doc_sha256": str(index), "arm": arm, "bytes": 100,
                             "identity_bits": 98 if arm == "paired" else 100,
                             "frozen_bits": 100, "shuffle_bits": 100})
    assert decision(rows)["advance"]
    for row in rows:
        if row["seed"] == SEEDS[-1] and row["arm"] == "shuffle":
            row["identity_bits"] = 97
    result = decision(rows)
    assert result["contrasts"]["over_shuffle"]["pass"]
    assert not result["contrasts"]["paired_minus_shuffle_training"]["pass"]
    assert not result["advance"]
