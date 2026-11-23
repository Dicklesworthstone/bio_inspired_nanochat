import copy

import pytest
import torch
from torch.nn import functional as F

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.rollout_credit import (
    RolloutCredit,
    pack_rollout_tokens,
    permute_advantages,
    rollout_direction,
    synaptic_token_logp,
)
from bio_inspired_nanochat.synaptic import SynapticConfig


def make_model():
    torch.manual_seed(315)
    return GPTSynaptic(GPTSynapticConfig(
        n_layer=1, n_embd=32, n_head=2, n_kv_head=2, vocab_size=31, sequence_len=560,
        syn_cfg=SynapticConfig(enable_hebbian=True, enable_presyn=False, enable_metabolism=False),
    )).train()


def test_direction_uses_only_selected_coactivities():
    rng = torch.Generator().manual_seed(92)
    x, y = torch.randn(6, 4, generator=rng), torch.randn(6, 5, generator=rng)
    pi, po = torch.randn(4, 3, generator=rng), torch.randn(5, 3, generator=rng)
    mask = torch.tensor([True, False, False, True, False, True])
    cross = torch.stack([torch.outer(a, b) for a, b in zip(x[mask], y[mask], strict=True)]).sum(0)
    expected = cross @ po @ pi.T @ cross
    x[~mask] = float("nan")
    y[~mask] = float("inf")
    torch.testing.assert_close(rollout_direction(x, y, mask, pi, po), expected / expected.norm())
    torch.testing.assert_close(rollout_direction(x, y, ~torch.ones_like(mask), pi, po), torch.zeros_like(expected))


def test_policy_log_prob_matches_full_head_reference():
    model = make_model()
    x = torch.randint(0, 31, (3, 11))
    targets = x.clone()
    targets[:, 4] = -1
    logits, _ = model(x, targets, train_mode=True, update_mem=False)
    expected = -F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1, reduction="none")
    torch.testing.assert_close(synaptic_token_logp(model, x, targets, update_mem=False), expected.view_as(targets))


def test_only_post_action_credit_distinguishes_single_token_rollouts():
    tokens = torch.tensor([[1, 4, 11, 8], [1, 4, 11, 17]])
    sampled = torch.tensor([[False, False, False, True]]).repeat(2, 1)
    inputs, targets, observation = pack_rollout_tokens(tokens, sampled, include_last=True)
    assert targets.tolist() == [[-1, -1, 8, -1], [-1, -1, 17, -1]]
    assert observation.tolist() == sampled.tolist()
    _, legacy_labels, legacy_mask = pack_rollout_tokens(tokens, sampled, include_last=False)
    assert legacy_labels.tolist() == [[-1, -1, 8], [-1, -1, 17]]
    assert not bool(legacy_mask.any())
    base = make_model()
    for mask, expected in ((targets >= 0, False), (observation, True)):
        model = copy.deepcopy(base)
        credit = RolloutCredit(model, lr=.03, rollouts_per_rank=2)
        with credit.observe(mask, torch.tensor([.5, -.5])):
            synaptic_token_logp(model, inputs, targets, update_mem=False)
        assert (credit.commit()["max_update_norm"] > 1e-6) == expected


def test_matched_gradients_then_same_write_across_a_remainder_and_permutation():
    base = make_model()
    x = torch.randint(0, 31, (5, 9))
    y = torch.roll(x, -1, dims=1)
    y[::2, -3:] = -1
    y[:, :3] = -1
    advantage = torch.tensor([.8, -.2, -.2, -.2, -.2])
    shuffled = permute_advantages(advantage, 931)
    torch.testing.assert_close(shuffled.sort().values, advantage.sort().values)

    def execute(model, groups, credit):
        opt = torch.optim.SGD(model.parameters(), lr=.001)
        initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        for indexes in groups:
            inputs, targets = x[indexes], y[indexes]
            if credit is None:
                logp = synaptic_token_logp(model, inputs, targets, update_mem=False)
            else:
                with credit.observe(targets >= 0, advantage[indexes]):
                    logp = synaptic_token_logp(model, inputs, targets, update_mem=False)
            (-(logp * advantage[indexes, None]).sum() / (y >= 0).sum()).backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, initial[name], atol=0, rtol=0)
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
        opt.step()
        if credit is not None:
            assert credit.commit()["max_update_norm"] > 0
        return gradients

    models = [copy.deepcopy(base) for _ in range(3)]
    full = [torch.arange(5)]
    remainder = [torch.tensor([3, 0]), torch.tensor([2, 1]), torch.tensor([4])]
    normal = execute(models[0], full, None)
    treated = execute(models[1], full, RolloutCredit(models[1], lr=.03, rollouts_per_rank=5))
    split = execute(models[2], remainder, RolloutCredit(models[2], lr=.03, rollouts_per_rank=5))
    for name in normal:
        torch.testing.assert_close(treated[name], normal[name], atol=0, rtol=0)
        torch.testing.assert_close(split[name], treated[name])
    for a, b in zip(models[1].parameters(), models[2].parameters(), strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("value", [0.0, 0.5])
def test_constant_reward_and_identical_rollout_cancellation(value):
    model = make_model()
    credit = RolloutCredit(model, lr=.03, rollouts_per_rank=2)
    tokens = torch.randint(0, 31, (1, 7)).repeat(2, 1)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    versions = [parameter._version for parameter in model.parameters()]
    with credit.observe(mask, torch.tensor([value, -value])):
        synaptic_token_logp(model, tokens, tokens, update_mem=False)
    norm = credit.commit()["max_update_norm"]
    if value == 0:
        assert norm == 0
        assert versions == [parameter._version for parameter in model.parameters()]
    else:
        # Rowwise CPU GEMMs can differ at roundoff even for identical tokens.
        assert norm <= 1e-8
    with pytest.raises(RuntimeError, match="saw 0"):
        credit.commit()
