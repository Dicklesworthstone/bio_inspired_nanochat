import importlib
from typing import Any, cast

import torch
import numpy as np
import pytest
from bio_inspired_nanochat.engine import KVCache
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn

# The maturin extension is a top-level module. Skip only when it is genuinely unbuilt.
try:
    rustbpe: Any = importlib.import_module("rustbpe")
except ModuleNotFoundError:
    rustbpe = None

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_moe_stats_cpu_parity():
    assert rustbpe is not None
    backend = cast(Any, rustbpe)
    B, T, k = 2, 128, 2
    E = 8
    
    idx = torch.randint(0, E, (B, T, k))
    gates = torch.rand(B, T, k)
    
    # Python reference
    me = torch.zeros(E)
    pe = torch.zeros(E)
    for e in range(E):
        mask = idx == e
        sel = mask.any(dim=-1)
        me[e] = sel.sum()
        pe[e] = gates.masked_select(mask).sum()
        
    # Rust version
    idx_np = idx.numpy().astype("int64")
    gates_np = gates.numpy()
    
    counts_rust, probs_rust = backend.accumulate_router_stats_cpu(idx_np, gates_np, E)

    assert np.allclose(counts_rust, me.numpy(), atol=1e-5)
    assert np.allclose(probs_rust, pe.numpy(), atol=1e-4)


@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
@pytest.mark.parametrize(
    ("idx_shape", "gates_shape"),
    [
        ((2, 3), (2, 3)),
        ((2, 3, 1), (2, 4, 1)),
        ((2, 3, 1), (2, 3, 1, 1)),
    ],
)
def test_moe_stats_cpu_rejects_invalid_gate_geometry(idx_shape, gates_shape):
    assert rustbpe is not None
    backend = cast(Any, rustbpe)
    idx = np.zeros(idx_shape, dtype=np.int64)
    gates = np.zeros(gates_shape, dtype=np.float32)

    expected = "idx must be 3D" if len(idx_shape) != 3 else "gates must have the same shape"
    with pytest.raises(ValueError, match=expected):
        backend.accumulate_router_stats_cpu(idx, gates, 4)


@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_moe_stats_cpu_rejects_zero_experts():
    assert rustbpe is not None
    backend = cast(Any, rustbpe)

    with pytest.raises(ValueError, match="num_experts must be positive"):
        backend.accumulate_router_stats_cpu(
            np.zeros((1, 1, 1), dtype=np.int64),
            np.ones((1, 1, 1), dtype=np.float32),
            0,
        )

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_metabolism_cpu_parity():
    assert rustbpe is not None
    backend = cast(Any, rustbpe)
    E = 8
    fatigue = torch.rand(E)
    energy = torch.rand(E)
    alpha_fatigue = torch.rand(E) * 0.1
    alpha_energy = torch.rand(E) * 0.1
    util = torch.rand(E)
    
    # Python reference
    f_py = fatigue.clone()
    e_py = energy.clone()
    f_py.mul_(1.0 - alpha_fatigue).add_(alpha_fatigue * util)
    e_py.mul_(1.0 - alpha_energy).add_(alpha_energy * (1.0 - util))
    
    # Rust version
    f_rust, e_rust = backend.update_metabolism_cpu(
        fatigue.numpy(), energy.numpy(), alpha_fatigue.numpy(), alpha_energy.numpy(), util.numpy()
    )

    assert np.allclose(f_rust, f_py.numpy(), atol=1e-5)
    assert np.allclose(e_rust, e_py.numpy(), atol=1e-5)


@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
@pytest.mark.parametrize("mismatched_position", range(1, 5))
def test_metabolism_cpu_rejects_mismatched_vector_lengths(mismatched_position):
    assert rustbpe is not None
    backend = cast(Any, rustbpe)
    arrays = [np.zeros(4, dtype=np.float32) for _ in range(5)]
    arrays[mismatched_position] = np.zeros(3, dtype=np.float32)
    names = ["fatigue", "energy", "alpha_fatigue", "alpha_energy", "util"]

    with pytest.raises(ValueError, match=rf"{names[mismatched_position]} length must match"):
        backend.update_metabolism_cpu(*arrays)

def test_stochastic_binomial_counts_matches_moments():
    from bio_inspired_nanochat.synaptic import _sample_binomial_counts

    torch.manual_seed(0)

    N = 50_000
    p = torch.full((N,), 0.3, dtype=torch.float32)
    n = torch.full((N,), 5.0, dtype=torch.float32)

    samples = _sample_binomial_counts(
        p,
        n,
        max_count=8,
        tau=1.0,
        mode="gumbel_sigmoid_ste",
    )

    mean_emp = float(samples.mean())
    var_emp = float(samples.var(unbiased=False))
    mean_true = float(5.0 * 0.3)
    var_true = float(5.0 * 0.3 * (1.0 - 0.3))

    assert abs(mean_emp - mean_true) < 0.03
    assert abs(var_emp - var_true) < 0.05


def test_presyn_release_is_deterministic_when_stochastic_train_frac_is_zero():
    from bio_inspired_nanochat.synaptic import build_presyn_state

    torch.manual_seed(0)

    cfg = SynapticConfig()
    cfg.stochastic_train_frac = 0.0

    pre_a = SynapticPresyn(d_head=16, cfg=cfg)
    pre_b = SynapticPresyn(d_head=16, cfg=cfg)

    B, H, Tk, Tq, K = 1, 2, 6, 3, 4
    drive = torch.randn(B, H, Tq, K)
    idx = torch.randint(0, Tk, (B, H, Tq, K))

    state_a = build_presyn_state(B, Tk, H, drive.device, drive.dtype, cfg)
    state_b = build_presyn_state(B, Tk, H, drive.device, drive.dtype, cfg)

    e_a = pre_a.release_canonical(state_a, drive, idx, train=True)
    e_b = pre_b.release_canonical(state_b, drive, idx, train=True)

    torch.testing.assert_close(e_a, e_b, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(pre_a.ema_e, pre_b.ema_e, rtol=0.0, atol=0.0)
    for key in ["C", "BUF", "RRP", "RES", "PR", "CL", "E", "AMP"]:
        torch.testing.assert_close(state_a[key], state_b[key], rtol=1e-6, atol=1e-6)
    for delay_a, delay_b in zip(state_a["DELAY"], state_b["DELAY"], strict=True):
        torch.testing.assert_close(delay_a, delay_b, rtol=1e-6, atol=1e-6)


def test_gpt_synaptic_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    syn_cfg = SynapticConfig(enable_presyn=False, lambda_loge=0.0, barrier_strength=0.0)
    cfg = GPTSynapticConfig(
        sequence_len=32,
        vocab_size=97,
        n_layer=2,
        n_head=2,
        n_kv_head=1,
        n_embd=16,
        dropout=0.0,
        synapses=True,
        syn_cfg=syn_cfg,
    )
    model = GPTSynaptic(cfg).eval()

    B, T = 1, 12
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    logits_full, _loss = model(idx, kv_cache=None, train_mode=False)

    head_dim = cfg.n_embd // cfg.n_head
    kv_cache = KVCache(
        batch_size=B,
        num_heads=cfg.n_kv_head,
        seq_len=T,
        head_dim=head_dim,
        num_layers=cfg.n_layer,
    )

    step_logits = []
    for t in range(T):
        logits_t, _ = model(idx[:, t : t + 1], kv_cache=kv_cache, train_mode=False)
        step_logits.append(logits_t[:, -1, :])
    logits_kv = torch.stack(step_logits, dim=1)

    torch.testing.assert_close(logits_kv, logits_full, rtol=1e-4, atol=1e-6)


def test_gpt_synaptic_loss_ignores_minus_one_targets():
    torch.manual_seed(0)
    syn_cfg = SynapticConfig(enable_presyn=False, lambda_loge=0.0, barrier_strength=0.0)
    cfg = GPTSynapticConfig(
        sequence_len=32,
        vocab_size=97,
        n_layer=2,
        n_head=2,
        n_kv_head=1,
        n_embd=16,
        dropout=0.0,
        synapses=True,
        syn_cfg=syn_cfg,
    )
    model = GPTSynaptic(cfg).eval()

    B, T = 2, 12
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = idx.clone()
    targets[:, : T // 2] = -1

    _logits, loss = model(idx, targets=targets, kv_cache=None, train_mode=False)
    assert loss is not None
    assert torch.isfinite(loss).all()


def test_gpt_synaptic_presyn_produces_finite_logits_and_loss():
    torch.manual_seed(0)
    syn_cfg = SynapticConfig()
    syn_cfg.stochastic_train_frac = 0.0
    cfg = GPTSynapticConfig(
        sequence_len=32,
        vocab_size=97,
        n_layer=2,
        n_head=2,
        n_kv_head=1,
        n_embd=16,
        dropout=0.0,
        synapses=True,
        syn_cfg=syn_cfg,
    )
    model = GPTSynaptic(cfg).eval()

    B, T = 2, 12
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = idx.clone()
    targets[:, : T // 2] = -1

    logits, loss = model(idx, targets=targets, kv_cache=None, train_mode=False)
    assert torch.isfinite(logits).all()
    assert loss is not None
    assert torch.isfinite(loss).all()


def test_tune_bio_params_top10_specs_match_synaptic_config():
    import scripts.tune_bio_params as tune

    cfg = SynapticConfig()
    for spec in tune.TOP10_PARAM_SPECS:
        assert hasattr(cfg, spec.name), spec.name


def test_tune_bio_params_top10_roundtrip_encode_decode():
    import scripts.tune_bio_params as tune

    cfg = SynapticConfig()
    vec = tune.encode_params(cfg, tune.TOP10_PARAM_SPECS)
    decoded = tune.decode_params(vec, tune.TOP10_PARAM_SPECS)
    for spec in tune.TOP10_PARAM_SPECS:
        torch.testing.assert_close(
            torch.tensor(decoded[spec.name]),
            torch.tensor(getattr(cfg, spec.name)),
            rtol=1e-6,
            atol=1e-9,
        )


def test_tune_bio_params_vector_cli_is_param_space():
    import scripts.tune_bio_params as tune

    cfg = SynapticConfig()
    parts = [str(float(getattr(cfg, spec.name))) for spec in tune.TOP10_PARAM_SPECS]
    vec = tune._parse_vector(",".join(parts), tune.TOP10_PARAM_SPECS)
    decoded = tune.decode_params(vec, tune.TOP10_PARAM_SPECS)
    for spec in tune.TOP10_PARAM_SPECS:
        torch.testing.assert_close(
            torch.tensor(decoded[spec.name]),
            torch.tensor(float(getattr(cfg, spec.name))),
            rtol=1e-6,
            atol=1e-9,
        )


def test_tune_bio_params_generate_batch_targets_are_next_token_copy_half():
    import scripts.tune_bio_params as tune

    batch = 2
    seq_len = 8
    vocab = 23
    x, y = tune.generate_batch(batch, seq_len, vocab, device="cpu")

    assert x.shape == (batch, seq_len)
    assert y.shape == (batch, seq_len)

    half = seq_len // 2
    torch.testing.assert_close(x[:, :half], x[:, half:], rtol=0, atol=0)

    # Next-token targets: only score predictions that generate the second half.
    assert bool((y[:, : half - 1] == -1).all())
    torch.testing.assert_close(y[:, half - 1 : seq_len - 1], x[:, half:seq_len], rtol=0, atol=0)
    assert bool((y[:, -1] == -1).all())
    assert bool((y != -1).any())


def test_tune_bio_params_merge_allgathered_fitness_prefers_finite():
    import scripts.tune_bio_params as tune

    t0 = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float64)
    t1 = torch.tensor([float("nan"), 2.0, float("nan")], dtype=torch.float64)
    merged = tune._merge_allgathered_fitness([t0, t1])
    assert merged == [1.0, 2.0, 3.0]


def test_tune_bio_params_merge_allgathered_fitness_uses_penalty_if_missing():
    import scripts.tune_bio_params as tune

    t0 = torch.tensor([float("nan"), float("nan")], dtype=torch.float64)
    merged = tune._merge_allgathered_fitness([t0])
    assert merged == [tune.PENALTY_LOSS, tune.PENALTY_LOSS]


def test_tune_bio_params_stagnation_improvement_frac():
    import scripts.tune_bio_params as tune

    # Need window+1 points.
    assert tune._stagnation_improvement_frac(best_loss_history=[1.0], window_gens=2) is None

    # 1% improvement over 2 gens.
    frac = tune._stagnation_improvement_frac(best_loss_history=[1.0, 0.995, 0.99], window_gens=2)
    assert frac is not None
    assert abs(frac - 0.01) < 1e-9


def test_tune_bio_params_rosenbrock_cmaes_converges():
    import scripts.tune_bio_params as tune

    xbest, fbest = tune.run_rosenbrock_2d_cmaes(seed=1, iterations=80, popsize=8, sigma0=0.5)
    err = float(np.linalg.norm(np.asarray(xbest) - np.array([1.0, 1.0])))
    assert err < 1e-2
    assert fbest < 1e-6


def test_tune_bio_params_rosenbrock_rejects_seed_zero():
    import scripts.tune_bio_params as tune

    with pytest.raises(ValueError, match="non-zero seed"):
        tune.run_rosenbrock_2d_cmaes(seed=0, iterations=10, popsize=4, sigma0=0.5)


def test_core_eval_mc_start_indices_do_not_drop_shared_answer_prefix():
    from bio_inspired_nanochat import core_eval

    class DummyTokenizer:
        def __init__(self):
            self._vocab: dict[str, int] = {}
            self._next_id = 1

        def get_bos_token_id(self) -> int:
            return 0

        def __call__(self, prompts, *, prepend: int):
            if isinstance(prompts, str):
                prompts = [prompts]
            out = []
            for p in prompts:
                toks = [prepend]
                for w in p.strip().split():
                    if w not in self._vocab:
                        self._vocab[w] = self._next_id
                        self._next_id += 1
                    toks.append(self._vocab[w])
                out.append(toks)
            return out

    tok = DummyTokenizer()
    item = {"query": "Q", "choices": ["the cat", "the dog"], "gold": 0}
    prompt_without, prompts = core_eval.render_prompts_mc(item, " ", fewshot_examples=[])
    tokens, start_idxs, _end_idxs = core_eval.batch_sequences_mc(tok, prompt_without, prompts)

    # If we incorrectly used the common prefix across *full prompts*, we'd exclude the shared
    # "the" token from scoring (start index would be after it). The corrected logic aligns a
    # shared prompt-without-choice against each prompt, so scoring includes the shared prefix.
    assert len(tokens) == 2
    assert start_idxs == [2, 2]
