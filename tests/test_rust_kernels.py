import torch
import numpy as np
import pytest
from bio_inspired_nanochat.engine import KVCache
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn

# Try to import rustbpe_native, skip if not available
try:
    from rustbpe import rustbpe as rustbpe_native
    rustbpe = rustbpe_native # For backwards compatibility with test calls
except ImportError:
    rustbpe = None

def softplus(x):
    return torch.log1p(torch.exp(x))

def sigmoid(x):
    return torch.sigmoid(x)

def presyn_step_python_ref(q, k, logits, state, cfg):
    B, H, T, D = q.shape
    
    # Unpack state
    c = state["C"].clone()
    buf = state["BUF"].clone()
    rrp = state["RRP"].clone()
    res = state["RES"].clone()
    pr = state["PR"].clone()
    cl = state["CL"].clone()
    e_st = state["E"].clone()
    
    # Constants
    rho_c = np.exp(-1.0 / cfg.tau_c)
    rho_b = np.exp(-1.0 / cfg.tau_buf)
    rho_p = np.exp(-1.0 / cfg.tau_prime)
    rho_r = np.exp(-1.0 / cfg.tau_rrp)
    rho_e = np.exp(-1.0 / cfg.tau_energy)
    sqrt_d = np.sqrt(D)
    
    syn_logit = torch.zeros_like(logits)
    
    for t in range(T):
        # 1. Compute Influx
        # logits slice (B, H, t, 0:t+1)
        log_t = logits[:, :, t, :t+1]
        clamped = log_t.clamp(-20.0, 20.0)
        drive = softplus(clamped)
        sum_drive = drive.sum(dim=-1) # (B, H)
        influx = sum_drive / (t + 1)
        
        # 2. Update State
        c_prev = c[:, :, t]
        buf_prev = buf[:, :, t]
        
        c_next = rho_c * c_prev + cfg.alpha_ca * influx - cfg.alpha_buf_on * c_prev * (1.0 - buf_prev) + cfg.alpha_buf_off * buf_prev
        buf_next = rho_b * buf_prev + cfg.alpha_buf_on * c_prev * (1.0 - buf_prev) - cfg.alpha_buf_off * buf_prev
        
        c_next = c_next.clamp(min=0.0)
        buf_next = buf_next.clamp(0.0, 1.0)
        
        # Update c and buf for this timestep
        
        pr_val = pr[:, :, t]
        rrp_val = rrp[:, :, t]
        res_val = res[:, :, t]
        e_val = e_st[:, :, t]
        
        pr_mid = (rho_p * pr_val + cfg.alpha_prime * (1.0 - pr_val)).clamp(0.0, 1.0)
        rrp_refill = (rho_r * rrp_val + cfg.alpha_refill * res_val).clamp(0.0, 1.0)
        res_mid = (res_val - cfg.alpha_refill * res_val).clamp(0.0, 1.0)
        e_mid = (rho_e * e_val + cfg.energy_in).clamp(0.0, 1.6)
        
        # 3. Compute Release
        q_t = q[:, :, t, :] # (B, H, D)
        
        c_val = c_next # Use updated C
        fast = c_val / (c_val + cfg.syt_fast_kd)
        slow = c_val / (c_val + cfg.syt_slow_kd)
        syt = 0.7 * fast + 0.3 * slow
        
        cl_val = cl[:, :, t]
        fuse_logit_base = 3.0 * syt + 2.0 * pr_mid - 2.0 * (cl_val + cfg.complexin_bias)
        fuse_base = sigmoid(fuse_logit_base)
        
        # Loop over j
        k_j = k[:, :, :t+1, :] # (B, H, t+1, D)
        # Dot product
        # (B, H, D) * (B, H, t+1, D) -> (B, H, t+1)
        dot = torch.einsum("bhd,bhjd->bhj", q_t, k_j)
        d_bilin = sigmoid(dot / sqrt_d)
        
        fuse_p = fuse_base.unsqueeze(-1) * d_bilin # (B, H, t+1)
        avail = rrp_refill.unsqueeze(-1)
        
        rr = (fuse_p * avail).clamp(0.0, 1.0)
        row_sum = rr.sum(dim=-1) # (B, H)
        
        scale = torch.ones_like(row_sum)
        mask_scale = row_sum > cfg.epsilon
        scale[mask_scale] = (rrp_refill[mask_scale] / row_sum[mask_scale]).clamp(max=1.0)
        
        rel = rr * scale.unsqueeze(-1) # (B, H, t+1)
        used = rel.sum(dim=-1)
        
        # 4. Update Final State
        rrp_n = (rrp_refill - used).clamp(0.0, 1.0)
        res_n = (res_mid + used).clamp(0.0, 1.0)
        pr_n = (pr_mid - cfg.alpha_unprime * used).clamp(0.0, 1.0)
        e_n = (e_mid - cfg.energy_cost_rel * used - cfg.energy_cost_pump * (1.0 - res_n)).clamp(0.0, 1.6)
        
        qamp = sigmoid(cfg.q_beta * (e_n - 0.5)) * cfg.qmax
        
        # 5. Syn Logit
        # dist
        j_indices = torch.arange(t + 1, device=q.device).float()
        t_float = float(t)
        dist = (t_float - j_indices).abs() / max(1.0, float(T))
        dist = dist.view(1, 1, -1)
        
        val = (rel * qamp.unsqueeze(-1)).clamp(min=cfg.epsilon).log() - cfg.barrier_strength * dist
        
        syn_logit[:, :, t, :t+1] = val
        syn_logit[:, :, t, t+1:] = np.log(cfg.epsilon)
        
        # Store state back
        c[:, :, t] = c_next
        buf[:, :, t] = buf_next
        rrp[:, :, t] = rrp_n
        res[:, :, t] = res_n
        pr[:, :, t] = pr_n
        e_st[:, :, t] = e_n
        
    new_state = {
        "C": c, "BUF": buf, "RRP": rrp, "RES": res, "PR": pr, "CL": cl, "E": e_st
    }
    return syn_logit, new_state

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_presyn_step_cpu_parity():
    assert rustbpe is not None
    B, H, T, D = 2, 4, 32, 16
    # Use config compatible with Rust implementation
    cfg = SynapticConfig()
    
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    logits = torch.randn(B, H, T, T)
    mask = torch.tril(torch.ones(T, T)).bool()
    logits.masked_fill_(~mask.view(1, 1, T, T), -20.0) # Use -20 instead of -inf for softplus stability in test
    
    # Build initial state tensors (random for robustness)
    state = {
        "C": torch.rand(B, H, T),
        "BUF": torch.rand(B, H, T),
        "RRP": torch.rand(B, H, T),
        "RES": torch.rand(B, H, T),
        "PR": torch.rand(B, H, T),
        "CL": torch.rand(B, H, T),
        "E": torch.rand(B, H, T),
    }
    
    # Run Rust version
    q_np = q.numpy()
    k_np = k.numpy()
    logits_np = logits.numpy()
    state_np = {k: v.numpy() for k, v in state.items()}
    
    syn_logit_rust, state_new_rust = rustbpe.presyn_step_cpu(q_np, k_np, logits_np, state_np, cfg)
    
    # Run Python Reference
    syn_logit_py, state_new_py = presyn_step_python_ref(q, k, logits, state, cfg)
    
    print("Comparing syn_logit...")
    mask_expanded = mask.view(1, 1, T, T).expand(B, H, T, T)
    diff = torch.abs(torch.from_numpy(syn_logit_rust) - syn_logit_py)
    diff_masked = diff[mask_expanded]
    print(f"Max diff: {diff_masked.max().item()}")
    
    assert torch.allclose(torch.from_numpy(syn_logit_rust)[mask_expanded], syn_logit_py[mask_expanded], atol=1e-4, rtol=1e-4)
    
    print("Comparing state...")
    for k in state:
        if k == "CL":
            continue  # CL is constant
        diff = torch.abs(torch.from_numpy(state_new_rust[k]) - state_new_py[k])
        print(f"State {k} max diff: {diff.max().item()}")
        assert torch.allclose(torch.from_numpy(state_new_rust[k]), state_new_py[k], atol=1e-4, rtol=1e-4)

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_moe_stats_cpu_parity():
    assert rustbpe is not None
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
    
    counts_rust, probs_rust = rustbpe.accumulate_router_stats_cpu(idx_np, gates_np, E)
    
    print("Comparing MoE stats...")
    print(f"Counts max diff: {np.abs(counts_rust - me.numpy()).max()}")
    print(f"Probs max diff: {np.abs(probs_rust - pe.numpy()).max()}")
    
    assert np.allclose(counts_rust, me.numpy(), atol=1e-5)
    assert np.allclose(probs_rust, pe.numpy(), atol=1e-4)

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_metabolism_cpu_parity():
    assert rustbpe is not None
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
    f_rust, e_rust = rustbpe.update_metabolism_cpu(
        fatigue.numpy(), energy.numpy(), alpha_fatigue.numpy(), alpha_energy.numpy(), util.numpy()
    )
    
    print("Comparing Metabolism...")
    print(f"Fatigue max diff: {np.abs(f_rust - f_py.numpy()).max()}")
    print(f"Energy max diff: {np.abs(e_rust - e_py.numpy()).max()}")
    
    assert np.allclose(f_rust, f_py.numpy(), atol=1e-5)
    assert np.allclose(e_rust, e_py.numpy(), atol=1e-5)

if __name__ == "__main__":
    if rustbpe:
        test_presyn_step_cpu_parity()
        test_moe_stats_cpu_parity()
        test_metabolism_cpu_parity()
        print("All tests passed!")
    else:
        print("rustbpe not installed, skipping tests")


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

    pre_train = SynapticPresyn(d_head=16, cfg=cfg)
    pre_eval = SynapticPresyn(d_head=16, cfg=cfg)

    B, H, Tk, Tq, K = 1, 2, 6, 3, 4
    drive = torch.randn(B, H, Tq, K)
    idx = torch.randint(0, Tk, (B, H, Tq, K))

    state_train = build_presyn_state(B, Tk, H, drive.device, drive.dtype, cfg)
    state_eval = build_presyn_state(B, Tk, H, drive.device, drive.dtype, cfg)

    e_train = pre_train.release_canonical(state_train, drive, idx, train=True)
    e_eval = pre_eval.release_canonical(state_eval, drive, idx, train=False)

    torch.testing.assert_close(e_train, e_eval, rtol=1e-6, atol=1e-6)
    for key in ["C", "BUF", "RRP", "RES", "PR", "CL", "E", "AMP"]:
        torch.testing.assert_close(state_train[key], state_eval[key], rtol=1e-6, atol=1e-6)
    for d_train, d_eval in zip(state_train["DELAY"], state_eval["DELAY"], strict=True):
        torch.testing.assert_close(d_train, d_eval, rtol=1e-6, atol=1e-6)


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
