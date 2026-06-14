"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import torch
import os
import tempfile
import pytest

from bio_inspired_nanochat.engine import KVCache

def test_use_calculator_is_ast_safe_and_supports_count() -> None:
    from bio_inspired_nanochat.engine import use_calculator

    assert use_calculator("1 + 2*3") == 7
    assert use_calculator("1,000 + 2") == 1002
    assert use_calculator("\"hello\".count(\"l\")") == 2
    assert use_calculator("2**8") is None
    assert use_calculator("\"x\".__class__") is None
    assert use_calculator("__import__('os').system('echo hi')") is None

def test_kv_cache_resize():
    """
    The KV cache was not resized correctly, more information here:
    https://github.com/karpathy/nanochat/pull/186
    This test reproduces the issue and will be merged alongside the fix.
    """

    batch_size = 2
    num_heads = 3
    seq_len = 4
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers
    )

    # Insert a single token with a distinct fill value to all layers
    def insert_token(token_idx):
        for layer_idx in range(num_layers):
            k = torch.full((batch_size, num_heads, 1, head_dim), fill_value=float(token_idx), dtype=torch.float32)
            v = torch.full((batch_size, num_heads, 1, head_dim), fill_value=float(token_idx * 100), dtype=torch.float32)
            kv_cache.insert_kv(layer_idx, k, v)

    # Insert 4 tokens (fills the initial seq_len=4)
    for i in range(4):
        insert_token(i)

    # Record the original state of the cache
    cache_tensor = kv_cache.kv_cache
    assert cache_tensor is not None, "KV cache tensor should be initialized"
    original_cache = cache_tensor.clone()
    original_seq_len = original_cache.shape[4]

    # Insert the 5th token, which will trigger a resize
    insert_token(4)
    # Verify that the cache actually resized
    resized_cache = kv_cache.kv_cache
    assert resized_cache is not None, "KV cache tensor should exist after resize"
    new_seq_len = resized_cache.shape[4]
    assert new_seq_len > original_seq_len, f"Cache did not resize: original seq_len={original_seq_len}, new seq_len={new_seq_len}"

    # Verify that the original 4 tokens are still intact after resize
    for layer_idx in range(num_layers):
        for token_idx in range(4):
            # Check that resized cache matches expected values
            expected_k = float(token_idx)
            expected_v = float(token_idx * 100)
            actual_k = resized_cache[layer_idx, 0, :, :, token_idx, :]
            actual_v = resized_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == expected_k).all(), f"Layer {layer_idx}, token {token_idx}: key corrupted, expected {expected_k}"
            assert (actual_v == expected_v).all(), f"Layer {layer_idx}, token {token_idx}: value corrupted, expected {expected_v}"
            # And that the original cache matches resized cache
            original_k = original_cache[layer_idx, 0, :, :, token_idx, :]
            original_v = original_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == original_k).all(), f"Layer {layer_idx}, token {token_idx}: key doesn't match original"
            assert (actual_v == original_v).all(), f"Layer {layer_idx}, token {token_idx}: value doesn't match original"


def test_base_eval_bundle_dir_validation_smoke():
    from scripts import base_eval

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = os.path.join(tmpdir, "eval_bundle")
        os.makedirs(os.path.join(bundle_dir, "eval_data"), exist_ok=True)
        with open(os.path.join(bundle_dir, "core.yaml"), "w", encoding="utf-8") as f:
            f.write("icl_tasks: []\n")
        with open(os.path.join(bundle_dir, "eval_meta_data.csv"), "w", encoding="utf-8") as f:
            f.write("Eval Task,Random baseline\n")

        base_eval._validate_eval_bundle_dir(bundle_dir)


def test_base_eval_bundle_dir_validation_errors_on_missing_files():
    from scripts import base_eval

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = os.path.join(tmpdir, "eval_bundle")
        os.makedirs(bundle_dir, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="missing:"):
            base_eval._validate_eval_bundle_dir(bundle_dir)


def test_core_eval_forward_model_handles_synaptic_tuple_outputs():
    from bio_inspired_nanochat.core_eval import forward_model
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig

    cfg = GPTSynapticConfig(
        sequence_len=32,
        vocab_size=97,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
    )
    model = GPTSynaptic(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16), dtype=torch.long)
    losses, preds = forward_model(model, input_ids)
    assert losses.shape == (2, 16)
    assert preds.shape == (2, 16)
    assert torch.isnan(losses[:, -1]).all()
    assert torch.isfinite(losses[:, :-1]).all()
    assert torch.isfinite(preds.float()).all()


def test_ultrametric_attention_forward_backward_finite():
    from bio_inspired_nanochat.gpt import GPT, GPTConfig

    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=97,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        attention_type="ultrametric",
        ultrametric_k=4,
        ultrametric_p=2,
        ultrametric_alpha=2.0,
        ultrametric_lcp_beta=32.0,
        ultrametric_query_chunk_size=16,
    )
    model = GPT(cfg)
    model.train()

    B, T = 2, 32
    idx = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)

    loss = model(idx, targets)
    assert torch.isfinite(loss), "loss should be finite"

    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "all grads should be finite"


def test_ultrametric_attention_kv_cache_decode_is_finite():
    from bio_inspired_nanochat.gpt import GPT, GPTConfig

    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=101,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        attention_type="ultrametric",
        ultrametric_k=4,
        ultrametric_query_chunk_size=16,
    )
    model = GPT(cfg)
    model.eval()

    head_dim = cfg.n_embd // cfg.n_head
    B = 2
    kv_cache = KVCache(
        batch_size=B,
        num_heads=cfg.n_kv_head,
        seq_len=cfg.sequence_len,
        head_dim=head_dim,
        num_layers=cfg.n_layer,
    )

    # Prefill with a short prompt.
    prompt = torch.randint(0, cfg.vocab_size, (B, 8), dtype=torch.long)
    logits = model.forward(prompt, kv_cache=kv_cache)
    assert logits.shape == (B, 8, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert kv_cache.get_pos() == 8

    # Decode one more token.
    next_tok = torch.randint(0, cfg.vocab_size, (B, 1), dtype=torch.long)
    logits2 = model.forward(next_tok, kv_cache=kv_cache)
    assert logits2.shape == (B, 1, cfg.vocab_size)
    assert torch.isfinite(logits2).all()
    assert kv_cache.get_pos() == 9


def test_synaptic_moe_xi_swap_swaps_metabolism_buffers():
    from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE

    cfg = SynapticConfig(
        enable_metabolism=True,
        enable_hebbian=False,
        native_genetics=False,
    )
    moe = SynapticMoE(
        n_embd=4,
        num_experts=2,
        top_k=1,
        hidden_mult=1,
        cfg=cfg,
        dropout=0.0,
    )

    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 1.0
        moe.router.weight[1, 0] = -1.0

        # Make metabolism visibly different between experts.
        moe.Xi.zero_()
        moe.Xi[0, 0] = 10.0
        moe.Xi[1, 0] = -10.0
        moe.Xi[0, 1] = 10.0
        moe.Xi[1, 1] = -10.0

    x = torch.zeros((1, 8, 4), dtype=torch.float32)
    x[0, ::2, 0] = 1.0
    x[0, 1::2, 0] = -1.0

    def run_once() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            moe.fatigue.zero_()
            moe.energy.fill_(1.0)
        moe.eval()
        moe(x)
        return moe.fatigue.detach().clone(), moe.energy.detach().clone()

    fat_a, en_a = run_once()

    with torch.no_grad():
        xi = moe.Xi.detach().clone()
        moe.Xi[0].copy_(xi[1])
        moe.Xi[1].copy_(xi[0])

    fat_b, en_b = run_once()

    assert torch.allclose(fat_a.flip(0), fat_b, atol=1e-6, rtol=0.0)
    assert torch.allclose(en_a.flip(0), en_b, atol=1e-6, rtol=0.0)


def test_kv_cache_prefill_presyn_state_list_expands_batch():
    from bio_inspired_nanochat.engine import KVCache
    from bio_inspired_nanochat.synaptic import build_presyn_state, SynapticConfig

    cfg = SynapticConfig()
    B_src = 1
    B_tgt = 3
    H = 2
    T = 4
    n_layers = 2

    def make_state(batch: int):
        return build_presyn_state(batch, T, H, device="cpu", dtype=torch.float32, cfg=cfg)

    src_cache = KVCache(batch_size=B_src, num_heads=H, seq_len=T, head_dim=8, num_layers=n_layers)
    src_cache.kv_cache = torch.zeros(src_cache.kv_shape, dtype=torch.float32)
    src_cache.pos = T
    src_cache.presyn_state = [make_state(B_src) for _ in range(n_layers)]

    tgt_cache = KVCache(batch_size=B_tgt, num_heads=H, seq_len=T, head_dim=8, num_layers=n_layers)
    tgt_cache.prefill(src_cache)

    assert isinstance(tgt_cache.presyn_state, list)
    assert len(tgt_cache.presyn_state) == n_layers
    for st in tgt_cache.presyn_state:
        assert st["C"].shape[0] == B_tgt
        assert st["RRP"].shape[0] == B_tgt


def test_gpt_synaptic_preserves_per_layer_presyn_state():
    from bio_inspired_nanochat.engine import KVCache
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig

    cfg = GPTSynapticConfig(
        sequence_len=16,
        vocab_size=99,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
    )
    model = GPTSynaptic(cfg).eval()

    head_dim = cfg.n_embd // cfg.n_head
    kv_cache = KVCache(
        batch_size=1,
        num_heads=cfg.n_kv_head,
        seq_len=cfg.sequence_len,
        head_dim=head_dim,
        num_layers=cfg.n_layer,
    )

    idx = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)
    logits, _ = model(idx, kv_cache=kv_cache, train_mode=False)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert isinstance(kv_cache.presyn_state, list)
    assert len(kv_cache.presyn_state) == cfg.n_layer
    for st in kv_cache.presyn_state:
        assert "C" in st and "RRP" in st and "E" in st


def test_splitmerge_resets_dead_expert():
    from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
    from bio_inspired_nanochat.synaptic_splitmerge import SplitMergeConfig, SplitMergeController

    cfg = SynapticConfig(enable_hebbian=False, enable_metabolism=True)
    moe = SynapticMoE(n_embd=8, num_experts=2, top_k=1, hidden_mult=1, cfg=cfg, dropout=0.0)

    # Legacy clone-and-reset contract (function_preserving=False). The function-preserving
    # default is covered separately in tests/test_function_preserving_splitmerge.py, where the
    # reset twin INHERITS the parent's metabolic state instead of resetting it to 0/1.
    sm_cfg = SplitMergeConfig(
        enabled=True,
        warmup_steps=0,
        min_step_interval=0,
        merges_per_call=0,
        splits_per_call=0,
        reset_health_max=0.05,
        resets_per_call=1,
        ddp_broadcast=False,
        function_preserving=False,
    )
    ctrl = SplitMergeController(moe, sm_cfg)

    with torch.no_grad():
        moe.fatigue.zero_()
        moe.energy.zero_()
        moe.fatigue[1] = 1.0
        moe.energy[1] = 1.0

    ctrl.step(global_step=10, optimizer=None)

    assert moe.fatigue[0].item() == 0.0
    assert moe.energy[0].item() == 1.0


def test_splitmerge_splits_clone_router_row():
    from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
    from bio_inspired_nanochat.synaptic_splitmerge import SplitMergeConfig, SplitMergeController

    cfg = SynapticConfig(enable_hebbian=False, enable_metabolism=True)
    moe = SynapticMoE(n_embd=8, num_experts=2, top_k=1, hidden_mult=1, cfg=cfg, dropout=0.0)

    sm_cfg = SplitMergeConfig(
        enabled=True,
        warmup_steps=0,
        min_step_interval=0,
        merges_per_call=0,
        splits_per_call=1,
        resets_per_call=0,
        split_health_min=0.5,
        clone_noise_linear=0.0,
        clone_noise_router=0.0,
        clone_noise_embed=0.0,
        ddp_broadcast=False,
        function_preserving=False,  # legacy clone-and-reset contract (see FP test file)
    )
    ctrl = SplitMergeController(moe, sm_cfg)

    with torch.no_grad():
        moe.fatigue.zero_()
        moe.energy.zero_()
        moe.fatigue[0] = 1.0
        moe.energy[0] = 1.0
        moe.fatigue[1] = 0.1
        moe.energy[1] = 0.1

    src_row = moe.router.weight[0].detach().clone()
    src_emb = moe.router_embeddings[0].detach().clone()

    ctrl.step(global_step=10, optimizer=None)

    assert torch.allclose(moe.router.weight[1], src_row, atol=0.0, rtol=0.0)
    assert torch.allclose(moe.router_embeddings[1], src_emb, atol=0.0, rtol=0.0)
    assert moe.fatigue[1].item() == 0.0
    assert moe.energy[1].item() == 1.0


def test_splitmerge_merges_clones_router_row():
    from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
    from bio_inspired_nanochat.synaptic_splitmerge import SplitMergeConfig, SplitMergeController

    cfg = SynapticConfig(enable_hebbian=False, enable_metabolism=True)
    moe = SynapticMoE(n_embd=8, num_experts=2, top_k=1, hidden_mult=1, cfg=cfg, dropout=0.0)

    sm_cfg = SplitMergeConfig(
        enabled=True,
        warmup_steps=0,
        min_step_interval=0,
        merges_per_call=1,
        splits_per_call=0,
        resets_per_call=0,
        merge_cosine_threshold=0.0,
        merge_health_max=0.2,
        clone_noise_linear=0.0,
        clone_noise_router=0.0,
        clone_noise_embed=0.0,
        ddp_broadcast=False,
        function_preserving=False,  # legacy clone-and-reset contract (see FP test file)
    )
    ctrl = SplitMergeController(moe, sm_cfg)

    with torch.no_grad():
        moe.fatigue.fill_(0.1)
        moe.energy.fill_(0.1)
        moe.router_embeddings[1].copy_(moe.router_embeddings[0])

    ctrl.step(global_step=10, optimizer=None)

    assert torch.allclose(moe.router.weight[1], moe.router.weight[0], atol=0.0, rtol=0.0)
    assert torch.allclose(moe.router_embeddings[1], moe.router_embeddings[0], atol=0.0, rtol=0.0)
    assert moe.fatigue[1].item() == 0.0
    assert moe.energy[1].item() == 1.0


def test_postsynaptic_genes_scale_camkii_pp1_updates():
    from bio_inspired_nanochat.synaptic import PostsynapticHebb, SynapticConfig

    cfg = SynapticConfig(
        enable_hebbian=True,
        camkii_thr=0.0,
        pp1_thr=10.0,
        camkii_up=0.1,
        pp1_tau=0.9,
    )
    hebb = PostsynapticHebb(d_k=4, d_v=8, cfg=cfg)
    y = torch.zeros((2, 8), dtype=torch.float32)
    ca_proxy = torch.ones(8, dtype=torch.float32)

    with torch.no_grad():
        hebb.camkii.zero_()
        hebb.pp1.zero_()
        hebb.update(y, ca_proxy, genes=torch.tensor([0.0, 0.0, 0.5, 0.5]))
        cam_lo = hebb.camkii.mean().item()
        pp1_lo = hebb.pp1.mean().item()

        hebb.camkii.zero_()
        hebb.pp1.zero_()
        hebb.update(y, ca_proxy, genes=torch.tensor([0.0, 0.0, 2.0, 2.0]))
        cam_hi = hebb.camkii.mean().item()
        pp1_hi = hebb.pp1.mean().item()

    assert cam_hi > cam_lo
    assert pp1_hi > pp1_lo


def test_ca_init_weight_linear_is_deterministic_and_fan_avg_scaled():
    from bio_inspired_nanochat.common import ca_init_weight_

    w1 = torch.empty((64, 32), dtype=torch.float32)
    w2 = torch.empty((64, 32), dtype=torch.float32)
    ca_init_weight_(w1, rule=30, seed=123, salt="test.linear", layout="out_in")
    ca_init_weight_(w2, rule=30, seed=123, salt="test.linear", layout="out_in")
    assert torch.allclose(w1, w2)

    fan_in = 32
    fan_out = 64
    target_var = 2.0 / float(fan_in + fan_out)
    assert abs(w1.mean().item()) < 1e-5
    assert abs(w1.var(unbiased=False).item() - target_var) / target_var < 1e-2

    w3 = torch.empty((64, 32), dtype=torch.float32)
    ca_init_weight_(w3, rule=30, seed=124, salt="test.linear", layout="out_in")
    assert not torch.allclose(w1, w3)


def test_ca_init_weight_conv_shape_is_fan_avg_scaled():
    from bio_inspired_nanochat.common import ca_init_weight_

    w = torch.empty((8, 3, 3, 3), dtype=torch.float32)
    ca_init_weight_(w, rule=116, seed=7, salt="test.conv", layout="out_in")
    fan_out = 8
    fan_in = 3 * 3 * 3
    target_var = 2.0 / float(fan_in + fan_out)
    assert abs(w.mean().item()) < 1e-5
    assert abs(w.var(unbiased=False).item() - target_var) / target_var < 1e-2


def test_gpt_ca_init_applies_to_block_linears():
    from bio_inspired_nanochat.gpt import GPT, GPTConfig

    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=97,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        n_embd=32,
        init_type="ca_rule30",
        init_seed=123,
    )
    model = GPT(cfg)
    model.init_weights()

    w = model.state_dict()["transformer.h.0.mlp.c_fc.weight"].detach().cpu()
    fan_out = int(w.size(0))
    fan_in = int(w.size(1))
    target_var = 2.0 / float(fan_in + fan_out)
    assert abs(w.mean().item()) < 1e-5
    assert abs(w.var(unbiased=False).item() - target_var) / target_var < 2e-2


def test_sample_next_token_top_k_zero_is_disabled_not_degenerate():
    """top_k=0 used to hit torch.topk(logits, 0) -> empty -> multinomial crash. It must instead be
    treated as 'no truncation' (HF convention), sampling over the full vocabulary."""
    from bio_inspired_nanochat.engine import sample_next_token

    rng = torch.Generator()
    rng.manual_seed(0)
    logits = torch.randn(3, 17)
    out = sample_next_token(logits, rng, temperature=1.0, top_k=0)  # must not raise
    assert out.shape == (3, 1)
    assert (out >= 0).all() and (out < 17).all()


def test_generate_resets_per_sequence_state_at_start():
    """A new prompt is a new scratchpad: because synaptic plasticity runs during inference,
    Engine.generate must reset the per-sequence state once at the start so one generation's fast
    adaptation cannot leak into the next (e.g. across chat turns)."""
    from bio_inspired_nanochat.engine import Engine
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig

    cfg = GPTSynapticConfig(
        sequence_len=64, vocab_size=64, n_layer=1, n_head=2, n_kv_head=2, n_embd=32
    )
    model = GPTSynaptic(cfg).eval()

    calls = {"n": 0}
    orig_reset = model.reset_sequence_state

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return orig_reset(*args, **kwargs)

    model.reset_sequence_state = _spy  # instance attribute shadows the bound method

    class _FakeTok:
        _special = {
            "<|python_start|>": -1, "<|python_end|>": -2,
            "<|output_start|>": -3, "<|output_end|>": -4, "<|assistant_end|>": -5,
        }
        def encode_special(self, s):
            return self._special[s]
        def get_bos_token_id(self):
            return -10
        def decode(self, toks):
            return ""
        def encode(self, s):
            return []

    eng = Engine(model, _FakeTok())
    # Drive the generator to completion (it is lazy; the reset fires on first iteration).
    list(eng.generate(tokens=[1, 2, 3], num_samples=1, max_tokens=2, temperature=1.0, top_k=None, seed=0))
    assert calls["n"] == 1, "generate() must reset per-sequence state exactly once at the start"


def test_gpt_synaptic_init_weights_after_to_empty_is_finite():
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
    from bio_inspired_nanochat.synaptic import SynapticConfig

    cfg = GPTSynapticConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=32,
        synapses=True,
        syn_cfg=SynapticConfig(
            enable_presyn=True,
            enable_hebbian=True,
            enable_metabolism=False,
            native_genetics=False,
        ),
        init_type="ca_rule30",
        init_seed=7,
    )

    with torch.device("meta"):
        model = GPTSynaptic(cfg)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()

    idx = torch.randint(0, cfg.vocab_size, (2, 16), dtype=torch.long)
    logits, loss = model(idx, targets=None)
    assert loss is None
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()
