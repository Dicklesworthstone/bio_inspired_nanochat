from typing import Any, cast

from bio_inspired_nanochat.torch_imports import torch  # noqa: F401
import triton
import triton.language as tl
import math

@triton.jit
def _sigmoid(x):
    x32 = x.to(tl.float32)
    out32 = 1.0 / (1.0 + tl.exp(-x32))
    return out32.to(x.dtype)

# -----------------------------------------------------------------------------
# Live deterministic decode kernel (jyb.2)
# -----------------------------------------------------------------------------


@triton.jit
def _stable_softplus(x):
    """Numerically stable float32 softplus for finite values and +/-inf."""
    x32 = x.to(tl.float32)
    return tl.maximum(x32, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x32)))


@triton.jit
def presyn_live_decode_kernel(
    Drive_ptr,
    Idx_ptr,
    Valid_ptr,
    Dots_ptr,
    C_ptr,
    BUF_ptr,
    RRP_ptr,
    RES_ptr,
    PR_ptr,
    CL_ptr,
    E_ptr,
    Delay0_ptr,
    Ema_ptr,
    EOut_ptr,
    NextState_ptr,
    NextDelay_ptr,
    rho_c,
    rho_b,
    alpha_ca,
    alpha_buf_on,
    alpha_buf_off,
    syt_fast_kd,
    syt_slow_kd,
    doc2_gain,
    complexin_bias,
    q_beta,
    qmax,
    prime_rate,
    unprime_per_release,
    nsf_recover,
    rec_rate,
    energy_fill,
    energy_max,
    energy_use,
    lambda_loge,
    epsilon,
    loge_bias_clamp,
    T_KEY,
    TOPK: tl.constexpr,
    N_SEQUENCE,
    NUM_KEY_BLOCKS,
    BLOCK_KEYS: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_DELAY: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    CLAMP_LOG_BIAS: tl.constexpr,
):
    """One physical kernel for the exact deterministic ``Tq == 1`` canonical step.

    Each program owns a disjoint key tile, scans the small top-k row, emits matching edges, and
    advances every key from the immutable prior-state snapshot. Repeated indices are accumulated
    exactly like the canonical scatter path. No cross-program reduction or grid-wide barrier is
    required because there is only one query position.
    """
    pid = tl.program_id(0)
    sequence = pid // NUM_KEY_BLOCKS
    key_block = pid - sequence * NUM_KEY_BLOCKS
    key = key_block * BLOCK_KEYS + tl.arange(0, BLOCK_KEYS)
    key_mask = key < T_KEY
    state_offset = sequence * T_KEY + key
    edge_base = sequence * TOPK

    c_prev = tl.load(C_ptr + state_offset, mask=key_mask, other=0.0).to(tl.float32)
    buf_prev = tl.load(BUF_ptr + state_offset, mask=key_mask, other=0.0).to(tl.float32)
    rrp_prev = tl.load(RRP_ptr + state_offset, mask=key_mask, other=0.0).to(tl.float32)
    res_prev = tl.load(RES_ptr + state_offset, mask=key_mask, other=0.0).to(tl.float32)
    pr_prev = tl.load(PR_ptr + state_offset, mask=key_mask, other=0.0).to(tl.float32)
    cl_prev = tl.load(CL_ptr + state_offset, mask=key_mask, other=0.0).to(tl.float32)
    energy_prev = tl.load(E_ptr + state_offset, mask=key_mask, other=0.0).to(tl.float32)

    release_sum = tl.zeros((BLOCK_KEYS,), dtype=tl.float32)
    drive_sum = tl.zeros((BLOCK_KEYS,), dtype=tl.float32)
    accessed = tl.zeros((BLOCK_KEYS,), dtype=tl.int1)
    ema_e = tl.load(Ema_ptr).to(tl.float32)

    for edge in range(0, TOPK):
        edge_offset = edge_base + edge
        selected_key = tl.load(Idx_ptr + edge_offset).to(tl.int32)
        drive = tl.load(Drive_ptr + edge_offset).to(tl.float32)
        if HAS_VALID:
            valid = tl.load(Valid_ptr + edge_offset).to(tl.int1)
        else:
            valid = True
        selected_valid = (selected_key >= 0) & (selected_key < T_KEY)
        matches = key_mask & selected_valid & (key == selected_key)
        active = matches & valid

        # Evaluate the expensive edge biology once per key block, not once per key lane. The
        # selected-state loads are scalar; only the owning lane receives the result below.
        selected_offset = sequence * T_KEY + selected_key
        c_selected = tl.load(C_ptr + selected_offset, mask=selected_valid, other=0.0).to(tl.float32)
        buf_selected = tl.load(BUF_ptr + selected_offset, mask=selected_valid, other=0.0).to(
            tl.float32
        )
        rrp_selected = tl.load(RRP_ptr + selected_offset, mask=selected_valid, other=0.0).to(
            tl.float32
        )
        pr_selected = tl.load(PR_ptr + selected_offset, mask=selected_valid, other=0.0).to(
            tl.float32
        )
        cl_selected = tl.load(CL_ptr + selected_offset, mask=selected_valid, other=0.0).to(
            tl.float32
        )
        energy_selected = tl.load(
            E_ptr + selected_offset, mask=selected_valid, other=0.0
        ).to(tl.float32)
        c_edge = tl.maximum(
            rho_c * c_selected
            + alpha_ca * _stable_softplus(drive)
            - alpha_buf_on * c_selected * (1.0 - buf_selected)
            + alpha_buf_off * buf_selected,
            0.0,
        )
        fast = c_edge / (c_edge + syt_fast_kd)
        slow = c_edge / (c_edge + syt_slow_kd)
        syt = 0.7 * fast + 0.3 * slow + doc2_gain * _sigmoid(4.0 * (c_edge - 0.12))
        fuse_base = _sigmoid(
            3.0 * syt + 2.0 * pr_selected - 2.0 * (cl_selected + complexin_bias)
        )
        probability = tl.minimum(tl.maximum(fuse_base * _sigmoid(drive), 0.0), 1.0)
        released = tl.where(valid & selected_valid, probability * rrp_selected, 0.0)
        release_sum += tl.where(matches, released, 0.0)
        drive_sum += tl.where(active, drive, 0.0)
        accessed = accessed | active

        qamp = _sigmoid(q_beta * (energy_selected - 0.5)) * qmax
        normalized_e = released * qamp / (ema_e + 1e-6)
        edge_out_ptr = EOut_ptr + edge_offset + tl.zeros((BLOCK_KEYS,), dtype=tl.int32)
        tl.store(edge_out_ptr, normalized_e, mask=matches)

        if WRITE_LOGITS:
            bias = lambda_loge * tl.log(epsilon + normalized_e)
            if CLAMP_LOG_BIAS:
                bias = tl.minimum(tl.maximum(bias, -loge_bias_clamp), loge_bias_clamp)
            prior_dot = tl.load(Dots_ptr + state_offset, mask=active, other=0.0)
            tl.store(Dots_ptr + state_offset, prior_dot + bias, mask=active)

    accessed_f = accessed.to(tl.float32)
    c_next = tl.maximum(
        rho_c * c_prev
        + alpha_ca * _stable_softplus(drive_sum) * accessed_f
        - alpha_buf_on * c_prev * (1.0 - buf_prev)
        + alpha_buf_off * buf_prev,
        0.0,
    )
    buf_next = tl.minimum(
        tl.maximum(
            rho_b * buf_prev
            + alpha_buf_on * c_prev * (1.0 - buf_prev)
            - alpha_buf_off * buf_prev,
            0.0,
        ),
        1.0,
    )

    rrp_depleted = tl.maximum(rrp_prev - release_sum, 0.0)
    if HAS_DELAY:
        delay0 = tl.load(Delay0_ptr + state_offset, mask=key_mask, other=0.0).to(
            tl.float32
        )
        res_refilled = res_prev + delay0
    else:
        res_refilled = res_prev
    take = tl.minimum(res_refilled, 1.0)
    res_next = tl.maximum(res_refilled - prime_rate * take, 0.0)
    rrp_next = tl.minimum(tl.maximum(rrp_depleted + prime_rate * take, 0.0), 30.0)
    pr_next = tl.minimum(
        tl.maximum(
            pr_prev * (1.0 - unprime_per_release * release_sum)
            + nsf_recover * (1.0 - pr_prev),
            0.0,
        ),
        1.0,
    )
    cl_next = tl.minimum(
        tl.maximum(cl_prev * 0.995 + 0.005 - unprime_per_release * release_sum, 0.0),
        1.0,
    )
    energy_next = tl.minimum(
        tl.maximum(
            energy_prev
            + energy_fill * (energy_max - energy_prev)
            - energy_use * release_sum,
            0.0,
        ),
        energy_max,
    )

    # NextState is laid out (7, N_sequence, T_KEY), so every returned state view is contiguous.
    total_state_stride = N_SEQUENCE * T_KEY
    tl.store(
        NextState_ptr + 0 * total_state_stride + state_offset, c_next, mask=key_mask
    )
    tl.store(
        NextState_ptr + 1 * total_state_stride + state_offset, buf_next, mask=key_mask
    )
    tl.store(
        NextState_ptr + 2 * total_state_stride + state_offset, rrp_next, mask=key_mask
    )
    tl.store(
        NextState_ptr + 3 * total_state_stride + state_offset, res_next, mask=key_mask
    )
    tl.store(
        NextState_ptr + 4 * total_state_stride + state_offset, pr_next, mask=key_mask
    )
    tl.store(
        NextState_ptr + 5 * total_state_stride + state_offset, cl_next, mask=key_mask
    )
    tl.store(
        NextState_ptr + 6 * total_state_stride + state_offset,
        energy_next,
        mask=key_mask,
    )
    if HAS_DELAY:
        # The queue tail has its own one-plane allocation: this avoids retaining a full state slab
        # through a view and preserves canonical replacement semantics for aliases of DELAY[0].
        tl.store(NextDelay_ptr + state_offset, release_sum * rec_rate, mask=key_mask)


def presyn_live_decode_step(
    state: dict[str, Any],
    drive,
    idx,
    cfg,
    *,
    ema_e,
    valid=None,
    logits=None,
    _interpret: bool = False,
):
    """Launch the canonical one-query Triton step and replace state tensors atomically.

    ``_interpret`` exists only for CPU correctness development under ``TRITON_INTERPRET=1``;
    production callers must pass CUDA tensors. The caller is responsible for the narrow dispatch
    contract (deterministic, no-grad, fixed kinetics, no metriplectic integration) and for trusted
    in-bounds indices from the live attention ``topk`` producer. This low-level wrapper is not
    exported from ``bio_inspired_nanochat.kernels``.
    """
    if drive.ndim != 4 or drive.shape[2] != 1:
        raise ValueError(
            f"live presyn kernel requires drive shape (B,H,1,K), got {drive.shape}"
        )
    if idx.shape != drive.shape:
        raise ValueError(
            f"idx shape must match drive shape {drive.shape}, got {idx.shape}"
        )
    if valid is not None and valid.shape != drive.shape:
        raise ValueError(
            f"valid shape must match drive shape {drive.shape}, got {valid.shape}"
        )
    if idx.dtype != torch.int64:
        raise ValueError(f"idx must have dtype torch.int64, got {idx.dtype}")
    if valid is not None and valid.dtype != torch.bool:
        raise ValueError(f"valid must have dtype torch.bool, got {valid.dtype}")
    if not drive.is_cuda and not _interpret:
        raise ValueError("live presyn Triton kernel requires CUDA tensors")
    if drive.dtype != torch.float32:
        raise ValueError(f"live presyn Triton kernel requires float32, got {drive.dtype}")

    B, H, _, topk = drive.shape
    state_shape = state["C"].shape
    if len(state_shape) != 3 or state_shape[:2] != (B, H):
        raise ValueError(f"state shape must begin with {(B, H)}, got {state_shape}")
    t_key = int(state_shape[2])
    expected_state_shape = (B, H, t_key)
    state_names = ("C", "BUF", "RRP", "RES", "PR", "CL", "E")
    for name in state_names:
        tensor = state[name]
        if tensor.shape != expected_state_shape:
            raise ValueError(
                f"state[{name!r}] must have shape {expected_state_shape}, got {tensor.shape}"
            )
        if tensor.device != drive.device or tensor.dtype != state["C"].dtype:
            raise ValueError(
                "all live presyn state tensors must share one device and dtype"
            )
    if state["C"].dtype != drive.dtype:
        raise ValueError("live presyn state and drive must share one dtype")
    if idx.device != drive.device:
        raise ValueError("idx and drive must be on the same device")
    if ema_e.numel() != 1 or ema_e.device != drive.device:
        raise ValueError("ema_e must be a one-element tensor on the drive device")
    if logits is not None:
        expected_logits_shape = (B, H, 1, t_key)
        if (
            logits.shape != expected_logits_shape
            or logits.device != drive.device
            or logits.dtype != drive.dtype
        ):
            raise ValueError(
                f"logits must have shape {expected_logits_shape}, device {drive.device}, and "
                f"dtype {drive.dtype}; got shape={logits.shape}, device={logits.device}, "
                f"dtype={logits.dtype}"
            )
        if not logits.is_contiguous():
            raise ValueError(
                "logits must be contiguous for in-place fused bias injection"
            )

    n_sequence = B * H
    state_inputs = [
        state[name].reshape(n_sequence, t_key).contiguous() for name in state_names
    ]
    c_state, buf_state, rrp_state, res_state, pr_state, cl_state, energy_state = (
        state_inputs
    )
    delay = state.get("DELAY", [])
    if not isinstance(delay, list):
        raise TypeError("state['DELAY'] must be a list of tensors")
    has_delay = cfg.endo_delay > 0
    if has_delay:
        if len(delay) != cfg.endo_delay:
            raise ValueError(
                f"state['DELAY'] must contain {cfg.endo_delay} entries, got {len(delay)}"
            )
        if any(
            entry.shape != expected_state_shape
            or entry.device != drive.device
            or entry.dtype != drive.dtype
            for entry in delay
        ):
            raise ValueError(
                "state['DELAY'] entries must match the key-state shape, device, and dtype"
            )
        delay0 = delay[0].reshape(n_sequence, t_key).contiguous()
    else:
        delay0 = state_inputs[0]

    drive_c = drive.contiguous()
    idx_c = idx.contiguous()
    valid_c = valid.contiguous() if valid is not None else drive_c
    # The supported jyb.2 slice keeps the canonical release and persistent EMA in float32.
    e_out = torch.zeros(drive_c.shape, device=drive.device, dtype=torch.float32)
    next_state = torch.empty(
        (7, n_sequence, t_key), device=drive.device, dtype=state["C"].dtype
    )
    next_delay_buffer = (
        torch.empty((n_sequence, t_key), device=drive.device, dtype=drive.dtype)
        if has_delay
        else c_state
    )
    dots = logits if logits is not None else e_out
    block_keys = 128
    num_key_blocks = triton.cdiv(t_key, block_keys)
    grid = (n_sequence * num_key_blocks,)

    presyn_live_decode_kernel[grid](
        drive_c,
        idx_c,
        valid_c,
        dots,
        c_state,
        buf_state,
        rrp_state,
        res_state,
        pr_state,
        cl_state,
        energy_state,
        delay0,
        ema_e,
        e_out,
        next_state,
        next_delay_buffer,
        rho_c=math.exp(-1.0 / cfg.tau_c),
        rho_b=math.exp(-1.0 / cfg.tau_buf),
        alpha_ca=cfg.alpha_ca,
        alpha_buf_on=cfg.alpha_buf_on,
        alpha_buf_off=cfg.alpha_buf_off,
        syt_fast_kd=cfg.syt_fast_kd,
        syt_slow_kd=cfg.syt_slow_kd,
        doc2_gain=cfg.doc2_gain,
        complexin_bias=cfg.complexin_bias,
        q_beta=cfg.q_beta,
        qmax=cfg.qmax,
        prime_rate=cfg.prime_rate,
        unprime_per_release=cfg.unprime_per_release,
        nsf_recover=cfg.nsf_recover,
        rec_rate=cfg.rec_rate,
        energy_fill=cfg.energy_fill,
        energy_max=cfg.energy_max,
        energy_use=cfg.energy_use,
        lambda_loge=cfg.lambda_loge,
        epsilon=cfg.epsilon,
        loge_bias_clamp=cfg.loge_bias_clamp,
        T_KEY=cast(Any, t_key),
        TOPK=cast(Any, topk),
        N_SEQUENCE=cast(Any, n_sequence),
        NUM_KEY_BLOCKS=cast(Any, num_key_blocks),
        BLOCK_KEYS=cast(Any, block_keys),
        HAS_VALID=cast(Any, valid is not None),
        HAS_DELAY=cast(Any, has_delay),
        WRITE_LOGITS=cast(Any, logits is not None),
        CLAMP_LOG_BIAS=cast(
            Any, bool(cfg.loge_bias_clamp and cfg.loge_bias_clamp > 0.0)
        ),
    )

    shape = (B, H, t_key)
    next_delay = list(delay[1:]) + [next_delay_buffer.view(shape)] if has_delay else []
    state.update(
        {
            "C": next_state[0].view(shape),
            "BUF": next_state[1].view(shape),
            "RRP": next_state[2].view(shape),
            "RES": next_state[3].view(shape),
            "PR": next_state[4].view(shape),
            "CL": next_state[5].view(shape),
            "E": next_state[6].view(shape),
            "DELAY": next_delay,
        }
    )
    return e_out
