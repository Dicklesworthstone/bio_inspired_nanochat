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


# -----------------------------------------------------------------------------
# Multi-query detached causal scan (l7c9 / jyb.2): training and prefill
# -----------------------------------------------------------------------------


@triton.jit
def _round_half_even(x):
    """torch.round semantics (ties to even) for the non-negative vesicle pool sizes."""
    lower = tl.floor(x)
    frac = x - lower
    lower_is_even = (lower - 2.0 * tl.floor(lower * 0.5)) == 0.0
    tie_up = (frac == 0.5) & (~lower_is_even)
    return tl.where((frac > 0.5) | tie_up, lower + 1.0, lower)


@triton.jit
def presyn_detached_scan_kernel(
    Drive_ptr,
    Idx_ptr,
    Valid_ptr,
    Uniform_ptr,
    Noise_ptr,
    State_ptr,
    Delay_ptr,
    ERaw_ptr,
    Rec_ptr,
    Mask_ptr,
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
    stochastic_frac,
    count_cap,
    N_SEQUENCE,
    T_KEY,
    FIRST_ACTIVE,
    TOPK,
    T_QUERY: tl.constexpr,
    NUM_TILES: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK_KEYS: tl.constexpr,
    N_DELAY: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    RECORD: tl.constexpr,
):
    """One program per (batch, head) row advances the exact causal recurrence query by query.

    The math is op-for-op ``synaptic._scripted_detached_presyn_scan``: query ``t`` gathers the
    state of its top-k keys, releases, and then every key of the active prefix
    ``FIRST_ACTIVE + t`` advances (accessed keys with their summed release/drive, the rest by
    the idle relaxation). The state lives in global memory and is updated in place; the two
    barriers order "every lane gathered" before "any key is overwritten" and "every key was
    written" before the next query gathers. Duplicate indices accumulate like ``scatter_add_``.

    Outputs the un-normalized edge release ``qamp * released``; the EMA normalizer couples all
    rows, so the wrapper applies it afterwards (it never feeds back into the state). With
    ``RECORD`` the gathered pre-update edge state (and the stochastic mask) is written for the
    one-pass drive gradient (``synaptic.presyn_edge_release``); no backward kernel is needed.
    Loop extents are compile-time (T_QUERY, NUM_TILES = cdiv(T_KEY, BLOCK_KEYS)) and tiles past
    the active prefix are skipped at run time.
    """
    sequence = tl.program_id(0)
    edge = tl.arange(0, K_PAD)
    edge_mask = edge < TOPK
    plane = N_SEQUENCE * T_KEY
    row = sequence * T_KEY
    rec_plane = N_SEQUENCE * T_QUERY * TOPK
    for t in range(0, T_QUERY):
        active = FIRST_ACTIVE + t
        edge_offset = (sequence * T_QUERY + t) * TOPK + edge
        valid = tl.load(Valid_ptr + edge_offset, mask=edge_mask, other=0).to(tl.int1) & edge_mask
        selected = tl.load(Idx_ptr + edge_offset, mask=edge_mask, other=0)
        selected = tl.where(valid, selected, 0)
        drive = tl.load(Drive_ptr + edge_offset, mask=edge_mask, other=0.0).to(tl.float32)
        gather = row + selected
        c_edge = tl.load(State_ptr + 0 * plane + gather, mask=edge_mask, other=0.0)
        buf_edge = tl.load(State_ptr + 1 * plane + gather, mask=edge_mask, other=0.0)
        rrp_edge = tl.load(State_ptr + 2 * plane + gather, mask=edge_mask, other=0.0)
        pr_edge = tl.load(State_ptr + 4 * plane + gather, mask=edge_mask, other=0.0)
        cl_edge = tl.load(State_ptr + 5 * plane + gather, mask=edge_mask, other=0.0)
        energy_edge = tl.load(State_ptr + 6 * plane + gather, mask=edge_mask, other=0.0)
        if RECORD:
            tl.store(Rec_ptr + 0 * rec_plane + edge_offset, c_edge, mask=edge_mask)
            tl.store(Rec_ptr + 1 * rec_plane + edge_offset, buf_edge, mask=edge_mask)
            tl.store(Rec_ptr + 2 * rec_plane + edge_offset, pr_edge, mask=edge_mask)
            tl.store(Rec_ptr + 3 * rec_plane + edge_offset, cl_edge, mask=edge_mask)
            tl.store(Rec_ptr + 4 * rec_plane + edge_offset, rrp_edge, mask=edge_mask)
            tl.store(Rec_ptr + 5 * rec_plane + edge_offset, energy_edge, mask=edge_mask)

        calcium = tl.maximum(
            rho_c * c_edge
            + alpha_ca * _stable_softplus(drive)
            - alpha_buf_on * c_edge * (1.0 - buf_edge)
            + alpha_buf_off * buf_edge,
            0.0,
        )
        fast = calcium / (calcium + syt_fast_kd)
        slow = calcium / (calcium + syt_slow_kd)
        sensor = 0.7 * fast + 0.3 * slow + doc2_gain * _sigmoid(4.0 * (calcium - 0.12))
        fuse_base = _sigmoid(3.0 * sensor + 2.0 * pr_edge - 2.0 * (cl_edge + complexin_bias))
        probability = tl.minimum(tl.maximum(fuse_base * _sigmoid(drive), 0.0), 1.0)
        released = probability * rrp_edge
        if STOCHASTIC:
            uniform = tl.load(Uniform_ptr + sequence * T_QUERY + t) + tl.zeros(
                (K_PAD,), dtype=tl.float32
            )
            sampled_mask = (uniform < stochastic_frac) & valid
            count = tl.minimum(tl.maximum(_round_half_even(rrp_edge), 0.0), count_cap)
            p32 = tl.minimum(tl.maximum(probability, 1e-6), 1.0 - 1e-6)
            deviation = tl.sqrt(count * p32 * (1.0 - p32) + 1e-6)
            noise = tl.load(Noise_ptr + edge_offset, mask=edge_mask, other=0.0)
            sampled = tl.minimum(tl.maximum(count * p32 + deviation * noise, 0.0), count)
            released = tl.where(sampled_mask, sampled, released)
            if RECORD:
                tl.store(Mask_ptr + edge_offset, sampled_mask.to(tl.int8), mask=edge_mask)
        released = tl.where(valid, released, 0.0)
        qamp = _sigmoid(q_beta * (energy_edge - 0.5)) * qmax
        tl.store(ERaw_ptr + edge_offset, released * qamp, mask=edge_mask)
        drive_valid = tl.where(valid, drive, 0.0)
        count_valid = tl.where(valid, 1.0, 0.0)

        tl.debug_barrier()
        for tile in range(0, NUM_TILES):
            if tile * BLOCK_KEYS < active:
                key = tile * BLOCK_KEYS + tl.arange(0, BLOCK_KEYS)
                key_mask = key < active
                offset = row + key
                match = (key[:, None] == selected[None, :]) & edge_mask[None, :]
                release_sum = tl.sum(tl.where(match, released[None, :], 0.0), axis=1)
                drive_sum = tl.sum(tl.where(match, drive_valid[None, :], 0.0), axis=1)
                access_count = tl.sum(tl.where(match, count_valid[None, :], 0.0), axis=1)
                accessed = tl.where(access_count > 0.0, 1.0, 0.0)

                c_prev = tl.load(State_ptr + 0 * plane + offset, mask=key_mask, other=0.0)
                buf_prev = tl.load(State_ptr + 1 * plane + offset, mask=key_mask, other=0.0)
                rrp_prev = tl.load(State_ptr + 2 * plane + offset, mask=key_mask, other=0.0)
                res_prev = tl.load(State_ptr + 3 * plane + offset, mask=key_mask, other=0.0)
                pr_prev = tl.load(State_ptr + 4 * plane + offset, mask=key_mask, other=0.0)
                cl_prev = tl.load(State_ptr + 5 * plane + offset, mask=key_mask, other=0.0)
                energy_prev = tl.load(State_ptr + 6 * plane + offset, mask=key_mask, other=0.0)

                c_next = tl.maximum(
                    rho_c * c_prev
                    + alpha_ca * _stable_softplus(drive_sum) * accessed
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
                rrp_next = tl.maximum(rrp_prev - release_sum, 0.0)
                res_next = res_prev
                if N_DELAY > 0:
                    res_next = res_prev + tl.load(Delay_ptr + offset, mask=key_mask, other=0.0)
                    for slot in tl.static_range(N_DELAY - 1):
                        shifted = tl.load(
                            Delay_ptr + (slot + 1) * plane + offset, mask=key_mask, other=0.0
                        )
                        tl.store(Delay_ptr + slot * plane + offset, shifted, mask=key_mask)
                    tl.store(
                        Delay_ptr + (N_DELAY - 1) * plane + offset,
                        release_sum * rec_rate,
                        mask=key_mask,
                    )
                take = tl.minimum(res_next, 1.0)
                res_next = tl.maximum(res_next - prime_rate * take, 0.0)
                rrp_next = tl.minimum(tl.maximum(rrp_next + prime_rate * take, 0.0), 30.0)
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
                tl.store(State_ptr + 0 * plane + offset, c_next, mask=key_mask)
                tl.store(State_ptr + 1 * plane + offset, buf_next, mask=key_mask)
                tl.store(State_ptr + 2 * plane + offset, rrp_next, mask=key_mask)
                tl.store(State_ptr + 3 * plane + offset, res_next, mask=key_mask)
                tl.store(State_ptr + 4 * plane + offset, pr_next, mask=key_mask)
                tl.store(State_ptr + 5 * plane + offset, cl_next, mask=key_mask)
                tl.store(State_ptr + 6 * plane + offset, energy_next, mask=key_mask)
        tl.debug_barrier()


def _ema_schedule(scale, ema0):
    """Closed form of ema_t = 0.99 * ema_{t-1} + 0.01 * scale_t for every query at once.

    Evaluated in float64 (0.99**-t stays finite far beyond any sequence length used here), so
    the per-query normalizers match the sequential float32 recurrence to float32 rounding.
    """
    steps = torch.arange(scale.numel(), device=scale.device, dtype=torch.float64)
    decay = torch.full_like(steps, 0.99)
    inv_weight = decay.pow(-steps)
    weighted = torch.cumsum(scale.to(torch.float64) * inv_weight, dim=0)
    ema = decay.pow(steps + 1.0) * ema0.to(torch.float64) + 0.01 * decay.pow(steps) * weighted
    return ema.to(torch.float32)


def presyn_detached_scan(
    state: dict[str, Any],
    drive,
    idx,
    valid,
    cfg,
    *,
    ema_e,
    train: bool,
    first_active_key_count: int,
    stochastic_frac: float = 0.0,
    uniform=None,
    noise=None,
    record_edges: bool = False,
    _interpret: bool = False,
):
    """Run the exact detached causal scan for a (B, H, T, K) query block in one launch.

    Advances ``state`` in place of ``_scripted_detached_presyn_scan`` and returns
    ``(output, ema_after, edges)``: the normalized per-edge release, the persistent EMA after
    the block, and (when ``record_edges``) the constants ``presyn_edge_release`` differentiates.
    ``uniform`` (B, H, T) and ``noise`` (B, H, T, K) are the stochastic draws (required when
    ``stochastic_frac > 0`` and ``train``). ``_interpret`` is for CPU correctness checks under
    ``TRITON_INTERPRET=1``; production callers pass CUDA tensors.
    """
    if drive.ndim != 4 or idx.shape != drive.shape or valid.shape != drive.shape:
        raise ValueError(
            f"drive, idx and valid must share a (B,H,T,K) shape; got {drive.shape}, "
            f"{idx.shape}, {valid.shape}"
        )
    if not drive.is_cuda and not _interpret:
        raise ValueError("the presyn scan kernel requires CUDA tensors")
    if drive.dtype != torch.float32 or state["C"].dtype != torch.float32:
        raise ValueError("the presyn scan kernel runs the float32 state recurrence")
    B, H, T, K = drive.shape
    t_key = int(state["C"].shape[2])
    if first_active_key_count < 1 or first_active_key_count + T - 1 > t_key:
        raise ValueError(
            f"queries [{first_active_key_count}, {first_active_key_count + T - 1}] exceed the "
            f"key-state extent {t_key}"
        )
    stochastic = bool(train and stochastic_frac > 0.0)
    if stochastic and (uniform is None or noise is None):
        raise ValueError("stochastic training needs pre-drawn uniform and noise tensors")
    names = ("C", "BUF", "RRP", "RES", "PR", "CL", "E")
    n_sequence = B * H
    slab = torch.stack([state[name].reshape(n_sequence, t_key) for name in names]).contiguous()
    delay = list(state.get("DELAY", []))
    n_delay = len(delay)
    delay_slab = (
        torch.stack([entry.reshape(n_sequence, t_key) for entry in delay]).contiguous()
        if n_delay
        else slab
    )
    e_raw = torch.zeros(drive.shape, device=drive.device, dtype=torch.float32)
    record = torch.empty(
        (6, B, H, T, K) if record_edges else (1,), device=drive.device, dtype=torch.float32
    )
    mask = torch.zeros(
        drive.shape if (record_edges and stochastic) else (1,),
        device=drive.device,
        dtype=torch.int8,
    )
    uniform_c = uniform.contiguous() if stochastic and uniform is not None else e_raw
    noise_c = noise.contiguous() if stochastic and noise is not None else e_raw
    presyn_detached_scan_kernel[(n_sequence,)](
        drive.contiguous(),
        idx.contiguous(),
        valid.contiguous(),
        uniform_c,
        noise_c,
        slab,
        delay_slab,
        e_raw,
        record,
        mask,
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
        stochastic_frac=float(stochastic_frac),
        count_cap=float(max(0, int(cfg.stochastic_count_cap))),
        N_SEQUENCE=cast(Any, n_sequence),
        T_KEY=cast(Any, t_key),
        FIRST_ACTIVE=cast(Any, int(first_active_key_count)),
        TOPK=cast(Any, K),
        T_QUERY=cast(Any, T),
        NUM_TILES=cast(Any, triton.cdiv(t_key, 128)),
        K_PAD=cast(Any, triton.next_power_of_2(max(1, K))),
        BLOCK_KEYS=cast(Any, 128),
        N_DELAY=cast(Any, n_delay),
        STOCHASTIC=cast(Any, stochastic),
        RECORD=cast(Any, bool(record_edges)),
    )

    shape = (B, H, t_key)
    state.update({name: slab[i].view(shape) for i, name in enumerate(names)})
    if n_delay:
        state["DELAY"] = [delay_slab[i].view(shape) for i in range(n_delay)]

    ema0 = ema_e.detach().reshape(()).to(torch.float32)
    if train:
        weight = valid.to(torch.float32)
        scale = (
            (e_raw.abs() * weight).sum(dim=(0, 1, 3)) / weight.sum(dim=(0, 1, 3)).clamp_min(1.0)
        ).clamp_min(1e-3)
        ema = _ema_schedule(scale, ema0)
    else:
        ema = ema0.expand(T)
    ema_view = ema.reshape(1, 1, T, 1)
    output = e_raw / (ema_view + 1e-6)
    edges: list = []
    if record_edges:
        stochastic_mask = (
            mask.to(torch.bool) if stochastic else torch.zeros(drive.shape, device=drive.device, dtype=torch.bool)
        )
        draw = noise_c if stochastic else torch.zeros_like(drive)
        edges = [record[i] for i in range(6)] + [stochastic_mask, draw, ema_view.clone()]
    return output, ema[-1].reshape(1).clone(), edges
