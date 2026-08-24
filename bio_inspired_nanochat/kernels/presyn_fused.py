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

@triton.jit
def _softplus(x):
    # softplus(x) = log(1 + exp(x))
    # For numerical stability, if x > 20, returns x
    x32 = x.to(tl.float32)
    out32 = tl.where(x32 > 20.0, x32, tl.log(1.0 + tl.exp(x32)))
    return out32.to(x.dtype)

@triton.jit
def presyn_fused_kernel_forward(
    Q_ptr, K_ptr, Logits_ptr,  # (B,H,T,D), (B,H,T,D), (B,H,T,T)
    C_ptr, BUF_ptr, RRP_ptr, RES_ptr, PR_ptr, CL_ptr, E_ptr, # (B,H,T)
    # Strides
    Logits_stride_b, Logits_stride_h, Logits_stride_t, Logits_stride_k,
    Q_stride_b, Q_stride_h, Q_stride_t, Q_stride_d,
    K_stride_b, K_stride_h, K_stride_t, K_stride_d,
    State_stride_b, State_stride_h, State_stride_t, 
    # Config constants
    T: tl.constexpr, D: tl.constexpr, BLOCK_SIZE_T: tl.constexpr,
    tau_c, tau_buf, tau_prime, tau_rrp, tau_energy,
    alpha_ca, alpha_buf_on, alpha_buf_off,
    alpha_prime, alpha_refill, alpha_unprime,
    energy_in, energy_cost_rel, energy_cost_pump,
    syt_fast_kd, syt_slow_kd, complexin_bias,
    qmax, q_beta, barrier_strength, epsilon, sqrt_D
):
    # Program ID (sequence index = batch * n_head + head)
    pid = tl.program_id(0)
    
    # Pointers to the start of this sequence (Batch, Head)
    # We assume input tensors are contiguous in B,H or we use the strides passed
    # pid corresponds to a flattened B*H index.
    # BUT we need to decompose pid into b, h if strides are arbitrary.
    # Assuming row-major packing (B, H, ...), pid maps linearly if strides match.
    # To be safe, we compute offsets using passed strides.
    # Wait, if we launch grid(B*H), we don't know H inside unless passed.
    # Let's assume standard contiguous layout for B*H part or just rely on pointer arithmetic 
    # from the caller who passes `ptr + pid * stride`.
    # NO, Triton kernel takes base pointers.
    
    # Simplified: Assume caller passes pointers to the specific sequence?
    # No, we launch one kernel for all sequences.
    
    # Let's compute the offsets for this sequence.
    # We need `num_heads` to split pid into b and h if strides differ.
    # Actually, we can just use the strides directly if we know `stride_h` and `stride_b`.
    # b = pid // num_heads
    # h = pid % num_heads
    # But we need num_heads passed in.
    
    # NOTE: For MVP, let's assume contiguous B*H and just treat as a batch of size (B*H).
    # The caller (Python) will reshape (B, H, T, ...) -> (B*H, T, ...) before passing.
    # This simplifies strides to just stride_seq and stride_t.
    
    # Offsets for this sequence
    # q_ptr = Q_ptr + pid * Q_stride_h * Q_stride_b 
    # Python wrapper will pass `Q` as (Batch*Head, T, D). 
    # So Q_ptr refers to (B*H, T, D).
    # Q_stride_b becomes "stride of sequence".
    
    # Actually, let's fix the signature in the python wrapper to flattening.
    # Then Q is (N_seq, T, D).
    # Q_stride_n, Q_stride_t, Q_stride_d.
    
    # Base pointers for this sequence
    # Q: (pid) * stride_seq
    # Logits: (pid) * stride_seq
    # State: (pid) * stride_seq
    
    # We calculate the specific start pointers
    q_seq_ptr = Q_ptr + pid * Q_stride_b
    k_seq_ptr = K_ptr + pid * K_stride_b
    l_seq_ptr = Logits_ptr + pid * Logits_stride_b
    
    c_ptr = C_ptr + pid * State_stride_b
    buf_ptr = BUF_ptr + pid * State_stride_b
    rrp_ptr = RRP_ptr + pid * State_stride_b
    res_ptr = RES_ptr + pid * State_stride_b
    pr_ptr = PR_ptr + pid * State_stride_b
    cl_ptr = CL_ptr + pid * State_stride_b
    e_ptr = E_ptr + pid * State_stride_b
    
    # Constants pre-calc
    rho_c = tl.exp(-1.0 / tau_c)
    rho_b = tl.exp(-1.0 / tau_buf)
    rho_p = tl.exp(-1.0 / tau_prime)
    rho_r = tl.exp(-1.0 / tau_rrp)
    rho_e = tl.exp(-1.0 / tau_energy)
    
    # Initial State (will be overwritten immediately by new step values)
    c_curr = 0.0
    buf_curr = 0.0
    rrp_curr = 0.8
    res_curr = 0.2
    pr_curr = 0.6
    e_curr = 0.8
    
    # Iterate over time t
    for t in range(T):
        # 1. Calculate Influx
        influx_accum = 0.0
        count_accum = 0.0
        
        for j_start in range(0, t + 1, BLOCK_SIZE_T):
            offsets = j_start + tl.arange(0, BLOCK_SIZE_T)
            mask = offsets <= t
            
            # Load logits row t, cols j
            # ptr = l_seq_ptr + t * stride_t + offsets * stride_k
            ptr = l_seq_ptr + t * Logits_stride_t + offsets * Logits_stride_k
            val = tl.load(ptr, mask=mask, other=0.0)
            
            # Softplus
            val_clamped = tl.where(val < -20.0, -20.0, val)
            val_clamped = tl.where(val_clamped > 20.0, 20.0, val_clamped)
            drive = _softplus(val_clamped)
            
            influx_accum += tl.sum(tl.where(mask, drive, 0.0))
            count_accum += tl.sum(tl.where(mask, 1.0, 0.0))
            
        influx = influx_accum / tl.maximum(count_accum, 1.0)
        
        # 2. Update State
        c_next = rho_c * c_curr + alpha_ca * influx - alpha_buf_on * c_curr * (1.0 - buf_curr) + alpha_buf_off * buf_curr
        buf_next = rho_b * buf_curr + alpha_buf_on * c_curr * (1.0 - buf_curr) - alpha_buf_off * buf_curr
        c_next = tl.where(c_next < 0.0, 0.0, c_next)
        buf_next = tl.where(buf_next < 0.0, 0.0, buf_next)
        buf_next = tl.where(buf_next > 1.0, 1.0, buf_next)
        
        pr_mid = (rho_p * pr_curr + alpha_prime * (1.0 - pr_curr))
        pr_mid = tl.where(pr_mid < 0.0, 0.0, pr_mid)
        pr_mid = tl.where(pr_mid > 1.0, 1.0, pr_mid)
        
        rrp_refill = (rho_r * rrp_curr + alpha_refill * res_curr)
        rrp_refill = tl.where(rrp_refill < 0.0, 0.0, rrp_refill)
        rrp_refill = tl.where(rrp_refill > 1.0, 1.0, rrp_refill)
        
        res_mid = (res_curr - alpha_refill * res_curr)
        res_mid = tl.where(res_mid < 0.0, 0.0, res_mid)
        res_mid = tl.where(res_mid > 1.0, 1.0, res_mid)
        
        e_mid = (rho_e * e_curr + energy_in)
        e_mid = tl.where(e_mid < 0.0, 0.0, e_mid)
        e_mid = tl.where(e_mid > 1.6, 1.6, e_mid)
        
        # 3. Release
        fast = c_next / (c_next + syt_fast_kd)
        slow = c_next / (c_next + syt_slow_kd)
        syt = 0.7 * fast + 0.3 * slow
        
        # CL is usually constant, load from state
        cl_val = tl.load(cl_ptr + t * State_stride_t)
        
        fuse_pre = 3.0 * syt + 2.0 * pr_mid - 2.0 * (cl_val + complexin_bias)
        fuse_sig = _sigmoid(fuse_pre)
        
        # Load Q[t] vector
        # Q ptr: q_seq_ptr + t * stride_t + arange(D) * stride_d
        q_offs = tl.arange(0, D)
        q_ptr_t = q_seq_ptr + t * Q_stride_t + q_offs * Q_stride_d
        q_vec = tl.load(q_ptr_t)
        
        raw_rel_accum = 0.0
        
        # Pass 1: Compute Sum Release
        for j_start in range(0, t + 1, BLOCK_SIZE_T):
            offsets = j_start + tl.arange(0, BLOCK_SIZE_T)
            mask = offsets <= t
            
            # Load K block: (BLOCK, D)
            # K ptr: k_seq_ptr + offsets[:, None]*stride_t + arange(D)[None, :]*stride_d
            # We need to be careful about broadcasting.
            k_ptrs = k_seq_ptr + (offsets[:, None] * K_stride_t) + (q_offs[None, :] * K_stride_d)
            k_block = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
            
            # Dot product Q[t] . K[j]
            # q_vec is (D). k_block is (BLOCK, D).
            # dot = sum(q * k, axis=1) -> (BLOCK)
            dot = tl.sum(q_vec[None, :] * k_block, axis=1)
            dot = dot / sqrt_D
            
            d_bilin = _sigmoid(dot)
            fuse_p = fuse_sig * d_bilin
            raw_rel = fuse_p * rrp_refill
            raw_rel = tl.where(raw_rel < 0.0, 0.0, raw_rel)
            raw_rel = tl.where(raw_rel > 1.0, 1.0, raw_rel)
            
            raw_rel_accum += tl.sum(tl.where(mask, raw_rel, 0.0))
            
        # 4. Scale & Update
        scale = 1.0
        if raw_rel_accum > 0.0:
            ratio = rrp_refill / raw_rel_accum
            scale = tl.where(ratio < 1.0, ratio, 1.0)
            
        used_rrp = 0.0
        
        # Pass 2: Compute used_rrp and SynLogit (need to recompute raw_rel)
        # Since we can't store intermediate T array in registers easily for large T.
        # We fuse the loop: recompute raw_rel, scale it, update logits, accumulate used.
        
        # Need E_new to compute qamp?
        # Wait, logic:
        # used_rrp = sum(release_frac)
        # E_new depends on used_rrp.
        # Qamp depends on E_new.
        # SynLogit depends on Qamp and release_frac.
        
        # So we need `used_rrp` BEFORE we can write SynLogit.
        # This requires the re-computation loop pattern:
        # Loop 1: Sum raw release -> get scale.
        # Loop 2: Sum scaled release -> get used_rrp. (Or just used_rrp = raw_rel_accum * scale?)
        #         Yes! used_rrp = raw_rel_accum * scale (mathematically true).
        #         So we don't need a second loop just for used_rrp.
        
        used_rrp = raw_rel_accum * scale
        
        # Update final states
        rrp_new = (rrp_refill - used_rrp)
        rrp_new = tl.where(rrp_new < 0.0, 0.0, rrp_new)
        rrp_new = tl.where(rrp_new > 1.0, 1.0, rrp_new)
        
        res_new = (res_mid + used_rrp)
        res_new = tl.where(res_new < 0.0, 0.0, res_new)
        res_new = tl.where(res_new > 1.0, 1.0, res_new)
        
        pr_new = (pr_mid - alpha_unprime * used_rrp)
        pr_new = tl.where(pr_new < 0.0, 0.0, pr_new)
        pr_new = tl.where(pr_new > 1.0, 1.0, pr_new)
        
        e_new = (e_mid - energy_cost_rel * used_rrp - energy_cost_pump * (1.0 - res_new))
        e_new = tl.where(e_new < 0.0, 0.0, e_new)
        e_new = tl.where(e_new > 1.6, 1.6, e_new)
        
        # Update state tensors in global memory
        tl.store(c_ptr + t * State_stride_t, c_next)
        tl.store(buf_ptr + t * State_stride_t, buf_next)
        tl.store(rrp_ptr + t * State_stride_t, rrp_new)
        tl.store(res_ptr + t * State_stride_t, res_new)
        tl.store(pr_ptr + t * State_stride_t, pr_new)
        tl.store(e_ptr + t * State_stride_t, e_new)
        
        # Prepare for next step
        c_curr = c_next
        buf_curr = buf_next
        rrp_curr = rrp_new
        res_curr = res_new
        pr_curr = pr_new
        e_curr = e_new
        
        # Compute Qamp
        qamp = _sigmoid(q_beta * (e_new - 0.5)) * qmax
        
        # Pass 3: Write Logits
        # logits += ln(release_frac * qamp + eps) - barrier
        for j_start in range(0, t + 1, BLOCK_SIZE_T):
            offsets = j_start + tl.arange(0, BLOCK_SIZE_T)
            mask = offsets <= t
            
            # Recompute raw_rel
            k_ptrs = k_seq_ptr + (offsets[:, None] * K_stride_t) + (q_offs[None, :] * K_stride_d)
            k_block = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
            dot = tl.sum(q_vec[None, :] * k_block, axis=1) / sqrt_D
            d_bilin = _sigmoid(dot)
            fuse_p = fuse_sig * d_bilin
            raw_rel = fuse_p * rrp_refill
            raw_rel = tl.where(raw_rel < 0.0, 0.0, raw_rel)
            raw_rel = tl.where(raw_rel > 1.0, 1.0, raw_rel)
            
            release_frac = raw_rel * scale
            
            # Barrier
            dist = tl.abs(t - offsets) / tl.maximum(T, 1.0)
            
            syn_logit = tl.log(release_frac * qamp + epsilon) - barrier_strength * dist
            
            # Add to existing logits
            # Logits is (B, H, T, T). We are at (t, offsets).
            # l_seq_ptr points to start of sequence (B, H).
            # We need to write to row t, cols offsets.
            # Stride t is Logits_stride_t. Stride k is Logits_stride_k.
            # NOTE: Logits_stride_k is the stride for the last dimension (T), which is usually 1.
            l_ptr = l_seq_ptr + t * Logits_stride_t + offsets * Logits_stride_k
            curr_l = tl.load(l_ptr, mask=mask, other=0.0)
            tl.store(l_ptr, curr_l + syn_logit, mask=mask)


def presyn_legacy_dense_step(q, k, logits, state, cfg):
    """
    Legacy dense-reference wrapper retained only until the jyb.1 backend disposition.

    This is not the live canonical equation and is intentionally not exported or dispatched.
    
    q: (B, H, T, D)
    k: (B, H, T, D)
    logits: (B, H, T, T)
    state: dict of tensors (B, H, T)
    """
    B, H, T, D = q.shape
    
    # Check constraints
    assert D in {16, 32, 64, 128}, "Head dim must be power of 2 <= 128 for optimal block load"
    
    # Flatten batch*head for the grid
    total_sequences = B * H
    
    # Reshape to (N_seq, T, ...) where N_seq = B*H
    # We need contiguous memory for the kernel to work with simple strides
    q_flat = q.reshape(total_sequences, T, D).contiguous()
    k_flat = k.reshape(total_sequences, T, D).contiguous()
    logits_flat = logits.reshape(total_sequences, T, T).contiguous()
    
    # Prepare state tensors (contiguous)
    state_tensors = {}
    for key in ["C", "BUF", "RRP", "RES", "PR", "CL", "E"]:
        state_tensors[key] = state[key].reshape(total_sequences, T).contiguous()
        
    N_seq = total_sequences
    
    # Grid: One block per sequence (N_seq)
    grid = (N_seq, )
    
    # Block size for T loops
    BLOCK_SIZE_T = 64 # Tuning parameter
    
    presyn_fused_kernel_forward[grid](
        q_flat, k_flat, logits_flat,
        state_tensors["C"], state_tensors["BUF"], state_tensors["RRP"],
        state_tensors["RES"], state_tensors["PR"], state_tensors["CL"], state_tensors["E"],
        # Strides (for flat inputs, stride_b is stride_seq)
        logits_flat.stride(0), 0, logits_flat.stride(1), logits_flat.stride(2), # stride_b, h(unused), t, k
        q_flat.stride(0), 0, q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), 0, k_flat.stride(1), k_flat.stride(2),
        state_tensors["C"].stride(0), 0, state_tensors["C"].stride(1),
        # Constants
        T=T, D=D, BLOCK_SIZE_T=cast(Any, BLOCK_SIZE_T),
        tau_c=cfg.tau_c, tau_buf=cfg.tau_buf, tau_prime=cfg.tau_prime,
        tau_rrp=cfg.tau_rrp, tau_energy=cfg.tau_energy,
        alpha_ca=cfg.alpha_ca, alpha_buf_on=cfg.alpha_buf_on,
        alpha_buf_off=cfg.alpha_buf_off, alpha_prime=cfg.alpha_prime,
        alpha_refill=cfg.alpha_refill, alpha_unprime=cfg.alpha_unprime,
        energy_in=cfg.energy_in, energy_cost_rel=cfg.energy_cost_rel,
        energy_cost_pump=cfg.energy_cost_pump,
        syt_fast_kd=cfg.syt_fast_kd, syt_slow_kd=cfg.syt_slow_kd,
        complexin_bias=cfg.complexin_bias,
        qmax=cfg.qmax, q_beta=cfg.q_beta,
        barrier_strength=cfg.barrier_strength, epsilon=cfg.epsilon,
        sqrt_D=math.sqrt(D)
    )
    
    # Copy back state (updates were in-place on contiguous copies)
    # We need to copy back to original if it wasn't contiguous.
    # If original was contiguous, view() might share memory.
    # If view() copied, we need to copy back.
    # To be safe:
    for key in ["C", "BUF", "RRP", "RES", "PR", "CL", "E"]:
        state[key].copy_(state_tensors[key].view(B, H, T))
        
    # Logits updated in place on flat view. If view shared mem, logits updated.
    # If view copied, we need copy back.
    if logits_flat.data_ptr() != logits.data_ptr():
        logits.copy_(logits_flat.view(B, H, T, T))
        
    return logits, state

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
