"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

from bio_inspired_nanochat.torch_imports import torch, F
import ast
import math
import numbers
import operator
import signal
import inspect
from contextlib import contextmanager
from collections import deque
from bio_inspired_nanochat.common import (
    autodetect_device_type,
    compute_init,
    stable_seed_u64,
)
from bio_inspired_nanochat.checkpoint_manager import load_model
from contextlib import nullcontext

# -----------------------------------------------------------------------------


def batch_slices(total_size: int, max_batch_size: int) -> tuple[tuple[int, int], ...]:
    """Partition ``range(total_size)`` into exact, non-empty bounded slices."""
    if isinstance(total_size, bool) or not isinstance(total_size, int) or total_size < 1:
        raise ValueError("total_size must be a positive integer")
    if (
        isinstance(max_batch_size, bool)
        or not isinstance(max_batch_size, int)
        or max_batch_size < 1
    ):
        raise ValueError("max_batch_size must be a positive integer")
    return tuple(
        (start, min(start + max_batch_size, total_size))
        for start in range(0, total_size, max_batch_size)
    )


# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"'{formula}': timed out after {duration} seconds")

    prev_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_calc_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_calc_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str)):
            return node.value
        raise ValueError("Unsupported literal in calculator expression")

    if isinstance(node, ast.UnaryOp):
        op_fn = _ALLOWED_UNARYOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError("Unsupported unary operator in calculator expression")
        val = _safe_calc_eval_node(node.operand)
        if not isinstance(val, (int, float)):
            raise ValueError("Unary operators only apply to numbers")
        return op_fn(val)

    if isinstance(node, ast.BinOp):
        op_fn = _ALLOWED_BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError("Unsupported binary operator in calculator expression")
        left = _safe_calc_eval_node(node.left)
        right = _safe_calc_eval_node(node.right)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("Binary operators only apply to numbers")
        return op_fn(left, right)

    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("Keyword args are not allowed in calculator calls")
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "count":
            raise ValueError("Only str.count(...) is allowed in calculator calls")
        obj = _safe_calc_eval_node(node.func.value)
        if not isinstance(obj, str):
            raise ValueError("count() is only allowed on string literals")
        if len(node.args) != 1:
            raise ValueError("count() requires exactly one argument")
        needle = _safe_calc_eval_node(node.args[0])
        if not isinstance(needle, str):
            raise ValueError("count() argument must be a string literal")
        return obj.count(needle)

    raise ValueError(f"Unsupported syntax in calculator expression: {type(node).__name__}")


def _safe_calc_eval(expr: str):
    parsed = ast.parse(expr, mode="eval")
    return _safe_calc_eval_node(parsed)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return _safe_calc_eval(formula)
    except Exception:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Safe calculator evaluator for tool-usage.

    Allowed:
    - Numeric literals and arithmetic: +, -, *, /, //, %, parentheses, unary +/-.
    - Literal string count: \"some string\".count(\"sub\") (no variables, no keywords).
    """
    expr = expr.replace(",", "").strip()
    if not expr:
        return None
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.presyn_state = None # Biological state (per-layer if synaptic model)
        self.pos = 0 # current position in time in the cache

    def reset(self):
        self.pos = 0
        self.presyn_state = None

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        # 1) validate the shapes
        if self.kv_cache is not None:
            raise ValueError("cannot prefill a non-empty KV cache")
        if other.kv_cache is None:
            raise ValueError("cannot prefill from an uninitialized KV cache")
        if not 0 <= other.pos <= other.kv_cache.size(4):
            raise ValueError(
                f"source cache position {other.pos} is outside its sequence capacity"
            )
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            # ix 0: num_layers, 1: k/v, 2: batch_size, 3: num_heads, 4: seq_len, 5: head_dim
            if ix in [0, 1, 3, 5]:
                # num_layers, k/v, num_heads, head_dim must match
                if dim1 != dim2:
                    raise ValueError(f"cache dimension {ix} mismatch: {dim1} != {dim2}")
            elif ix == 2:
                # batch_size can be expanded
                if dim1 != dim2 and dim2 != 1:
                    raise ValueError(f"cache batch mismatch: {dim1} != {dim2}")
            elif ix == 4:
                # seq_len: self must be longer than other
                if dim1 < dim2:
                    raise ValueError(f"cache sequence capacity mismatch: {dim1} < {dim2}")
        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, : other.pos, :] = other.kv_cache[
            :, :, :, :, : other.pos, :
        ]
        # 4) update the pos
        self.pos = other.pos
        # 5) copy presyn_state if exists
        if other.presyn_state is not None:
            target_B = self.kv_shape[2]
            src_B = other.kv_shape[2] # Assuming other.kv_shape is correct

            def clone_presyn_state(state_dict):
                if state_dict is None:
                    return None
                new_state = {}
                for key, value in state_dict.items():
                    if isinstance(value, list):
                        # delay queue is a list of tensors
                        new_queue = []
                        for item in value:
                            if isinstance(item, torch.Tensor):
                                if src_B == 1 and target_B > 1:
                                    # Expand batch dim
                                    expanded = item.expand(
                                        target_B, *item.shape[1:]
                                    ).clone()
                                    new_queue.append(expanded)
                                else:
                                    new_queue.append(item.clone())
                            else:
                                new_queue.append(item)
                        new_state[key] = new_queue
                    elif isinstance(value, torch.Tensor):
                        if src_B == 1 and target_B > 1:
                            new_state[key] = value.expand(
                                target_B, *value.shape[1:]
                            ).clone()
                        else:
                            new_state[key] = value.clone()
                    else:
                        new_state[key] = value
                return new_state

            if isinstance(other.presyn_state, list):
                self.presyn_state = [clone_presyn_state(st) for st in other.presyn_state]
            elif isinstance(other.presyn_state, dict):
                self.presyn_state = clone_presyn_state(other.presyn_state)
            else:
                self.presyn_state = None

    def fork_batch(self, repeats: int):
        """Clone each cache row `repeats` times for isolated candidate continuations.

        Candidate branches must never advance or mutate the committed decode cache. Rows are
        repeat-interleaved so flattened `(batch, candidate)` token ids align with both KV tensors and
        every tensor in the per-layer presynaptic state.
        """
        if isinstance(repeats, bool) or not isinstance(repeats, int):
            raise TypeError(f"repeats must be a positive integer, got {repeats!r}")
        if repeats < 1:
            raise ValueError(f"repeats must be a positive integer, got {repeats!r}")
        if self.kv_cache is None:
            raise ValueError("cannot fork an uninitialized KV cache")

        source_batch = int(self.kv_cache.size(2))
        forked = KVCache(
            batch_size=source_batch * repeats,
            num_heads=int(self.kv_cache.size(3)),
            seq_len=self.pos + 1,
            head_dim=int(self.kv_cache.size(5)),
            num_layers=int(self.kv_cache.size(0)),
        )
        forked.kv_cache = torch.empty(
            forked.kv_shape,
            dtype=self.kv_cache.dtype,
            device=self.kv_cache.device,
        )
        forked.kv_cache[:, :, :, :, : self.pos, :] = self.kv_cache[
            :, :, :, :, : self.pos, :
        ].repeat_interleave(repeats, dim=2)
        forked.pos = self.pos

        def repeat_state(state):
            if state is None:
                return None
            repeated = {}
            for key, value in state.items():
                if isinstance(value, list):
                    repeated[key] = [
                        (
                            item.repeat_interleave(repeats, dim=0).clone()
                            if isinstance(item, torch.Tensor)
                            and item.ndim > 0
                            and item.shape[0] == source_batch
                            else item.clone() if isinstance(item, torch.Tensor) else item
                        )
                        for item in value
                    ]
                elif (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == source_batch
                ):
                    repeated[key] = value.repeat_interleave(repeats, dim=0).clone()
                elif isinstance(value, torch.Tensor):
                    repeated[key] = value.clone()
                else:
                    repeated[key] = value
            return repeated

        if isinstance(self.presyn_state, list):
            forked.presyn_state = [repeat_state(state) for state in self.presyn_state]
        elif isinstance(self.presyn_state, dict):
            forked.presyn_state = repeat_state(self.presyn_state)
        return forked

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # as much as we need plus buffer of 1024
            t_needed = (t_needed + 1023) & ~1023 # then round up to the nearest multiple of 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.inference_mode()
def predictive_entropy_nats(logits):
    """Shannon entropy (nats) of softmax(logits) per row: the uncertainty selective decoding acts on.

    On the synaptic model this is the deterministic-release predictive entropy; the committed
    calibration artifact found it indistinguishable from MC-vesicle entropy at toy scale
    (delta ECE 4e-6), so one forward is used rather than an MC ensemble.
    """
    log_p = torch.log_softmax(logits.float(), dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1)


def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError(
            f"logits must have shape (batch, vocabulary), got {tuple(logits.shape)}"
        )
    if not logits.dtype.is_floating_point:
        raise ValueError("logits must use a floating dtype")
    invalid_values = torch.isnan(logits) | torch.isposinf(logits)
    if bool(invalid_values.any()) or not bool(torch.isfinite(logits).any(dim=-1).all()):
        raise ValueError(
            "each logits row must contain a finite candidate and may use only -inf for masking"
        )
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, numbers.Real)
        or not math.isfinite(float(temperature))
        or temperature < 0.0
    ):
        raise ValueError("temperature must be finite and non-negative")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0
    ):
        raise ValueError("top_k must be a non-negative integer or None")
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use
        try:
            self._supports_train_mode = (
                "train_mode" in inspect.signature(self.model.forward).parameters
            )
        except (TypeError, ValueError):
            self._supports_train_mode = False

    def _forward(self, ids, kv_cache):
        if self._supports_train_mode:
            return self.model.forward(ids, kv_cache=kv_cache, train_mode=False)
        return self.model.forward(ids, kv_cache=kv_cache)

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42,
                 yield_metrics=False, deliberation=None, self_correction=None, selective=None):
        """Same as generate, but does single prefill and then clones the KV cache.

        ``deliberation`` (bead r00r.1.2): an optional ``DeliberationConfig`` (or a prebuilt
        ``DeliberationController``) that turns on free-energy "ponder". Per token, bounded candidate
        continuations are advanced on isolated cache branches; their relaxed free energies enter the
        actual decode logits, while effort/confidence modulate temperature. Default ``None`` and a
        zero-step budget preserve the single-step baseline exactly.

        ``self_correction`` (beads re4e.1, re4e.1.2): an optional ``SelfCorrectionConfig``,
        ``SelfCorrectingGenerator``, or boolean flag that turns on the closed-loop
        detect -> deliberate -> regenerate -> recheck loop. Default ``None``/``False`` preserves
        the baseline streaming fastpath with zero overhead.
        """
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens)
        ):
            raise ValueError("tokens must be a non-empty list of integer token IDs")
        vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", None)
        if isinstance(vocab_size, int) and any(
            token < 0 or token >= vocab_size for token in tokens
        ):
            raise ValueError(f"tokens must be inside model vocabulary [0, {vocab_size})")
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 0
        ):
            raise ValueError("max_tokens must be a non-negative integer or None")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, numbers.Real)
            or not math.isfinite(float(temperature))
            or temperature < 0.0
        ):
            raise ValueError("temperature must be finite and non-negative")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer or None")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # A new prompt is a new scratchpad: synaptic plasticity runs during inference (the
        # consolidation/eligibility update is gated on no_grad, which inference_mode satisfies), so
        # without this reset one generation's fast adaptation leaks into the next (e.g. across chat
        # turns). Defaults clear the per-sequence eligibility traces + CaMKII/PP1/BDNF gate while
        # preserving the backprop-trained slow/fast weights. See GPTSynaptic.reset_sequence_state.
        if hasattr(self.model, "reset_sequence_state"):
            self.model.reset_sequence_state()

        # Build self-correcting generator if requested (bead re4e.1.2)
        self_correcting_gen = None
        if self_correction is not None and self_correction is not False:
            from bio_inspired_nanochat.self_correcting_generator import (
                SelfCorrectionConfig,
                SelfCorrectingGenerator,
            )
            if isinstance(self_correction, SelfCorrectingGenerator):
                self_correcting_gen = self_correction
            elif isinstance(self_correction, SelfCorrectionConfig):
                if self_correction.enabled:
                    self_correcting_gen = SelfCorrectingGenerator(self.model, cfg=self_correction)
            elif isinstance(self_correction, bool) and self_correction:
                self_correcting_gen = SelfCorrectingGenerator(
                    self.model, cfg=SelfCorrectionConfig(enabled=True)
                )
            elif isinstance(self_correction, dict):
                cfg = SelfCorrectionConfig(**self_correction)
                if cfg.enabled:
                    self_correcting_gen = SelfCorrectingGenerator(self.model, cfg=cfg)
            else:
                raise TypeError(
                    "self_correction must be a SelfCorrectionConfig, "
                    "SelfCorrectingGenerator, bool, mapping, or None"
                )

        if self_correcting_gen is not None and self_correcting_gen.cfg.enabled:
            if num_samples != 1:
                raise ValueError(
                    "self-correcting generation currently supports exactly one sample; "
                    "duplicating one repaired trajectory would not produce independent samples"
                )
            # Run closed-loop self-correction over the prompt
            prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            max_new = max_tokens if max_tokens is not None else 64
            assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
            bos = self.tokenizer.get_bos_token_id()
            if self_correcting_gen.cfg.abstain_token_id in (assistant_end, bos):
                raise ValueError(
                    "abstain_token_id must not equal a terminal assistant or BOS token"
                )
            traj = self_correcting_gen.generate(
                prompt=prompt_tensor,
                max_new_tokens=max_new,
                temperature=temperature,
                top_k=0 if top_k is None else top_k,
                rng=rng,
            )
            generated_only = traj.final_tokens[len(tokens):]
            for tok in generated_only:
                token_column = [tok] * num_samples
                token_masks = [0 if traj.is_abstention else 1] * num_samples
                if yield_metrics:
                    metrics = {
                        "self_correction": {
                            "outcome": traj.outcome.value,
                            "attempts_used": traj.attempts_used,
                            "is_abstention": traj.is_abstention,
                            "wall_time_ms": traj.total_wall_time_ms,
                        }
                    }
                    yield token_column, token_masks, metrics
                else:
                    yield token_column, token_masks
                if tok == assistant_end or tok == bos:
                    break
            return

        # wmel.1: selective decoding. ``selective`` is an UncertaintyDecodingConfig (or True / a dict
        # of its fields). When the predictive entropy of a step exceeds
        # ``max_predictive_entropy_nats`` the policy refuses to emit that token: the row ends with
        # <|assistant_end|> and the event (row, action, entropy, threshold) is reported in the
        # metrics of that step (``yield_metrics=True``). ``None``/``False`` leaves generation
        # byte-identical. Not combined with self_correction (that path returns above).
        selective_cfg = None
        if selective is not None and selective is not False:
            from bio_inspired_nanochat.adaptive_compute import UncertaintyDecodingConfig

            if isinstance(selective, UncertaintyDecodingConfig):
                selective_cfg = selective if selective.enabled else None
            elif isinstance(selective, bool):
                selective_cfg = UncertaintyDecodingConfig(enabled=True)
            elif isinstance(selective, dict):
                selective_cfg = UncertaintyDecodingConfig(**{"enabled": True, **selective})
            else:
                raise TypeError(
                    "selective must be an UncertaintyDecodingConfig, True, or a dict of its fields, "
                    f"got {type(selective).__name__}"
                )
        # Build the deliberation controller (None ⟹ baseline single-step decode). Lazy import keeps
        # the metriplectic/numpy stack off engine.py's import path unless deliberation is requested.
        deliberation_controller = None
        if deliberation is not None:
            from bio_inspired_nanochat.deliberation import DeliberationConfig, make_controller
            deliberation_controller = (
                make_controller(deliberation) if isinstance(deliberation, DeliberationConfig) else deliberation
            )

        def _decode_temperature(kv_cache_for_state, candidate_scores=None):
            """Per-token effective decode temperature; identity (= ``temperature``) without deliberation."""
            if deliberation_controller is None:
                return temperature
            return deliberation_controller.effective_temperature(
                getattr(kv_cache_for_state, "presyn_state", None),
                temperature,
                candidate_scores=candidate_scores,
            )

        def _deliberated_logits(step_logits, kv_cache_for_state):
            """Score bounded candidate continuations by relaxed free energy on isolated branches."""
            if (
                deliberation_controller is None
                or deliberation_controller.cfg.max_iters == 0
                or getattr(kv_cache_for_state, "presyn_state", None) is None
            ):
                return step_logits, None
            candidate_k = min(
                deliberation_controller.cfg.candidate_top_k,
                int(step_logits.shape[-1]),
            )
            if top_k is not None and top_k > 0:
                candidate_k = min(candidate_k, top_k)
            candidate_ids = torch.topk(step_logits, candidate_k, dim=-1).indices
            candidate_cache = kv_cache_for_state.fork_batch(candidate_k)
            packed_ids = candidate_ids.reshape(-1, 1)
            self._forward(packed_ids, candidate_cache)
            candidate_scores = deliberation_controller.relax_candidate_states(
                candidate_cache.presyn_state,
                candidate_shape=tuple(candidate_ids.shape),
            )
            if candidate_scores is None:
                return step_logits, None
            return (
                deliberation_controller.candidate_energy_logits(
                    step_logits,
                    candidate_ids,
                    candidate_scores,
                ),
                candidate_scores,
            )

        # Get the special tokens we need to coordinate the tool use state machine
        def get_special(s):
            return self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        result = self._forward(ids, kv_cache_prefill)
        # Handle both GPT (returns logits) and GPTSynaptic (returns (logits, None))
        if isinstance(result, tuple):
            logits, _ = result
        else:
            logits = result
        logits = logits[:, -1, :]
        # The prefill logits choose the first generated token, so they must use the same
        # deliberation hook as every subsequent decode step.  Do not ponder for a zero-token
        # request: no token will consume the result and the controller trace must stay empty.
        prefill_temperature = temperature
        prefill_candidate_scores = None
        if max_tokens is None or max_tokens > 0:
            logits, prefill_candidate_scores = _deliberated_logits(logits, kv_cache_prefill)
            prefill_temperature = _decode_temperature(
                kv_cache_prefill,
                prefill_candidate_scores,
            )
        next_ids = sample_next_token(logits, rng, prefill_temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Get sampled tokens - either from prefill or from forward pass
            if first_iteration:
                # Use the tokens we already sampled from prefill
                if num_samples == 1:
                    sampled_tokens = [sampled_tokens[0]]
                else:
                    # Sample independently per row from the same prefill logits
                    logits_rep = logits.expand(num_samples, -1).contiguous()
                    next_ids = sample_next_token(logits_rep, rng, prefill_temperature, top_k)
                    sampled_tokens = next_ids[:, 0].tolist()
                first_iteration = False
            else:
                # Forward the model and get the next token for each row
                result = self._forward(ids, kv_cache_decode)  # (B, T, vocab_size) or (logits, None)
                # Handle both GPT (returns logits) and GPTSynaptic (returns (logits, None))
                if isinstance(result, tuple):
                    logits, _ = result
                else:
                    logits = result
                logits = logits[:, -1, :]  # (B, vocab_size) at last time step
                # r00r.1.6: candidate-specific relaxed free energy is an additive energy term in
                # the actual decode logits. The branch cache is isolated; max_iters=0 is identity.
                logits, candidate_scores = _deliberated_logits(logits, kv_cache_decode)
                next_ids = sample_next_token(
                    logits,
                    rng,
                    _decode_temperature(kv_cache_decode, candidate_scores),
                    top_k,
                )  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()

            # wmel.1: predictive entropy of the logits that produced this step's samples.
            selective_actions = None
            selective_events = []
            if selective_cfg is not None:
                ent = predictive_entropy_nats(logits)
                if ent.numel() == 1 and num_samples > 1:
                    ent = ent.expand(num_samples)
                selective_actions = [
                    (float(ent[i]), bool(float(ent[i]) > selective_cfg.max_predictive_entropy_nats))
                    for i in range(num_samples)
                ]
            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                if selective_actions is not None and not is_forced and selective_actions[i][1]:
                    # The policy refuses a token this uncertain: end the row and report why.
                    next_token = assistant_end
                    token_masks[-1] = 0
                    selective_events.append({
                        "row": i,
                        "action": selective_cfg.terminal_action,
                        "entropy_nats": selective_actions[i][0],
                        "threshold_nats": selective_cfg.max_predictive_entropy_nats,
                    })
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            if yield_metrics:
                # Extract metrics
                metrics = {}
                
                # 1. MoE Stats (Gates, Indices)
                # We scan layers for SynapticMoE
                moe_stats = []
                if hasattr(self.model, 'h'):
                    for i, block in enumerate(self.model.h):
                        if hasattr(block, 'mlp') and hasattr(block.mlp, 'last_ctx'):
                            ctx = block.mlp.last_ctx
                            if ctx:
                                # ctx has 'gates', 'indices', 'x'
                                # We want gates (B, 1, K) probably?
                                # Note: in generate, we process 1 token step.
                                # gates shape is (B, 1, K) typically.
                                # We convert to simple list
                                gates = ctx.get('gates')
                                indices = ctx.get('indices')
                                if gates is not None and indices is not None:
                                    # Just take the first batch item for visualization simplicity if num_samples > 1?
                                    # Or return all. Let's return list of lists.
                                    moe_stats.append({
                                        "layer": i,
                                        "gates": gates.float().cpu().numpy().tolist(),
                                        "indices": indices.cpu().numpy().tolist()
                                    })
                metrics['moe'] = moe_stats
                if selective_cfg is not None:
                    metrics['selective'] = {
                        "entropy_nats": [a[0] for a in (selective_actions or [])],
                        "threshold_nats": selective_cfg.max_predictive_entropy_nats,
                        "events": selective_events,
                    }
                
                # 2. Presynaptic Stats (RRP, C)
                presyn_state = kv_cache_decode.presyn_state
                if isinstance(presyn_state, list):
                    presyn_state = presyn_state[-1] if presyn_state else None
                if presyn_state:
                    presyn_stats = {}
                    # Map friendly metric keys to presynaptic state keys
                    state_key_map = {
                        "c": "C",
                        "rrp": "RRP",
                        "sn": "PR",
                        "cl": "CL",
                        "en": "E",
                        "amp": "AMP",
                        "buf": "BUF",
                    }
                    for out_key, state_key in state_key_map.items():
                        if state_key in presyn_state:
                            t = presyn_state[state_key]  # (B, H, T)
                            if t.shape[-1] > 0:
                                last_val = t[..., -1]  # (B, H)
                                presyn_stats[out_key] = (
                                    last_val.float().cpu().numpy().tolist()
                                )
                    metrics["presyn"] = presyn_stats

                # 3. Postsynaptic Memory Stats (CaMKII - Long Term Potentiation)
                # This represents "how much the model is learning/writing to memory" at this step
                memory_stats = []
                if hasattr(self.model, 'h'):
                    for block in self.model.h:
                        layer_camkii = []
                        # Check MLP
                        if hasattr(block, 'mlp'):
                            # Case 1: MoE
                            if hasattr(block.mlp, 'experts'):
                                for exp in block.mlp.experts:
                                    # Check fc1 and fc2
                                    if hasattr(exp, 'fc1') and hasattr(exp.fc1, 'post') and hasattr(exp.fc1.post, 'camkii'):
                                        layer_camkii.append(exp.fc1.post.camkii.mean().item())
                                    if hasattr(exp, 'fc2') and hasattr(exp.fc2, 'post') and hasattr(exp.fc2.post, 'camkii'):
                                        layer_camkii.append(exp.fc2.post.camkii.mean().item())
                            # Case 2: Standard Synaptic MLP
                            elif hasattr(block.mlp, 'fc') and hasattr(block.mlp.fc, 'post') and hasattr(block.mlp.fc.post, 'camkii'):
                                layer_camkii.append(block.mlp.fc.post.camkii.mean().item())
                                if hasattr(block.mlp, 'proj') and hasattr(block.mlp.proj, 'post') and hasattr(block.mlp.proj.post, 'camkii'):
                                    layer_camkii.append(block.mlp.proj.post.camkii.mean().item())
                        
                        if layer_camkii:
                            memory_stats.append(sum(layer_camkii) / len(layer_camkii))
                        else:
                            memory_stats.append(0.0)
                metrics['memory'] = memory_stats

                yield token_column, token_masks, metrics
            else:
                yield token_column, token_masks
            
            num_generated += 1
            # Prepare ids for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, *, max_batch_size=None, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.

        ``max_batch_size`` bounds peak decode batch size without dropping a
        remainder. Each chunk receives a deterministic derived seed, so identical
        inputs/configuration reproduce across Python processes and DDP ranks.
        """
        max_batch_size = num_samples if max_batch_size is None else max_batch_size
        slices = batch_slices(num_samples, max_batch_size)
        if len(slices) > 1:
            base_seed = kwargs.pop("seed", 42)
            if isinstance(base_seed, bool) or not isinstance(base_seed, int):
                raise ValueError("seed must be an integer")
            all_results = []
            all_masks = []
            for chunk_index, (start, end) in enumerate(slices):
                chunk_seed = stable_seed_u64(
                    base_seed,
                    salt=f"engine.generate_batch.chunk:{chunk_index}",
                ) % (2**63)
                chunk_results, chunk_masks = self.generate_batch(
                    tokens,
                    num_samples=end - start,
                    seed=chunk_seed,
                    **kwargs,
                )
                all_results.extend(chunk_results)
                all_masks.extend(chunk_masks)
            return all_results, all_masks

        # Strip yield_metrics from kwargs since generate_batch doesn't use metrics
        # and generate() yields 3 values when yield_metrics=True which would cause
        # unpacking to fail below.
        kwargs.pop("yield_metrics", None)
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    device_type = autodetect_device_type()
    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    if device_type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    if device_type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_masks in stream:
            token = token_column[0] # only print out the first row
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
