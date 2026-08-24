# Hugging Face bio adapters

`bio_inspired_nanochat.hf_bio_adapter` injects the project’s synaptic dynamics into the
feed-forward projections of pretrained Hugging Face transformers. It operates at a common
architectural seam instead of reimplementing whole model families:

- PyTorch `nn.Linear` projections used by Llama/Mistral, OPT, BERT-like, and related models.
- Hugging Face `Conv1D` projections used by GPT-2.
- Explicit module-name globs for architectures whose feed-forward names are not in the safe
  defaults.

The default matcher only selects feed-forward/MLP paths. It does not rewrite attention
projections, embeddings, or the language-model head. If nothing matches, injection raises with a
sample of available affine module names; it never silently returns an unmodified model.

## Inject and fine-tune

```python
from transformers import AutoModelForCausalLM

from bio_inspired_nanochat.hf_bio_adapter import (
    bio_adapter_parameters,
    inject_bio_adapters,
)
from bio_inspired_nanochat.synaptic import SynapticConfig

model = AutoModelForCausalLM.from_pretrained("your-local-checkpoint")
report = inject_bio_adapters(
    model,
    SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
        enable_metabolism=True,
    ),
)
optimizer_parameters = bio_adapter_parameters(model)
```

Injection copies every source weight and bias into `SynapticLinear.w_slow`, zeros the fast-weight
path, and zeros the output side of the low-rank postsynaptic residual. The injected model therefore
starts numerically equivalent to the source model (within normal floating-point kernel rounding),
while the non-zero complementary low-rank factor preserves a live gradient path.

During an adapting forward, each projection advances bounded calcium and energy summaries from its
activation stream. Those signals gate the existing `SynapticLinear` fast-weight path. Eligibility,
CaMKII/PP1, BDNF, and fast/slow writes use the canonical project implementation rather than an
adapter-specific approximation. The `enable_presyn`, `enable_hebbian`, and `enable_metabolism`
fields are the mechanism toggles.

For an unusual architecture, pass exact names or shell-style module globs:

```python
inject_bio_adapters(
    model,
    target_patterns=("encoder.blocks.*.custom_ff_in", "encoder.blocks.*.custom_ff_out"),
)
```

Inspect live dynamics with `bio_adapter_metrics(model)`. Pause online state and weight updates with
`set_bio_adaptation(model, False)`; this does not discard learned adapter weights. At a sequence
boundary, `reset_bio_adapters(model)` clears transient traces and calcium/energy state while
preserving pretrained and consolidated slow weights.

## Portable bundles

Adapter-only fine-tunes can be packaged without copying the base checkpoint:

```python
from bio_inspired_nanochat.hf_bio_adapter import load_bio_adapter, save_bio_adapter

save_bio_adapter(model, "artifacts/my-bio-adapter")

fresh_model = AutoModelForCausalLM.from_pretrained("your-local-checkpoint")
load_bio_adapter(
    fresh_model,
    "artifacts/my-bio-adapter",
    adaptation_enabled=False,
)
```

The directory contains `bio_adapter.json` and `bio_adapter.safetensors`. Loading checks the base
model type, exact adapter topology, schema version, and tensor key set before accepting the bundle.
Use the same base checkpoint revision that produced the bundle; the manifest identifies the model
type but intentionally does not pretend to verify an arbitrary remote checkpoint’s content hash.

## External checkpoint evidence

The CPU-friendly E2E downloads a pinned public `sshleifer/tiny-gpt2` revision with the mandated
user-agent string, then performs all Transformers loads in local-only mode:

```bash
uv run python -m scripts.e2e.hf_bio_adapter
```

It verifies source/injected logit equivalence, runs a short adapter-only fine-tune, requires
non-zero calcium, eligibility, and fast-weight telemetry, packages the result, and reloads it into
a fresh copy of the external base checkpoint. The strict report is written to
`results/hf_bio_adapter.json`.

Focused local validation:

```bash
uv run python -m pytest tests/test_hf_bio_adapter.py -v
```
