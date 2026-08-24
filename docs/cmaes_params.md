# CMA-ES parameter loading in training (`load_cmaes_params`)

Bead `c2l`. Lets a training run start from the optimum found by the CMA-ES search
(`scripts/tune_bio_params` writes `best_params.json`), instead of hand-editing
`SynapticConfig` defaults.

## Usage

```bash
uv run python -m scripts.base_train \
    synapses=1 load_cmaes_params=runs/cmaes/top10/best_params.json \
    depth=4 ...
```

`load_cmaes_params` is an ordinary configurator setting (any `key=value` pair on the
command line works; see `bio_inspired_nanochat/configurator.py`). It must point to a JSON
file and requires `synapses=1`.

## File format

A flat JSON object mapping **`SynapticConfig` field names** to numbers:

```json
{
  "tau_rrp": 35.0,
  "camkii_up": 0.08,
  "lambda_loge": 0.9
}
```

Validation rules (`bio_inspired_nanochat/cmaes_params.py`):

| Rule | On violation |
| --- | --- |
| Keys must be real `SynapticConfig` field names | `ValueError` listing up to 5 closest valid names |
| Values must be numbers | `ValueError`; booleans are rejected deliberately |
| Top level must be a JSON object | `ValueError` naming the actual type found |
| File must be readable, valid JSON, non-empty | `ValueError` with file/line detail |

Nothing is silently skipped: a typo'd key would otherwise train with default kinetics while
looking tuned.

## Semantics

- The overlay happens AFTER `SynapticConfig` construction and BEFORE model build, so the full
  resulting config flows into checkpoint provenance (`synaptic_config_to_meta`) exactly like a
  hand-edited config.
- **Resume refuses the flag**: overlaying search parameters onto a resumed checkpoint would
  desync the model from its saved config; start a fresh run instead.
- The overlay does NOT touch non-synaptic settings (depth, lr, ...); combine with ordinary
  configurator overrides for those.

Programmatic use (e.g. the joint 48D search, bead scq):

```python
from bio_inspired_nanochat.cmaes_params import apply_cmaes_params

syn_cfg = apply_cmaes_params(SynapticConfig(), "best_params.json")
```
