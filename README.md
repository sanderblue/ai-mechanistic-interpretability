# interp — a small mechanistic-interpretability lab

[![ci](https://github.com/sanderblue/ai-mechanistic-interpretability/actions/workflows/ci.yml/badge.svg)](https://github.com/sanderblue/ai-mechanistic-interpretability/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Two foundational techniques for looking inside a transformer — the **logit lens** and
**causal tracing** (activation patching) — built on one model-adapter interface so the
*same* experiment runs on **GPT-2** (via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens))
and **Qwen3-0.6B** (via [nnsight](https://nnsight.net)), fast enough to iterate on a laptop.

<table>
<tr>
<td><img src="docs/figures/logit_lens_gpt2.png" width="100%"></td>
<td><img src="docs/figures/causal_trace_qwen3.png" width="100%"></td>
</tr>
<tr>
<td align="center"><em>GPT-2: P(" oxygen") crystallises in the upper-mid layers.</em></td>
<td align="center"><em>Qwen3: the France→Paris fact lives on the subject token, then hands off to the final token.</em></td>
</tr>
</table>

> All four figures with interpretation: **[docs/results.md](docs/results.md)** ·
> the techniques and backend engineering notes: **[docs/methodology.md](docs/methodology.md)**

## Contents

- [What it does](#what-it-does)
- [Install](#install)
- [Usage](#usage)
- [Architecture](#architecture)
- [Extending it](#extending-it)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Notes & limitations](#notes--limitations)
- [References](#references)

## What it does

| Technique | Question it answers | Output |
| --- | --- | --- |
| **Logit lens** | At each layer, what token is the model "currently betting on"? | a probability-vs-depth curve |
| **Causal tracing** | Where in the network does a specific fact live? | a [layer × token] recovery heatmap |

Both run on small models (`gpt2`, `qwen3-0.6b`), are driven by YAML configs, and write a
figure plus a reproducible JSON result per run.

This is the "learn the method cheaply" stage of a larger project on **confabulation** —
when a model fabricates a fact it was never given, what differs internally between a fact
it *has* and one it *lacks*? The repo owns the tools on small models first, built to scale
to the larger one later (see [Roadmap](#roadmap)).

## Install

Requires Python ≥ 3.10. Backends are optional extras — install only what you need.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tl,nnsight,dev]"     # or just ".[tl]" / ".[nnsight]"
```

Runs on CPU, Apple-silicon MPS, or CUDA — the device is auto-detected (`cuda → mps → cpu`).

## Usage

A run is fully described by its config file:

```bash
python -m interp.run <experiment> --config <config.yaml> [--model ...] [--device ...] [--tag ...]
```

```bash
make e01   # logit lens on GPT-2          (downloads ~500MB the first time)
make e02   # causal tracing on GPT-2
python -m interp.run logit_lens   --config configs/logit_lens_qwen3.yaml   # same, on Qwen3
python -m interp.run causal_tracing --config configs/causal_tracing_qwen3.yaml
```

Outputs land in `outputs/<experiment>/<tag>/` as `result.json`, the resolved
`config.yaml`, and a `.png` figure — with captured provenance (device, dtype, seed,
library versions, git SHA). You can also call the library directly:

```python
from interp import load_model, logit_lens

model = load_model("qwen3-0.6b")          # or "gpt2"
result = logit_lens(model, "Water is made of hydrogen and", " oxygen")
print(result.crossover_layer)             # layer where " oxygen" becomes top-1
```

## Architecture

Three layers, each independent of the others:

```
 experiments/      logit_lens · causal_tracing        (registered plug-ins)
       │ uses
 interp core       lenses · patching · metrics · viz  (backend-agnostic logic)
       │ via
 ModelAdapter      run_with_cache · forward · unembed (one interface)
       ├── TransformerLensAdapter   →  GPT-2
       └── NNsightAdapter           →  Qwen3 (and, later, larger models)
```

The **adapter** is the load-bearing idea: experiment code speaks in abstract *sites*
("the residual stream after layer 7") and never touches a backend- or
architecture-specific hook name. An [integration test](tests/test_adapter_consistency.py)
loads GPT-2 under *both* backends and checks they produce the same next-token
distribution (**KL ≈ 0**), so results don't depend on which backend produced them. The
fiddly parts of making that true on each stack are written up in
[docs/methodology.md](docs/methodology.md#3-backend-engineering-notes).

## Extending it

The common cases are one-file changes:

| To add… | Do this |
| --- | --- |
| a new **experiment** | drop a `@register_experiment("name")` class in `experiments/`, add a config |
| a new **model** | add one line to `MODEL_REGISTRY` in [`interp/models/__init__.py`](interp/models/__init__.py) |
| a new **architecture** | add a `Layout` in [`interp/models/layouts.py`](interp/models/layouts.py) |
| a new **hook site** | add a `Site` value + its mapping in each adapter |

## Testing

```bash
make test              # fast, hermetic unit tests (no model downloads) — this is CI
make test-integration  # model-backed tests on both backends (downloads weights)
make lint              # ruff check + format
```

CI runs only the unit tests and lint, with neither backend installed — the green badge
reflects code correctness, not network luck.

## Roadmap

Scoped to the two techniques and the core they sit on. Built so the larger arc slots in
without changing that core:

- **Known-vs-absent trajectories** — does a fact the model *has* sharpen gradually, while
  a fabricated one only commits late?
- **An abstention direction** — fit a probe separating "I never mentioned that" from
  confabulation, then steer it.
- **Scale up** — the nnsight adapter is already the API a larger model would use.

## Notes & limitations

- Prompts are chosen to be ones the model actually gets right; GPT-2-small's modest
  confidence on some facts is shown, not hidden.
- Causal tracing uses **interchange** (resample) corruption, not ROME's Gaussian
  embedding noise — it needs only the patch primitive, so it's robust across both
  backends — [why](docs/methodology.md#corruption-interchange-not-noise). (Noise is supported on TransformerLens.)
- Some MPS kernels aren't bit-deterministic, so seeds fix sampling/corruption but not
  every float on Apple silicon.

## References

- nostalgebraist (2020), *interpreting GPT: the logit lens*.
- Meng, Bau, Andonian, Belinkov (2022), *Locating and Editing Factual Associations in GPT* (ROME).
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) · [nnsight](https://nnsight.net)

## License

MIT — see [LICENSE](LICENSE).
