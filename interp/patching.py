"""Causal tracing via activation patching.

The question: *where in the network does a fact causally live?* The method:

1. Run the **clean** prompt ("The Eiffel Tower is located in the city of") and
   cache the residual stream after every layer.
2. Run a **corrupt** prompt that swaps the subject ("The Colosseum ...") — the
   model now prefers the counterfactual answer.
3. For every (layer, position), copy the *clean* residual into the corrupt run and
   measure how much of the correct answer is restored (``recovery``).

A bright band in the resulting [layer × position] heatmap marks the activations
that carry the fact — classically, mid-layer activations at the subject's tokens.

Reference: Meng et al., "Locating and Editing Factual Associations in GPT" (ROME),
2022 — adapted here to interchange (resample) corruption for backend robustness.
"""

from __future__ import annotations

from dataclasses import dataclass

from interp.metrics import answer_prob, logit_diff
from interp.models.base import ModelAdapter, Patch, Site
from interp.prompts import FactPrompt, corrupted_positions

_METRICS = {
    "logit_diff": lambda logits, a, c: logit_diff(logits, a, c),
    "answer_prob": lambda logits, a, c: answer_prob(logits, a),
}


@dataclass
class CausalTraceResult:
    clean_prompt: str
    corrupt_prompt: str
    answer: str
    counterfactual_answer: str
    metric: str
    layers: list[int]
    str_tokens: list[str]
    corrupted_positions: list[int]
    clean_metric: float
    corrupt_metric: float
    recovery: list[list[float]]  # [n_layers][n_positions]
    peak: dict  # {layer, position, recovery} of the strongest restoration

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def causal_trace(
    model: ModelAdapter,
    fact: FactPrompt,
    metric: str = "logit_diff",
    layers: list[int] | None = None,
) -> CausalTraceResult:
    """Sweep activation patches over (layer, position) and return a recovery map."""
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Options: {sorted(_METRICS)}")
    score = _METRICS[metric]

    clean_tokens = model.to_tokens(fact.clean_prompt)
    corrupt_tokens = model.to_tokens(fact.corrupt_prompt)
    answer_id = model.first_token_id(fact.answer)
    counter_id = model.first_token_id(fact.counterfactual_answer)
    positions = corrupted_positions(clean_tokens, corrupt_tokens)
    str_tokens = model.to_str_tokens(fact.clean_prompt)
    seq_len = clean_tokens.shape[1]

    # Baselines: clean (answer preferred) and corrupt (counterfactual preferred).
    clean_logits, cache = model.run_with_cache(clean_tokens, [Site.RESID_POST])
    clean_metric = score(clean_logits, answer_id, counter_id)
    corrupt_logits = model.forward(corrupt_tokens)
    corrupt_metric = score(corrupt_logits, answer_id, counter_id)

    from interp.metrics import recovery as recovery_fn

    layers = layers if layers is not None else list(range(model.n_layers))
    grid: list[list[float]] = []
    peak = {"layer": -1, "position": -1, "recovery": float("-inf")}
    for layer in layers:
        clean_resid = cache.resid_post(layer)  # [1, seq, d]
        row: list[float] = []
        for pos in range(seq_len):
            patch = Patch(Site.RESID_POST, layer, pos, clean_resid[0, pos, :])
            patched_logits = model.forward(corrupt_tokens, patches=[patch])
            rec = recovery_fn(
                score(patched_logits, answer_id, counter_id), clean_metric, corrupt_metric
            )
            row.append(rec)
            if rec > peak["recovery"]:
                peak = {"layer": layer, "position": pos, "recovery": rec}
        grid.append(row)

    return CausalTraceResult(
        clean_prompt=fact.clean_prompt,
        corrupt_prompt=fact.corrupt_prompt,
        answer=fact.answer,
        counterfactual_answer=fact.counterfactual_answer,
        metric=metric,
        layers=layers,
        str_tokens=str_tokens,
        corrupted_positions=positions,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        recovery=grid,
        peak=peak,
    )
