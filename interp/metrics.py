"""Scalar read-outs over logits.

Small, pure tensor functions (no model, no I/O) so they are trivially unit-tested
in CI and reused across experiments. Everything returns a Python ``float``/``int``
so results are JSON-serialisable.
"""

from __future__ import annotations

import torch


def logits_at(logits: torch.Tensor, position: int = -1) -> torch.Tensor:
    """Return the ``[vocab]`` logit vector at ``position`` (assumes batch size 1)."""
    if logits.ndim == 3:  # [batch, seq, vocab]
        return logits[0, position, :]
    if logits.ndim == 2:  # [seq, vocab]
        return logits[position, :]
    return logits  # [vocab]


def answer_logit(logits: torch.Tensor, token_id: int, position: int = -1) -> float:
    return float(logits_at(logits, position)[token_id])


def answer_prob(logits: torch.Tensor, token_id: int, position: int = -1) -> float:
    return float(logits_at(logits, position).softmax(-1)[token_id])


def answer_rank(logits: torch.Tensor, token_id: int, position: int = -1) -> int:
    """0 == the token is the model's top prediction at this position."""
    vec = logits_at(logits, position)
    return int((vec > vec[token_id]).sum())


def logit_diff(logits: torch.Tensor, answer_id: int, counter_id: int, position: int = -1) -> float:
    """Logit of the correct answer minus a counterfactual — the clean tracing metric.

    Preferred over a raw probability because it is unbounded and not squashed by the
    softmax, so it stays sensitive in the regime causal tracing cares about.
    """
    vec = logits_at(logits, position)
    return float(vec[answer_id] - vec[counter_id])


def recovery(patched: float, clean: float, corrupt: float, eps: float = 1e-8) -> float:
    """Normalised restoration from a patch: 0 == corrupt baseline, 1 == clean.

    ``(patched - corrupt) / (clean - corrupt)``. When the metric barely moves
    between clean and corrupt (degenerate denominator) the patch can't be credited,
    so we return 0.
    """
    denom = clean - corrupt
    if abs(denom) < eps:
        return 0.0
    return (patched - corrupt) / denom
