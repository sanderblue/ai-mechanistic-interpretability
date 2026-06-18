"""Prompts and the interchange-corruption helper.

Causal tracing here uses **interchange** (a.k.a. resample) corruption: run a clean
prompt and a minimally-different "corrupt" prompt that swaps the subject, then
patch clean activations back in. This needs only token sequences and the activation
patch primitive, so it is robust across both backends (unlike embedding noise,
which depends on backend-specific hook behaviour).

The two prompts must tokenise to the same length and differ only at the subject;
:func:`corrupted_positions` finds the differing positions by a plain token-id diff,
which needs no tokenizer offset mapping and works identically on every backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass
class FactPrompt:
    """A clean/corrupt prompt pair for causal tracing.

    ``clean_prompt`` and ``corrupt_prompt`` should be identical except for the
    subject (e.g. "The Eiffel Tower ..." vs "The Colosseum ...") and should tokenise
    to the same number of tokens. ``answer`` is the clean fact's continuation
    (e.g. " Paris"); ``counterfactual_answer`` is the corrupt prompt's
    (e.g. " Rome") and defines the logit-difference metric.
    """

    clean_prompt: str
    corrupt_prompt: str
    answer: str
    counterfactual_answer: str
    subject: str | None = None


def load_facts(path: str | Path) -> dict[str, FactPrompt]:
    """Load fact pairs from a YAML file into ``{id: FactPrompt}``."""
    raw = yaml.safe_load(Path(path).read_text())
    return {key: FactPrompt(**value) for key, value in raw["facts"].items()}


def _to_list(ids) -> list[int]:
    if isinstance(ids, torch.Tensor):
        return ids.flatten().tolist()
    return list(ids)


def corrupted_positions(clean_ids, corrupt_ids) -> list[int]:
    """Token positions where the clean and corrupt sequences differ.

    Raises if the two do not tokenise to equal length — that misalignment would
    make the position-by-position causal-tracing heatmap meaningless, so we fail
    loudly with guidance rather than silently producing a wrong plot.
    """
    clean = _to_list(clean_ids)
    corrupt = _to_list(corrupt_ids)
    if len(clean) != len(corrupt):
        raise ValueError(
            f"Clean and corrupt prompts must tokenise to the same length, got "
            f"{len(clean)} vs {len(corrupt)}. Choose subjects whose names use the "
            f"same number of tokens for this model's tokenizer."
        )
    return [i for i, (a, b) in enumerate(zip(clean, corrupt, strict=True)) if a != b]
