"""Model loading: friendly names → a loaded :class:`ModelAdapter`.

``load_model("gpt2")`` and ``load_model("qwen3-0.6b")`` are the two entry points
the experiments use. The registry records each model's Hugging Face id, its
recommended backend, and (for nnsight) its architecture layout. Adding a model is
one line here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from interp.config import default_dtype, detect_device
from interp.models.base import (
    ActivationCache,
    InputNoise,
    ModelAdapter,
    ModelInfo,
    Patch,
    Site,
)

__all__ = [
    "ActivationCache",
    "InputNoise",
    "ModelAdapter",
    "ModelInfo",
    "Patch",
    "Site",
    "MODEL_REGISTRY",
    "ModelSpec",
    "load_model",
]


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    default_backend: str  # "transformer_lens" | "nnsight"
    layout: str  # architecture layout name (used by the nnsight backend)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # GPT-2: the mech-interp reference model. Tiny, instant to iterate on.
    "gpt2": ModelSpec("gpt2", "transformer_lens", "gpt2"),
    "gpt2-medium": ModelSpec("gpt2-medium", "transformer_lens", "gpt2"),
    # Qwen3-0.6B: small enough for a laptop, same family as the 14B persona model.
    "qwen3-0.6b": ModelSpec("Qwen/Qwen3-0.6B", "nnsight", "qwen"),
}


def load_model(
    name: str,
    backend: str = "auto",
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> ModelAdapter:
    """Load a model by friendly name (or raw HF id) on the chosen backend."""
    spec = MODEL_REGISTRY.get(name)
    hf_id = spec.hf_id if spec else name
    layout = spec.layout if spec else "qwen"

    chosen = backend
    if chosen == "auto":
        chosen = spec.default_backend if spec else "nnsight"

    device = detect_device(device)
    dtype = dtype or default_dtype(device)

    if chosen == "transformer_lens":
        from interp.models.transformer_lens_adapter import TransformerLensAdapter

        return TransformerLensAdapter(hf_id, device=device, dtype=dtype)
    if chosen == "nnsight":
        from interp.models.nnsight_adapter import NNsightAdapter

        return NNsightAdapter(hf_id, layout=layout, device=device, dtype=dtype)
    raise ValueError(f"Unknown backend {chosen!r} (expected 'transformer_lens' or 'nnsight')")
