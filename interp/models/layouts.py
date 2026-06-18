"""Per-architecture module paths for the nnsight backend.

nnsight reaches into a model by attribute path (e.g. ``transformer.h`` for GPT-2,
``model.layers`` for Qwen/Llama). TransformerLens hides this behind uniform hook
names, but for raw Hugging Face models we need to know the layout. Supporting a
new architecture family is one entry here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    """Where the interesting modules live, relative to the HF model root.

    Paths are dotted attribute chains. ``layers`` points at the ``nn.ModuleList``
    of transformer blocks; ``final_norm`` and ``unembed`` are applied (in that
    order) to a residual to read logits; ``embed`` is the input token embedding.
    """

    layers: str
    final_norm: str
    unembed: str
    embed: str


LAYOUTS: dict[str, Layout] = {
    # GPT-2 family (HF GPT2LMHeadModel)
    "gpt2": Layout(
        layers="transformer.h",
        final_norm="transformer.ln_f",
        unembed="lm_head",
        embed="transformer.wte",
    ),
    # Qwen2/Qwen3 + Llama-style decoders (HF *ForCausalLM with a `.model` trunk)
    "qwen": Layout(
        layers="model.layers",
        final_norm="model.norm",
        unembed="lm_head",
        embed="model.embed_tokens",
    ),
    "llama": Layout(
        layers="model.layers",
        final_norm="model.norm",
        unembed="lm_head",
        embed="model.embed_tokens",
    ),
}


def get_layout(name: str) -> Layout:
    if name not in LAYOUTS:
        raise KeyError(f"Unknown architecture layout {name!r}. Known: {sorted(LAYOUTS)}")
    return LAYOUTS[name]


def resolve(root, path: str):
    """Follow a dotted attribute path from ``root`` (an nnsight envoy or module)."""
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj
