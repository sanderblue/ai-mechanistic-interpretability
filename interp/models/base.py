"""The backend-agnostic model interface.

This is the seam that lets the same experiment code run on TransformerLens
(GPT-2) and nnsight (Qwen3, and later the 14B). It normalises *two* axes at once:

* **backend** — TransformerLens hook names vs nnsight tracing proxies, and
* **architecture** — GPT-2's ``transformer.h[i]`` blocks vs Qwen/Llama's
  ``model.layers[i]`` blocks with a final RMSNorm.

Experiment code speaks only in abstract :class:`Site` values (e.g. "the residual
stream after layer 7") and never sees a backend- or architecture-specific name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import torch


class Site(str, Enum):
    """An abstract place in the network where we read or write activations.

    Kept deliberately small — these are the points the current experiments need.
    Adding ``MLP_OUT`` / ``ATTN_OUT`` later means one enum value plus its mapping
    in each adapter.
    """

    EMBED = "embed"  # token embeddings (input to block 0)
    RESID_PRE = "resid_pre"  # residual stream entering a layer
    RESID_POST = "resid_post"  # residual stream leaving a layer


# A cache is keyed by (site, layer). EMBED ignores the layer index (stored as 0).
CacheKey = tuple[Site, int]


class ActivationCache:
    """A thin, backend-neutral container of cached activations.

    Wraps a ``{(site, layer): tensor}`` dict with convenience accessors so
    experiment code reads ``cache.resid_post(7)`` instead of indexing raw hook
    names. Tensors keep their native shape ``[batch, seq, d_model]``.
    """

    def __init__(self, data: dict[CacheKey, torch.Tensor]):
        self._data = data

    def __getitem__(self, key: CacheKey) -> torch.Tensor:
        site, layer = key
        return self._data[(Site(site), layer)]

    def __contains__(self, key: CacheKey) -> bool:
        site, layer = key
        return (Site(site), layer) in self._data

    def resid_post(self, layer: int) -> torch.Tensor:
        return self._data[(Site.RESID_POST, layer)]

    def resid_pre(self, layer: int) -> torch.Tensor:
        return self._data[(Site.RESID_PRE, layer)]

    def embed(self) -> torch.Tensor:
        return self._data[(Site.EMBED, 0)]

    def keys(self) -> list[CacheKey]:
        return list(self._data.keys())


@dataclass
class Patch:
    """Overwrite one token position at one site with a fixed value.

    ``value`` has shape ``[d_model]`` (it replaces ``act[:, position, :]``). This
    is the primitive behind activation patching / causal tracing.
    """

    site: Site
    layer: int
    position: int
    value: torch.Tensor


@dataclass
class InputNoise:
    """ROME-style corruption: add Gaussian noise to embeddings at ``positions``.

    Applied at the embedding site, so the sequence length and every downstream
    position stay aligned between the clean and corrupted runs — which is what
    makes the causal-tracing heatmap interpretable position-by-position.
    """

    positions: Sequence[int]
    std: float
    seed: int = 0


@dataclass
class ModelInfo:
    """Static facts about a loaded model, filled in by each adapter."""

    name: str
    backend: str
    n_layers: int
    d_model: int
    device: str
    dtype: torch.dtype
    prepends_bos: bool = False
    extra: dict = field(default_factory=dict)


class ModelAdapter(ABC):
    """Uniform read/write access to a transformer's internals.

    Concrete subclasses (:class:`~interp.models.transformer_lens_adapter.TransformerLensAdapter`,
    :class:`~interp.models.nnsight_adapter.NNsightAdapter`) implement the abstract
    methods; everything else in the codebase depends only on this surface.
    """

    info: ModelInfo

    # --- convenience properties (read off ModelInfo) ---------------------------
    @property
    def n_layers(self) -> int:
        return self.info.n_layers

    @property
    def d_model(self) -> int:
        return self.info.d_model

    @property
    def device(self) -> str:
        return self.info.device

    @property
    def dtype(self) -> torch.dtype:
        return self.info.dtype

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def backend(self) -> str:
        return self.info.backend

    @property
    def prepends_bos(self) -> bool:
        return self.info.prepends_bos

    # --- tokenisation ----------------------------------------------------------
    @property
    @abstractmethod
    def tokenizer(self):
        """The underlying Hugging Face tokenizer (exposed by both backends)."""

    @abstractmethod
    def to_tokens(self, text: str, prepend_bos: bool | None = None) -> torch.Tensor:
        """Encode text to a ``[1, seq]`` LongTensor on the model's device."""

    @abstractmethod
    def to_str_tokens(self, text: str, prepend_bos: bool | None = None) -> list[str]:
        """Encode text and decode each token back to a string (for axis labels)."""

    @abstractmethod
    def to_string(self, tokens) -> str:
        """Decode a token id / list / tensor back to text."""

    def first_token_id(self, text: str) -> int:
        """Token id of the first token of ``text`` (no special tokens).

        Used to turn an answer string like ``" Paris"`` into the single vocab
        index we track through the network.
        """
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"{text!r} tokenised to zero tokens")
        return int(ids[0])

    # --- forward passes --------------------------------------------------------
    @abstractmethod
    def run_with_cache(
        self, tokens: torch.Tensor, sites: Sequence[Site]
    ) -> tuple[torch.Tensor, ActivationCache]:
        """Run a forward pass, returning ``(logits, cache)`` for the given sites."""

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        patches: Sequence[Patch] = (),
        input_noise: InputNoise | None = None,
    ) -> torch.Tensor:
        """Run a forward pass with optional interventions, returning logits.

        ``patches`` overwrite activations at specific (site, layer, position);
        ``input_noise`` corrupts embeddings. Both can be combined in a single
        pass — exactly what the causal-tracing sweep needs (corrupt the subject,
        then patch one clean activation back in).
        """

    # --- logit lens primitive --------------------------------------------------
    @abstractmethod
    def unembed(self, resid: torch.Tensor, apply_final_ln: bool = True) -> torch.Tensor:
        """Project a residual-stream vector to vocab logits.

        With ``apply_final_ln=True`` this applies the model's final
        LayerNorm/RMSNorm before the unembedding matrix — i.e. the standard
        "logit lens" read of what the model is currently betting on.
        """

    @abstractmethod
    def embedding_std(self) -> float:
        """Std of the input embedding matrix; sets the causal-tracing noise scale."""
