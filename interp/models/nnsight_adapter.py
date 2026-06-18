"""nnsight backend.

Default backend for Qwen3 — and, crucially, the *same* adapter API that will
later drive the 14B persona model on the multi-GPU rig. nnsight wraps raw Hugging
Face ``transformers`` models, so it supports any architecture HF does.

nnsight 0.7 specifics this adapter is written around (learned empirically — see
``docs/methodology.md``):

* ``lm._model`` is the underlying HF module; we call its final-norm and unembed
  modules eagerly for the logit lens.
* Inside a ``with lm.trace(...)`` block, nnsight rewrites the body. Only the
  results of ``name = expr.save()`` and in-place mutations of objects that exist
  *outside* the block (e.g. ``list.append``) survive. So we cache by appending to
  an external list and read logits via a single saved name.
* Interventions must assign a **concrete tensor** — reading a proxy on the
  right-hand side (proxy arithmetic) does not propagate. Activation patching
  assigns cached clean tensors, so it works; embedding *noise* would require
  proxy arithmetic, so it is intentionally unsupported here (use interchange
  corruption, which is the cross-backend default).
* In transformers >= 5 the decoder blocks return a bare ``[batch, seq, d_model]``
  tensor; ``_as_hidden`` keeps us robust to the older tuple-returning blocks too.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import torch

from interp.models.base import (
    ActivationCache,
    InputNoise,
    ModelAdapter,
    ModelInfo,
    Patch,
    Site,
)
from interp.models.layouts import get_layout, resolve


def _as_hidden(output):
    """Decoder blocks return either a bare tensor (transformers >= 5) or a tuple."""
    if isinstance(output, tuple):
        return output[0]
    return output


class NNsightAdapter(ModelAdapter):
    def __init__(self, hf_id: str, layout: str, device: str, dtype: torch.dtype):
        from nnsight import LanguageModel

        self._layout = get_layout(layout)

        # Set precision at load time via the `dtype` kwarg (Qwen ships bf16; we want
        # float32 off-CUDA). We deliberately do NOT call `.to()`/`.eval()` *before*
        # this point: mutating `lm._model` in place can make nn.Module.train recurse
        # through nnsight's envoy wrapping.
        lm, actual_device = self._load(LanguageModel, hf_id, device, dtype)
        self._lm = lm
        self._raw = lm._model
        # nnsight's envoy wrapping can make nn.Module.train recurse; harmless to skip
        # since these models have no active dropout at inference.
        with contextlib.suppress(RecursionError):
            self._raw.eval()

        self.info = ModelInfo(
            name=hf_id,
            backend="nnsight",
            n_layers=lm.config.num_hidden_layers,
            d_model=lm.config.hidden_size,
            device=actual_device,
            dtype=dtype,
            prepends_bos=False,  # HF-native tokenisation, no BOS prepended
        )
        self._init_unembed()

    def _init_unembed(self) -> None:
        """Snapshot the final-norm + unembed weights as plain tensors.

        We must NOT call these modules' ``forward()`` eagerly: nnsight instruments
        them, so an out-of-trace call routes through its interleaver and segfaults.
        We also ``.clone()`` the weights: the dispatched model's parameters are
        mmap-backed (accelerate/safetensors), which is fine inside nnsight's traced
        forward but bus-errors in an external BLAS call — cloning materialises them
        in normal memory. The logit lens then recomputes norm + unembed with
        functional ops (see :meth:`unembed`).
        """

        def materialise(tensor):
            return tensor.detach().clone().contiguous() if tensor is not None else None

        norm = resolve(self._raw, self._layout.final_norm)
        head = resolve(self._raw, self._layout.unembed)
        self._norm_weight = materialise(norm.weight)
        self._norm_bias = materialise(getattr(norm, "bias", None))
        self._norm_eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-5)))
        self._norm_is_rms = "rms" in type(norm).__name__.lower()
        self._unembed_weight = materialise(head.weight)  # [vocab, d_model]
        self._unembed_bias = materialise(getattr(head, "bias", None))

    @staticmethod
    def _load(LanguageModel, hf_id: str, device: str, dtype: torch.dtype):
        """Dispatch the model, falling back to CPU if the device is unavailable."""
        try:
            return LanguageModel(hf_id, device_map=device, dispatch=True, dtype=dtype), device
        except Exception:
            return LanguageModel(hf_id, device_map="cpu", dispatch=True, dtype=dtype), "cpu"

    # --- tokenisation ----------------------------------------------------------
    @property
    def tokenizer(self):
        return self._lm.tokenizer

    def to_tokens(self, text: str, prepend_bos: bool | None = None) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if prepend_bos and self.tokenizer.bos_token_id is not None:
            bos = torch.tensor([[self.tokenizer.bos_token_id]])
            ids = torch.cat([bos, ids], dim=1)
        return ids.to(self.device)

    def to_str_tokens(self, text: str, prepend_bos: bool | None = None) -> list[str]:
        ids = self.to_tokens(text, prepend_bos)[0].tolist()
        return [self.tokenizer.decode([i]) for i in ids]

    def to_string(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.flatten().tolist()
        elif isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens)

    # --- forward passes --------------------------------------------------------
    def run_with_cache(
        self, tokens: torch.Tensor, sites: Sequence[Site]
    ) -> tuple[torch.Tensor, ActivationCache]:
        tokens = tokens.to(self.device)
        layers = resolve(self._lm, self._layout.layers)
        embed = resolve(self._lm, self._layout.embed)
        # Only the embedding feeds RESID_PRE[0]/EMBED; saving it is also the one place
        # the HF embedding envoy is touched, so guard it behind an actual request.
        need_embed = Site.EMBED in sites or Site.RESID_PRE in sites

        post_acts: list = []
        embed_acts: list = []
        with self._lm.trace(tokens):
            # nnsight 0.7 wants saves in forward-execution order: embeddings, then blocks.
            if need_embed:
                embed_acts.append(embed.output.save())
            for i in range(self.n_layers):
                post_acts.append(layers[i].output.save())
            logits = self._lm.output.logits.save()

        data: dict = {}
        if Site.RESID_POST in sites:
            for i in range(self.n_layers):
                data[(Site.RESID_POST, i)] = _as_hidden(post_acts[i])
        if Site.EMBED in sites:
            data[(Site.EMBED, 0)] = _as_hidden(embed_acts[0])
        if Site.RESID_PRE in sites:
            # resid_pre[l] is resid_post[l-1]; layer 0 is the embedding.
            for i in range(self.n_layers):
                src = embed_acts[0] if i == 0 else post_acts[i - 1]
                data[(Site.RESID_PRE, i)] = _as_hidden(src)
        return logits, ActivationCache(data)

    def forward(
        self,
        tokens: torch.Tensor,
        patches: Sequence[Patch] = (),
        input_noise: InputNoise | None = None,
    ) -> torch.Tensor:
        if input_noise is not None:
            raise NotImplementedError(
                "Embedding-noise corruption is unsupported on the nnsight backend "
                "(it requires in-trace proxy arithmetic). Use corruption='interchange'."
            )
        tokens = tokens.to(self.device)
        layers = resolve(self._lm, self._layout.layers)
        resid_patches = [p for p in patches if p.site == Site.RESID_POST]
        if any(p.site != Site.RESID_POST for p in patches):
            raise NotImplementedError("nnsight backend currently patches RESID_POST only")

        with self._lm.trace(tokens):
            for patch in resid_patches:
                value = patch.value.to(self.device, self.dtype)
                layers[patch.layer].output[:, patch.position, :] = value
            logits = self._lm.output.logits.save()
        return logits

    # --- logit lens ------------------------------------------------------------
    def unembed(self, resid: torch.Tensor, apply_final_ln: bool = True) -> torch.Tensor:
        resid = resid.to(self._unembed_weight.device, self._unembed_weight.dtype)
        if apply_final_ln:
            if self._norm_is_rms:
                var = resid.pow(2).mean(-1, keepdim=True)
                resid = resid * torch.rsqrt(var + self._norm_eps) * self._norm_weight
            else:
                resid = torch.nn.functional.layer_norm(
                    resid, (self.d_model,), self._norm_weight, self._norm_bias, self._norm_eps
                )
        logits = resid @ self._unembed_weight.t()
        if self._unembed_bias is not None:
            logits = logits + self._unembed_bias
        return logits

    def embedding_std(self) -> float:
        # Clone for the same mmap-storage reason as the unembed weights.
        return float(self._raw.get_input_embeddings().weight.detach().clone().std())
