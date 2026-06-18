"""TransformerLens backend.

Default backend for GPT-2. TransformerLens gives us a uniform hook namespace
(``blocks.{i}.hook_resid_post``) and folds LayerNorm / centres the unembedding by
default — which is exactly the canonical setup the logit lens assumes — so this
adapter is a thin, faithful mapping from our abstract :class:`Site`s onto TL hooks.

The ``transformer_lens`` import is local to this module so the package can run on
the nnsight backend alone (and so CI never installs it).
"""

from __future__ import annotations

from collections import defaultdict
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

_SITE_TO_HOOK = {
    Site.RESID_PRE: "resid_pre",
    Site.RESID_POST: "resid_post",
}


class TransformerLensAdapter(ModelAdapter):
    def __init__(self, hf_id: str, device: str, dtype: torch.dtype):
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(hf_id, device=device)
        model = model.to(dtype)
        model.eval()
        self._model = model
        self.info = ModelInfo(
            name=hf_id,
            backend="transformer_lens",
            n_layers=model.cfg.n_layers,
            d_model=model.cfg.d_model,
            device=device,
            dtype=dtype,
            prepends_bos=True,  # TL prepends BOS by default
        )

    # --- tokenisation ----------------------------------------------------------
    @property
    def tokenizer(self):
        return self._model.tokenizer

    def to_tokens(self, text: str, prepend_bos: bool | None = None) -> torch.Tensor:
        kwargs = {} if prepend_bos is None else {"prepend_bos": prepend_bos}
        return self._model.to_tokens(text, **kwargs)

    def to_str_tokens(self, text: str, prepend_bos: bool | None = None) -> list[str]:
        kwargs = {} if prepend_bos is None else {"prepend_bos": prepend_bos}
        return self._model.to_str_tokens(text, **kwargs)

    def to_string(self, tokens) -> str:
        return self._model.to_string(tokens)

    # --- forward passes --------------------------------------------------------
    def _hook_name(self, site: Site, layer: int) -> str:
        if site == Site.EMBED:
            return "hook_embed"
        return f"blocks.{layer}.hook_{_SITE_TO_HOOK[site]}"

    def run_with_cache(
        self, tokens: torch.Tensor, sites: Sequence[Site]
    ) -> tuple[torch.Tensor, ActivationCache]:
        wanted = set()
        for site in sites:
            if site == Site.EMBED:
                wanted.add("hook_embed")
            else:
                for layer in range(self.n_layers):
                    wanted.add(self._hook_name(site, layer))

        logits, cache = self._model.run_with_cache(tokens, names_filter=lambda name: name in wanted)

        data = {}
        for site in sites:
            if site == Site.EMBED:
                data[(Site.EMBED, 0)] = cache["hook_embed"]
            else:
                for layer in range(self.n_layers):
                    data[(site, layer)] = cache[self._hook_name(site, layer)]
        return logits, ActivationCache(data)

    def forward(
        self,
        tokens: torch.Tensor,
        patches: Sequence[Patch] = (),
        input_noise: InputNoise | None = None,
    ) -> torch.Tensor:
        by_hook: dict[str, list[Patch]] = defaultdict(list)
        for patch in patches:
            by_hook[self._hook_name(patch.site, patch.layer)].append(patch)

        fwd_hooks = []

        for hook_name, hook_patches in by_hook.items():

            def make_patch_hook(hook_patches=hook_patches):
                def hook(act, hook):  # noqa: ARG001 (TL hook signature)
                    for patch in hook_patches:
                        act[:, patch.position, :] = patch.value.to(act.dtype)
                    return act

                return hook

            fwd_hooks.append((hook_name, make_patch_hook()))

        if input_noise is not None:
            noise = self._make_noise(input_noise)

            def noise_hook(act, hook):  # noqa: ARG001
                positions = list(input_noise.positions)
                act[:, positions, :] = act[:, positions, :] + noise.to(act.dtype)
                return act

            fwd_hooks.append(("hook_embed", noise_hook))

        return self._model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    def _make_noise(self, input_noise: InputNoise) -> torch.Tensor:
        # Generate on CPU for deterministic, device-independent noise, then move.
        generator = torch.Generator().manual_seed(input_noise.seed)
        noise = torch.randn(len(input_noise.positions), self.d_model, generator=generator)
        return (noise * input_noise.std).to(self.device)

    # --- logit lens ------------------------------------------------------------
    def unembed(self, resid: torch.Tensor, apply_final_ln: bool = True) -> torch.Tensor:
        if apply_final_ln:
            resid = self._model.ln_final(resid)
        return self._model.unembed(resid)

    def embedding_std(self) -> float:
        return float(self._model.W_E.std())
