"""Activation-patching invariants (opt-in: ``pytest -m integration``).

These pin down the patch primitive's correctness on both backends:
* patching a clean activation back into its own run changes nothing (identity), and
* a full causal trace recovers the fact at the subject token.
"""

import pytest
import torch

from interp.models.base import Patch, Site


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["transformer_lens", "nnsight"])
def test_identity_patch_is_a_noop(backend):
    from interp import load_model

    model = load_model("gpt2", backend=backend, device="cpu")
    tokens = model.to_tokens("The capital of France is")
    base, cache = model.run_with_cache(tokens, [Site.RESID_POST])

    last_layer = model.n_layers - 1
    last_pos = tokens.shape[1] - 1
    identity = Patch(
        Site.RESID_POST, last_layer, last_pos, cache.resid_post(last_layer)[0, last_pos, :]
    )
    patched = model.forward(tokens, patches=[identity])

    assert torch.allclose(base[0, -1, :].float(), patched[0, -1, :].float(), atol=1e-3)


@pytest.mark.integration
def test_causal_trace_peaks_on_the_subject_token():
    from interp import causal_trace, load_model
    from interp.prompts import FactPrompt

    model = load_model("gpt2", backend="transformer_lens", device="cpu")
    fact = FactPrompt(
        clean_prompt="The capital of France is",
        corrupt_prompt="The capital of Japan is",
        answer=" Paris",
        counterfactual_answer=" Tokyo",
        subject="France",
    )
    result = causal_trace(model, fact)

    # Clean prefers the answer, corrupt prefers the counterfactual.
    assert result.clean_metric > 0 > result.corrupt_metric
    # The strongest restoration sits on a corrupted (subject) position, near-complete.
    assert result.peak["position"] in result.corrupted_positions
    assert result.peak["recovery"] > 0.8
