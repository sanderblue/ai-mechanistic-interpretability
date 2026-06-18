"""Cross-backend faithfulness check (opt-in: ``pytest -m integration``).

Loads GPT-2 under *both* backends and asserts they produce the same next-token
distribution. This is the load-bearing claim of the adapter design: experiment
results don't depend on which backend produced them. (Raw logits differ because
TransformerLens centres the unembedding — a softmax-invariant shift — so we compare
probability distributions, not raw logit vectors.)
"""

import pytest
import torch

from interp.models.base import Site


@pytest.mark.integration
def test_tl_and_nnsight_agree_on_gpt2():
    from interp import load_model

    tl = load_model("gpt2", backend="transformer_lens", device="cpu")
    nn = load_model("gpt2", backend="nnsight", device="cpu")

    prompt = "Water is made of hydrogen and"
    tl_tokens = tl.to_tokens(prompt, prepend_bos=False)
    nn_tokens = nn.to_tokens(prompt)
    assert tl_tokens.flatten().tolist() == nn_tokens.flatten().tolist()

    _, tl_cache = tl.run_with_cache(tl_tokens, [Site.RESID_POST])
    _, nn_cache = nn.run_with_cache(nn_tokens, [Site.RESID_POST])

    tl_probs = (
        tl.unembed(tl_cache.resid_post(tl.n_layers - 1)[:, -1, :]).flatten().float().softmax(-1)
    )
    nn_probs = (
        nn.unembed(nn_cache.resid_post(nn.n_layers - 1)[:, -1, :]).flatten().float().softmax(-1)
    )

    assert int(tl_probs.argmax()) == int(nn_probs.argmax())
    kl = torch.nn.functional.kl_div(nn_probs.log(), tl_probs, reduction="sum")
    assert float(kl) < 1e-3
    assert float((tl_probs - nn_probs).abs().max()) < 1e-3
