"""The logit lens.

At each layer, project that layer's residual stream through the model's final norm
and unembedding to read the token distribution the model is "currently betting on".
Watching the correct token climb in probability (and rank) across depth shows *where*
a prediction crystallises — gradual sharpening through the mid-stack looks like
genuine retrieval; a sudden late jump looks like a last-moment commit.

Reference: nostalgebraist, "interpreting GPT: the logit lens" (2020).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from interp.metrics import answer_prob, answer_rank, logits_at
from interp.models.base import ModelAdapter, Site


@dataclass
class LensResult:
    prompt: str
    answer: str
    answer_id: int
    layers: list[int]
    answer_prob: list[float]  # P(answer) read at each layer
    answer_rank: list[int]  # rank of answer token at each layer (0 == top-1)
    top_token: list[str]  # the layer's own top-1 token (for intuition)
    final_prob: float  # P(answer) at the last layer
    crossover_layer: int | None  # first layer at which the answer becomes top-1

    def to_dict(self) -> dict:
        return asdict(self)


def logit_lens(
    model: ModelAdapter,
    prompt: str,
    answer: str,
    position: int = -1,
    layers: list[int] | None = None,
) -> LensResult:
    """Run the logit lens for ``answer`` at ``position`` over the given layers."""
    tokens = model.to_tokens(prompt)
    answer_id = model.first_token_id(answer)
    _, cache = model.run_with_cache(tokens, [Site.RESID_POST])

    layers = layers if layers is not None else list(range(model.n_layers))
    probs: list[float] = []
    ranks: list[int] = []
    tops: list[str] = []
    for layer in layers:
        resid = cache.resid_post(layer)[:, position, :]
        layer_logits = model.unembed(resid)
        probs.append(answer_prob(layer_logits, answer_id))
        ranks.append(answer_rank(layer_logits, answer_id))
        tops.append(model.to_string(int(logits_at(layer_logits).argmax())))

    crossover = next((layer for layer, rank in zip(layers, ranks, strict=True) if rank == 0), None)
    return LensResult(
        prompt=prompt,
        answer=answer,
        answer_id=answer_id,
        layers=layers,
        answer_prob=probs,
        answer_rank=ranks,
        top_token=tops,
        final_prob=probs[-1],
        crossover_layer=crossover,
    )
