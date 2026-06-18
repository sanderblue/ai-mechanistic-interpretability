# Methodology

This document explains the two techniques the lab implements, and the
backend-specific engineering that makes them run identically on TransformerLens and
nnsight.

## 1. The logit lens

A transformer builds its prediction by repeatedly adding to a **residual stream** —
a running vector at each token position that every layer reads from and writes to.
The final prediction is `unembed(final_norm(residual))`. The logit lens
([nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens))
asks: what if we apply that same read-out to the residual stream at an *earlier*
layer?

```
layer_logits = unembed(final_norm(resid_post[layer]))
```

The resulting distribution is "what the model would predict if it had to stop
thinking now". Tracking the correct token's probability across depth shows *when* a
prediction forms. A token that climbs gradually through the mid-stack looks like
genuine, distributed retrieval; one that only appears in the last layer or two looks
like a late commit.

Implemented in [`interp/lenses.py`](../interp/lenses.py). The lens needs exactly two
adapter primitives: `run_with_cache` (to read `resid_post` at every layer) and
`unembed` (to project a residual to logits).

## 2. Causal tracing (activation patching)

The logit lens is *correlational* — it shows where a prediction is legible, not
where it is *computed*. Causal tracing
([Meng et al., 2022, ROME](https://arxiv.org/abs/2202.05262)) makes it causal:

1. Run a **clean** prompt and cache the residual stream after every layer.
2. Run a **corrupt** prompt where the subject is changed, so the model now prefers a
   different answer.
3. For every (layer, position), copy the *clean* residual into the corrupt run and
   measure how much of the correct answer is restored.

The restoration metric is **recovery**:

```
recovery = (patched_metric - corrupt_metric) / (clean_metric - corrupt_metric)
```

where the metric is the **logit difference** between the correct and counterfactual
answers (e.g. `logit(" Paris") − logit(" Tokyo")`). Recovery is 0 at the corrupt
baseline and 1 when the patch fully restores the clean behaviour. Plotting it as a
[layer × position] heatmap reveals which activations carry the fact.

Implemented in [`interp/patching.py`](../interp/patching.py). It needs
`run_with_cache` plus `forward(patches=...)`, where a patch overwrites one
(site, layer, position) with a cached clean value.

### Corruption: interchange, not noise

ROME corrupts the subject by adding Gaussian noise to its token *embeddings*. We use
**interchange** (a.k.a. resample) corruption instead: the corrupt prompt is a real,
minimally-different prompt that swaps the subject (`France → Japan`). Two reasons:

1. **Robustness.** Interchange needs only the activation-patch primitive — overwrite
   a clean residual into a run of corrupt tokens. It requires no embedding hook,
   which (see below) is exactly the operation that is fragile on the nnsight stack.
2. **Determinism.** There is no noise scale to tune; the corrupt run is a fixed,
   reproducible forward pass.

The cost is a constraint: the clean and corrupt prompts must tokenise to the same
length and differ only at the subject, which the lab enforces and checks by a plain
token-id diff ([`corrupted_positions`](../interp/prompts.py)). Gaussian-noise
corruption is still supported on the TransformerLens backend via
`forward(input_noise=...)`.

## 3. Backend engineering notes

These are the non-obvious things learned making one interface sit cleanly over two
very different libraries (TransformerLens 3.x, nnsight 0.7, transformers 5.x). They
live here because they explain *why* the adapters look the way they do.

**Layer outputs are bare tensors (transformers ≥ 5).** Both GPT-2's `GPT2Block` and
Qwen3's `Qwen3DecoderLayer` return a `[batch, seq, d_model]` tensor, not a tuple, so
patching is a clean `output[:, pos, :] = value`. The adapter keeps an `_as_hidden`
guard for older tuple-returning blocks.

**TransformerLens centres the unembedding.** With its default `fold_ln` +
`center_unembed`, TL's logits differ from raw HF logits by a per-token constant. That
constant is invisible to softmax (and to argmax), so the two backends agree on the
*distribution* (KL ≈ 0) while their raw logit vectors look nearly orthogonal. The
cross-backend test compares probabilities, not raw logits, for this reason.

**nnsight 0.7's trace block is source-rewritten.** Inside `with model.trace(...)`,
only the results of `name = expr.save()` and in-place mutations of objects defined
*outside* the block survive. So the adapter caches activations by appending to an
external list, and reads logits via a single saved name. Interventions must assign a
**concrete tensor** — proxy arithmetic on the right-hand side (e.g. `x = x + noise`)
does not propagate, which is the deeper reason embedding-noise corruption is not on
the nnsight path.

**The logit lens can't call nnsight-instrumented modules.** nnsight patches the
wrapped model's `forward`s, so calling `lm._model.lm_head(...)` eagerly (outside a
trace) routes through its interleaver and segfaults. The adapter instead snapshots
the final-norm and unembed *weights* and recomputes the projection with functional
ops. Those weights are mmap-backed by accelerate/safetensors — valid inside the
model's forward but bus-erroring in an external BLAS call — so they are `.clone()`d
into normal memory first.

These are the kind of details that don't show up until you run the code on a real,
current stack; isolating each to one place in the adapter is what keeps the
experiment code clean.
