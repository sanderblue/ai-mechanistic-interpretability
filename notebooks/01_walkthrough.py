# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Walkthrough: logit lens + causal tracing on GPT-2
#
# This is the library used directly, the way you would in a notebook while
# exploring. It is paired with a notebook via [jupytext](https://jupytext.readthedocs.io)
# (`jupytext --to notebook notebooks/01_walkthrough.py`) but also runs as a plain
# script. Everything here is the same code the CLI runs.

# %%
from interp import causal_trace, load_model, logit_lens
from interp.prompts import FactPrompt
from interp.viz import plot_causal_trace, plot_logit_lens

model = load_model("gpt2", device="cpu")  # TransformerLens backend by default
print(model.name, model.backend, f"{model.n_layers} layers")

# %% [markdown]
# ## Logit lens
#
# Read, at every layer, the token the model is currently betting on. Watch the
# answer's probability climb across depth.

# %%
lens = logit_lens(model, "Water is made of hydrogen and", " oxygen")
print(f"final P({lens.answer!r}) = {lens.final_prob:.3f}")
print(f"becomes top-1 at layer {lens.crossover_layer}")
for layer, prob, top in zip(lens.layers, lens.answer_prob, lens.top_token, strict=True):
    print(f"  L{layer:>2}  P={prob:.3f}  top-1={top!r}")

plot_logit_lens(lens, "lens_walkthrough.png")

# %% [markdown]
# ## Causal tracing
#
# Locate *where the France→Paris fact lives* by patching clean activations into a
# corrupted ("…Japan…") run, one (layer, position) at a time.

# %%
fact = FactPrompt(
    clean_prompt="The capital of France is",
    corrupt_prompt="The capital of Japan is",
    answer=" Paris",
    counterfactual_answer=" Tokyo",
    subject="France",
)
trace = causal_trace(model, fact)
print(f"clean logit-diff   = {trace.clean_metric:+.2f}  (prefers Paris)")
print(f"corrupt logit-diff = {trace.corrupt_metric:+.2f}  (prefers Tokyo)")
print(f"peak recovery {trace.peak['recovery']:.2f} at layer {trace.peak['layer']}, "
      f"token position {trace.peak['position']} ({trace.str_tokens[trace.peak['position']]!r})")

plot_causal_trace(trace, "trace_walkthrough.png")

# %% [markdown]
# ## Swap the model — same code
#
# The adapter means the only change to run all of the above on Qwen3-0.6B (a
# different architecture, via the nnsight backend) is the model name:
#
# ```python
# model = load_model("qwen3-0.6b")   # nnsight backend, runs on MPS/CPU
# ```
