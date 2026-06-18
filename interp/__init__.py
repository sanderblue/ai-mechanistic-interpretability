"""interp — a small, extensible mechanistic-interpretability lab.

Public surface:
    load_model(name, backend=..., device=...) -> ModelAdapter
    Site, Patch, InputNoise            -- the intervention primitives
    logit_lens(...), causal_trace(...) -- the two core techniques
"""

from interp.lenses import logit_lens
from interp.models import InputNoise, ModelAdapter, Patch, Site, load_model
from interp.patching import causal_trace

__version__ = "0.1.0"

__all__ = [
    "load_model",
    "ModelAdapter",
    "Site",
    "Patch",
    "InputNoise",
    "logit_lens",
    "causal_trace",
    "__version__",
]
