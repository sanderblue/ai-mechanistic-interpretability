"""Importing this package registers every experiment with the registry.

Add a new experiment by dropping a module here that calls
``@register_experiment(...)`` and importing it below.
"""

from experiments import causal_tracing, logit_lens  # noqa: F401

__all__ = ["causal_tracing", "logit_lens"]
