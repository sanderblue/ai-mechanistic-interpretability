"""The experiment plugin system.

An experiment is a small class that takes a resolved config dict and a loaded
model adapter and returns an :class:`ExperimentResult`. Registering it with the
``@register_experiment("name")`` decorator makes it runnable from the CLI. Adding
a new experiment therefore touches *only* a new file under ``experiments/`` — the
core never changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from interp.models.base import ModelAdapter


@dataclass
class ExperimentResult:
    """The structured output of an experiment run.

    ``metrics`` are scalar summaries (JSON-serialisable). ``artifacts`` map a
    short name to a file path (e.g. a saved figure). ``metadata`` is filled in by
    the runner with run provenance (see :class:`interp.config.RunMetadata`).
    """

    name: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Experiment(ABC):
    """Base class for all experiments.

    The runner sets :attr:`out_dir` to a fresh per-run directory before calling
    :meth:`run`; write any figures or arrays there and record their paths in
    ``ExperimentResult.artifacts``.
    """

    name: str = "experiment"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.out_dir: str | None = None

    @abstractmethod
    def run(self, model: ModelAdapter) -> ExperimentResult:
        """Execute the experiment against a loaded model adapter."""


_REGISTRY: dict[str, type[Experiment]] = {}


def register_experiment(name: str):
    """Class decorator that registers an :class:`Experiment` under ``name``."""

    def decorator(cls: type[Experiment]) -> type[Experiment]:
        if name in _REGISTRY:
            raise ValueError(f"Experiment {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_experiment(name: str) -> type[Experiment]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown experiment {name!r}. Available: {available_experiments()}")
    return _REGISTRY[name]


def available_experiments() -> list[str]:
    return sorted(_REGISTRY)
