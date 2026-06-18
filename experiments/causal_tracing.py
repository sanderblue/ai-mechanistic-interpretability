"""E02 — Causal tracing: locate where a fact lives via activation patching.

Config keys: ``fact`` (mapping with ``clean_prompt``, ``corrupt_prompt``,
``answer``, ``counterfactual_answer``, optional ``subject``), optional ``metric``
("logit_diff" | "answer_prob").
"""

from __future__ import annotations

from pathlib import Path

from interp.models.base import ModelAdapter
from interp.patching import causal_trace
from interp.prompts import FactPrompt, load_facts
from interp.registry import Experiment, ExperimentResult, register_experiment
from interp.viz import plot_causal_trace


@register_experiment("causal_tracing")
class CausalTracingExperiment(Experiment):
    def _resolve_fact(self) -> FactPrompt:
        """A fact is given either inline (``fact:``) or by id (``fact_id:``)."""
        if "fact_id" in self.config:
            facts_path = self.config.get("facts_path", "data/facts.yaml")
            return load_facts(facts_path)[self.config["fact_id"]]
        return FactPrompt(**self.config["fact"])

    def run(self, model: ModelAdapter) -> ExperimentResult:
        fact = self._resolve_fact()
        metric = self.config.get("metric", "logit_diff")

        result = causal_trace(model, fact, metric=metric)

        artifacts: dict[str, str] = {}
        if self.out_dir:
            artifacts["figure"] = plot_causal_trace(result, Path(self.out_dir) / "causal_trace.png")

        return ExperimentResult(
            name=self.name,
            config=self.config,
            metrics={
                "clean_metric": result.clean_metric,
                "corrupt_metric": result.corrupt_metric,
                "peak": result.peak,
                "trace": result.to_dict(),
            },
            artifacts=artifacts,
        )
