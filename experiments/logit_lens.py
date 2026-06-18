"""E01 — Logit lens: watch a prediction crystallise across depth.

Config keys: ``prompt`` (str), ``answer`` (str, e.g. " Paris"), optional
``position`` (int, default -1).
"""

from __future__ import annotations

from pathlib import Path

from interp.lenses import logit_lens
from interp.models.base import ModelAdapter
from interp.registry import Experiment, ExperimentResult, register_experiment
from interp.viz import plot_logit_lens


@register_experiment("logit_lens")
class LogitLensExperiment(Experiment):
    def run(self, model: ModelAdapter) -> ExperimentResult:
        prompt = self.config["prompt"]
        answer = self.config["answer"]
        position = self.config.get("position", -1)

        result = logit_lens(model, prompt, answer, position=position)

        artifacts: dict[str, str] = {}
        if self.out_dir:
            artifacts["figure"] = plot_logit_lens(result, Path(self.out_dir) / "logit_lens.png")

        return ExperimentResult(
            name=self.name,
            config=self.config,
            metrics={
                "final_answer_prob": result.final_prob,
                "crossover_layer": result.crossover_layer,
                "answer_id": result.answer_id,
                "trajectory": result.to_dict(),
            },
            artifacts=artifacts,
        )
