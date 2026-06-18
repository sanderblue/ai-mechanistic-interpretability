"""Command-line runner: ``python -m interp.run <experiment> --config <yaml>``.

A run is fully described by its config file, so it is reproducible: the runner
resolves the model/seed/device, executes the experiment, and writes the resolved
config, a JSON result, and any figures to ``outputs/<experiment>/<tag>/`` together
with captured environment provenance.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime
from pathlib import Path

import yaml

import experiments  # noqa: F401  -- importing registers all experiments
from interp.config import RunMetadata, default_dtype, detect_device, set_seed
from interp.models import load_model
from interp.registry import available_experiments, get_experiment


def make_out_dir(experiment: str, tag: str | None = None) -> Path:
    stamp = tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path("outputs") / experiment / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interp.run",
        description="Run a mechanistic-interpretability experiment from a config file.",
    )
    parser.add_argument("experiment", help=f"experiment to run; one of {available_experiments()}")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--model", help="override the model name in the config")
    parser.add_argument("--backend", help="override backend (transformer_lens | nnsight | auto)")
    parser.add_argument("--device", help="override device (cpu | mps | cuda | auto)")
    parser.add_argument("--seed", type=int, help="override seed")
    parser.add_argument("--tag", help="name this run's output dir (default: timestamp)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_name = args.model or cfg["model"]
    backend = args.backend or cfg.get("backend", "auto")
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    device = detect_device(args.device or cfg.get("device"))
    dtype = default_dtype(device)

    set_seed(seed)
    print(
        f"[interp] {args.experiment} | model={model_name} backend={backend} "
        f"device={device} dtype={str(dtype).replace('torch.', '')} seed={seed}"
    )

    model = load_model(model_name, backend=backend, device=device, dtype=dtype)
    print(
        f"[interp] loaded {model.name} via {model.backend} "
        f"({model.n_layers} layers, d_model={model.d_model})"
    )

    experiment = get_experiment(args.experiment)(cfg)
    out_dir = make_out_dir(args.experiment, args.tag)
    experiment.out_dir = str(out_dir)
    result = experiment.run(model)

    result.metadata = dataclasses.asdict(
        RunMetadata.capture(
            model=model_name, backend=model.backend, device=device, dtype=dtype, seed=seed
        )
    )

    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "result.json").write_text(
        json.dumps(dataclasses.asdict(result), indent=2, default=str)
    )

    print(f"[interp] wrote {out_dir / 'result.json'}")
    for key, path in result.artifacts.items():
        print(f"[interp] artifact {key}: {path}")
    summary = {k: v for k, v in result.metrics.items() if not isinstance(v, (dict, list))}
    print(f"[interp] summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
