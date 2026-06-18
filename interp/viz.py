"""Plotting. Matplotlib only, headless (Agg), saves PNGs.

Two figures: the logit-lens trajectory (a line) and the causal-trace heatmap (the
canonical [layer x position] recovery map).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never tries to open a window
import matplotlib.pyplot as plt  # noqa: E402

from interp.lenses import LensResult  # noqa: E402
from interp.patching import CausalTraceResult  # noqa: E402


def _clean_token(tok: str) -> str:
    """Make a token printable as an axis label."""
    return tok.replace("\n", "\\n").replace(" ", "·") or "∅"


def plot_logit_lens(result: LensResult, path: str | Path, title: str | None = None) -> str:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(result.layers, result.answer_prob, marker="o", color="#2b6cb0")
    ax.set_xlabel("layer")
    ax.set_ylabel(f"P({result.answer!r})")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    if result.crossover_layer is not None:
        ax.axvline(result.crossover_layer, color="#c53030", linestyle="--", alpha=0.7)
        ax.text(
            result.crossover_layer,
            0.95,
            f" becomes top-1 @ L{result.crossover_layer}",
            color="#c53030",
            va="top",
            fontsize=9,
        )

    ax.set_title(title or f"Logit lens: {result.prompt!r} → {result.answer!r}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_causal_trace(result: CausalTraceResult, path: str | Path, title: str | None = None) -> str:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(max(6, len(result.str_tokens) * 0.7), 6))
    im = ax.imshow(
        result.recovery,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_yticks(range(len(result.layers)))
    ax.set_yticklabels(result.layers)
    ax.set_ylabel("layer (residual stream patched)")

    labels = [_clean_token(t) for t in result.str_tokens]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("token position patched")

    # Mark the corrupted (subject) positions, where recovery is expected to peak.
    for pos in result.corrupted_positions:
        ax.get_xticklabels()[pos].set_color("#c53030")
        ax.get_xticklabels()[pos].set_fontweight("bold")

    fig.colorbar(im, ax=ax, label="recovery (0=corrupt, 1=clean)")
    ax.set_title(title or f"Causal trace: {result.answer!r} (metric={result.metric})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
