"""Runtime configuration: device/dtype selection, seeding, and run provenance.

Everything that touches the machine (which accelerator, which precision) or that
makes a run reproducible (seeds, captured environment) lives here, so the rest of
the code never has to special-case hardware.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import torch


def detect_device(preferred: str | None = None) -> str:
    """Resolve a torch device string.

    ``preferred`` (or ``"auto"``/``None``) triggers auto-detection in the order
    CUDA → MPS → CPU. On this project's target laptop (Apple silicon) that means
    MPS; CUDA is reserved for the multi-GPU rig the 14B work will run on.
    """
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype:
    """Pick a sensible dtype for a device.

    We use float32 on MPS and CPU: at the 0.6B–small scale it is fast enough, and
    it sidesteps the half-precision rough edges that still exist on MPS (which
    would quietly corrupt a logit-lens read). bfloat16 is reserved for CUDA, where
    it is both well-supported and worth the memory savings for the larger models.
    """
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch.

    Note (documented honestly): a few MPS kernels are not bit-for-bit
    deterministic across runs, so this fixes *sampling and corruption noise* but
    does not guarantee identical floating-point outputs on Apple silicon.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _version(pkg: str) -> str | None:
    try:
        return importlib_metadata.version(pkg)
    except importlib_metadata.PackageNotFoundError:
        return None


@dataclass
class RunMetadata:
    """Provenance captured next to every result, so a run can be reproduced."""

    timestamp: str
    model: str
    backend: str
    device: str
    dtype: str
    seed: int
    python_version: str
    platform: str
    git_sha: str | None
    versions: dict[str, str | None] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        *,
        model: str,
        backend: str,
        device: str,
        dtype: torch.dtype,
        seed: int,
    ) -> RunMetadata:
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model=model,
            backend=backend,
            device=device,
            dtype=str(dtype).replace("torch.", ""),
            seed=seed,
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            git_sha=_git_sha(),
            versions={
                pkg: _version(pkg)
                for pkg in ("torch", "transformers", "transformer-lens", "nnsight", "numpy")
            },
        )
