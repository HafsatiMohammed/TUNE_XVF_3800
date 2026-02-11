from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Levels:
    peak: float
    peak_dbfs: float
    rms: float
    rms_dbfs: float
    clipped: bool
    clip_fraction: float


def _db(x: float, eps: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(x, eps)))


def compute_levels(x: np.ndarray, clip_threshold: float = 0.999) -> Levels:
    x = np.asarray(x, dtype=np.float32).flatten()
    if x.size == 0:
        return Levels(0.0, -120.0, 0.0, -120.0, False, 0.0)

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    clip_mask = np.abs(x) >= clip_threshold
    clip_fraction = float(np.mean(clip_mask.astype(np.float32)))
    clipped = bool(clip_fraction > 0.0)

    return Levels(
        peak=peak,
        peak_dbfs=_db(peak),
        rms=rms,
        rms_dbfs=_db(rms),
        clipped=clipped,
        clip_fraction=clip_fraction,
    )
