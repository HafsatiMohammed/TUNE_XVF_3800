from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def segment_indices(segments: Dict[str, Tuple[float, float]], fs_hz: int, n_samples: int) -> Dict[str, slice]:
    out = {}
    for name, (t0, t1) in segments.items():
        i0 = int(max(0, round(t0 * fs_hz)))
        if t1 < 0:
            i1 = n_samples
        else:
            i1 = int(min(n_samples, round(t1 * fs_hz)))
        if i1 <= i0:
            i1 = min(n_samples, i0 + 1)
        out[name] = slice(i0, i1)
    return out
