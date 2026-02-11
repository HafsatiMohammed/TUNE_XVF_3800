from __future__ import annotations
import numpy as np


def _mean_power(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).flatten()
    if x.size == 0:
        return 0.0
    return float(np.mean(x*x))


def erle_db(mic: np.ndarray, resid: np.ndarray, eps: float = 1e-20) -> float:
    """
    ERLE proxy = 10log10(P_mic / P_resid) during far-end-only.
    """
    pm = _mean_power(mic)
    pr = _mean_power(resid)
    return float(10.0 * np.log10(max(pm, eps) / max(pr, eps)))


def resid_to_mic_db(mic: np.ndarray, resid: np.ndarray, eps: float = 1e-20) -> float:
    """
    10log10(P_resid/P_mic). More negative is better.
    """
    pm = _mean_power(mic)
    pr = _mean_power(resid)
    return float(10.0 * np.log10(max(pr, eps) / max(pm, eps)))


def power_db(x: np.ndarray, eps: float = 1e-20) -> float:
    p = _mean_power(x)
    return float(10.0 * np.log10(max(p, eps)))
