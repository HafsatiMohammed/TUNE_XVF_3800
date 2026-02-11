from __future__ import annotations
import numpy as np


def mean_power(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).flatten()
    if x.size == 0:
        return 0.0
    return float(np.mean(x * x))


def power_dbfs(x: np.ndarray, eps: float = 1e-20) -> float:
    p = mean_power(x)
    return float(10.0 * np.log10(max(p, eps)))


def estimate_silence_level_for_aec(
    ref_post_silence: np.ndarray,
    *,
    fs_hz: int,
    safety_margin_percent: float = 2.0,
    channel=None,
) -> float:
    p = mean_power(ref_post_silence)
    return float(p * (1.0 + safety_margin_percent / 100.0))


def short_term_rms_db(x: np.ndarray, window_samples: int = 256, hop_samples: int = 128, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).flatten()
    if x.size < window_samples:
        rms = np.sqrt(np.mean(x * x) + eps)
        return np.array([20*np.log10(rms + eps)], dtype=np.float32)

    out = []
    for i in range(0, x.size - window_samples + 1, hop_samples):
        w = x[i:i+window_samples]
        rms = np.sqrt(np.mean(w*w) + eps)
        out.append(20.0*np.log10(rms + eps))
    return np.asarray(out, dtype=np.float32)
