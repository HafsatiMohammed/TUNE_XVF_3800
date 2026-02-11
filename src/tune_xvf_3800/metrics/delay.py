from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class DelayResult:
    lag_samples: int
    lags: np.ndarray
    corr: np.ndarray


def _nextpow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def estimate_delay(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs_hz: int,
    method: str = "gcc_phat",
    max_lag_seconds: float = 0.05,
    pack_factor=None,
) -> DelayResult:
    """
    Returns lag in samples (positive => x lags y; i.e., x occurs AFTER y).
    """
    x = np.asarray(x, dtype=np.float32).flatten()
    y = np.asarray(y, dtype=np.float32).flatten()
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    max_lag = int(round(max_lag_seconds * fs_hz))
    nfft = _nextpow2(2*n)

    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    R = X * np.conj(Y)

    if method == "gcc_phat":
        denom = np.abs(R)
        denom[denom < 1e-12] = 1e-12
        R = R / denom

    cc = np.fft.irfft(R, nfft)
    cc = np.concatenate((cc[-(n-1):], cc[:n]))
    lags = np.arange(-(n-1), n, dtype=np.int32)

    # restrict to +/- max_lag
    keep = (lags >= -max_lag) & (lags <= max_lag)
    lags_k = lags[keep]
    cc_k = cc[keep]

    idx = int(np.argmax(np.abs(cc_k)))
    lag = int(lags_k[idx])

    return DelayResult(lag_samples=lag, lags=lags_k, corr=cc_k)
