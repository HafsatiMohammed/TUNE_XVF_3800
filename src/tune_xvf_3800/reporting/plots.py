from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from ..metrics.noise import short_term_rms_db


def _mkdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def plot_waveform_with_dbfs_lines(x: np.ndarray, *, title: str, lines_dbfs: List[float], out_path: Path, max_samples: int = 32000) -> None:
    _mkdir(out_path)
    x = np.asarray(x, dtype=np.float32).flatten()[:max_samples]
    t = np.arange(x.size)

    plt.figure()
    plt.plot(t, x)
    for db in lines_dbfs:
        amp = 10 ** (db / 20.0)
        plt.axhline(+amp, linestyle="--")
        plt.axhline(-amp, linestyle="--")
        plt.text(0, amp, f"{db:.1f} dBFS", va="bottom")
    plt.title(title)
    plt.xlabel("Sample (logical)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bar_with_threshold(values: Dict[str, float], *, threshold: float, title: str, ylabel: str, out_path: Path) -> None:
    _mkdir(out_path)
    keys = list(values.keys())
    vals = [values[k] for k in keys]

    plt.figure()
    plt.bar(keys, vals)
    plt.axhline(threshold, linestyle="--")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_segments_rms(x: np.ndarray, fs_hz: int, segments: Dict[str, Tuple[float, float]], *, title: str, out_path: Path) -> None:
    """
    Plot short-term RMS(dB) with shaded segments.
    """
    _mkdir(out_path)
    env = short_term_rms_db(x, window_samples=int(fs_hz*0.02), hop_samples=int(fs_hz*0.01))  # 20ms / 10ms
    t = np.arange(env.size) * 0.01  # hop=10ms

    plt.figure()
    plt.plot(t, env)

    for name, (t0, t1) in segments.items():
        if t1 < 0:
            t1 = t[-1] if t.size else 0.0
        plt.axvspan(t0, t1, alpha=0.12)
        plt.text(t0, np.max(env) if env.size else 0.0, name, va="top")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Short-term RMS (dBFS)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_correlation(lags: np.ndarray, corr: np.ndarray, *, band: Tuple[int, int], title: str, out_path: Path) -> None:
    _mkdir(out_path)
    plt.figure()
    plt.plot(lags, corr)
    lo, hi = band
    plt.axvspan(lo, hi, alpha=0.2)
    plt.axvline(0, linestyle="--")
    plt.title(title)
    plt.xlabel("Lag (samples)  (+ means mic lags ref)")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
