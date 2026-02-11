from __future__ import annotations
import soundfile as sf
import numpy as np

def read_wav(path: str) -> tuple[np.ndarray, int]:
    x, fs = sf.read(path, always_2d=True)
    return x.astype(np.float32, copy=False), int(fs)

def write_wav(path: str, x: np.ndarray, fs: int) -> None:
    sf.write(path, x, fs)
