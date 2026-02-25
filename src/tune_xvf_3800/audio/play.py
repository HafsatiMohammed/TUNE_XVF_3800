from __future__ import annotations
import subprocess
import numpy as np
from typing import List
from .io import write_wav

def play_aplay(wav_path: str, *, device: str) -> None:
    cmd = ["aplay", "-D", device, wav_path]
    subprocess.check_call(cmd)

def play_array_via_aplay(tmp_wav_path: str, x: np.ndarray, fs: int, *, device: str) -> None:
    write_wav(tmp_wav_path, x, fs)
    play_aplay(tmp_wav_path, device=device)


def build_aplay_cmd(wav_path: str, *, device: str) -> List[str]:
    return ["aplay", "-D", device, wav_path]