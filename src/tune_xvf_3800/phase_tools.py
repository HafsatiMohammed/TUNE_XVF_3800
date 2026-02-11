from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from .audio.io import read_wav, write_wav
from .audio.packing import unpack_packed_channel
from .metrics.levels import compute_levels
from .metrics.noise import estimate_silence_level_for_aec, compute_noise_metrics
from .metrics.delay import estimate_delay

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def unpack_capture_to_logical(
    capture_wav: str,
    *,
    pack_factor: int,
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, np.ndarray]:
    """
    capture_wav is 48k stereo with packed slots.
    Returns dict label -> logical_signal (Fs=48k/pack_factor).
    Also writes each signal as wav in out_dir.
    """
    x, fs = read_wav(capture_wav)
    if fs != 48000:
        # still works, but your pack mapping assumes 48k physical
        pass
    L = x[:, 0]
    R = x[:, 1]
    Lslots = unpack_packed_channel(L, pack_factor)
    Rslots = unpack_packed_channel(R, pack_factor)

    # default mapping keys
    slot_map = {
        "L0": Lslots[0], "L1": Lslots[1], "L2": Lslots[2],
        "R0": Rslots[0], "R1": Rslots[1], "R2": Rslots[2],
    }

    logical_fs = int(round(fs / pack_factor))
    out: dict[str, np.ndarray] = {}
    for slot_key, sig in slot_map.items():
        label = labels.get(slot_key, slot_key)
        out[label] = sig
        write_wav(str(out_dir / f"{label}_{logical_fs}Hz.wav"), sig.reshape(-1, 1), logical_fs)
    return out

def db_diff(a_db: float, b_db: float) -> float:
    return float(a_db - b_db)
