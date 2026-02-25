from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from .audio.io import read_wav, write_wav
from .audio.packing import unpack_packed_channel


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

    It always saves all 6 unpacked logical channels as WAV files:
      L0/L1/L2 and R0/R1/R2
    and, when labels are provided, saves labeled alias WAV files too.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    x, fs = read_wav(capture_wav)
    L = x[:, 0]
    R = x[:, 1]
    Lslots = unpack_packed_channel(L, pack_factor)
    Rslots = unpack_packed_channel(R, pack_factor)

    slot_map = {
        "L0": Lslots[0], "L1": Lslots[1], "L2": Lslots[2],
        "R0": Rslots[0], "R1": Rslots[1], "R2": Rslots[2],
    }

    logical_fs = int(round(fs / pack_factor))

    # Always save raw slot files
    for slot_key, sig in slot_map.items():
        write_wav(str(out_dir / f"{slot_key}_{logical_fs}Hz.wav"), sig.reshape(-1, 1), logical_fs)

    # Save label aliases + return dict by label where possible
    out: dict[str, np.ndarray] = {}
    manifest: dict[str, str] = {}

    for slot_key, sig in slot_map.items():
        label = labels.get(slot_key, slot_key)
        out[label] = sig
        manifest[slot_key] = label
        if label != slot_key:
            write_wav(str(out_dir / f"{label}_{logical_fs}Hz.wav"), sig.reshape(-1, 1), logical_fs)

    save_json(out_dir / "manifest.json", {
        "capture_wav": capture_wav,
        "physical_fs_hz": fs,
        "logical_fs_hz": logical_fs,
        "pack_factor": pack_factor,
        "slot_to_label": manifest,
    })

    return out


def db_diff(a_db: float, b_db: float) -> float:
    return float(a_db - b_db)
