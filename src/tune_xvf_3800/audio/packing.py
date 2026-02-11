from __future__ import annotations
import numpy as np

def unpack_packed_channel(x_48k: np.ndarray, pack_factor: int) -> list[np.ndarray]:
    """
    x_48k: shape (N,) float
    returns [slot0, slot1, slot2] each shape (~N/pack_factor,) at Fs=48k/pack_factor
    """
    x = np.asarray(x_48k)
    if x.ndim != 1:
        raise ValueError("Expected mono array")
    pf = int(pack_factor)
    return [x[i::pf].copy() for i in range(pf)]

def pack_logical_channels(slots: list[np.ndarray], pack_factor: int) -> np.ndarray:
    """
    slots: list of logical signals (each same length), returns interleaved 48k mono.
    """
    pf = int(pack_factor)
    if len(slots) != pf:
        raise ValueError(f"Expected {pf} slots, got {len(slots)}")
    L = min(len(s) for s in slots)
    slots = [np.asarray(s[:L]) for s in slots]
    out = np.zeros(L * pf, dtype=np.float32)
    for i, s in enumerate(slots):
        out[i::pf] = s.astype(np.float32, copy=False)
    return out

def make_packed_stereo_reference(
    ref_logical: np.ndarray,
    *,
    pack_factor: int,
    ref_physical: str,
    ref_slot: int,
    duration_pad_s: float = 0.0,
    fs_physical_hz: int = 48000,
) -> np.ndarray:
    """
    Creates 48k stereo array with packed reference placed into (L/R, slot).
    Other physical channel and other slots are zeros.

    ref_logical is assumed sampled at fs_physical/pack_factor (e.g. 16k).
    Output is shape (N48, 2) float32.
    """
    pf = int(pack_factor)
    slot = int(ref_slot)
    if not (0 <= slot < pf):
        raise ValueError("ref_slot out of range")
    phys = ref_physical.upper()
    if phys not in ("L", "R"):
        raise ValueError("ref_physical must be L or R")

    ref_logical = np.asarray(ref_logical, dtype=np.float32)
    if duration_pad_s > 0:
        pad_logical = int(round((fs_physical_hz / pf) * duration_pad_s))
        ref_logical = np.concatenate([np.zeros(pad_logical, np.float32), ref_logical])

    # build slots
    Llog = len(ref_logical)
    zeros = np.zeros(Llog, np.float32)
    slots = [zeros.copy() for _ in range(pf)]
    slots[slot] = ref_logical
    packed = pack_logical_channels(slots, pf)

    N48 = len(packed)
    stereo = np.zeros((N48, 2), dtype=np.float32)
    if phys == "L":
        stereo[:, 0] = packed
    else:
        stereo[:, 1] = packed
    return stereo
