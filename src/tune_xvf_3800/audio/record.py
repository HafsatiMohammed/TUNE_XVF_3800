from __future__ import annotations
import math
import subprocess
from typing import List


def _arecord_duration_arg(seconds: float) -> str:
    # arecord -d expects an integer number of seconds
    return str(max(1, int(math.ceil(float(seconds)))))


def build_arecord_cmd(out_wav: str, *, device: str, fs_hz: int, seconds: float, channels: int = 2) -> List[str]:
    return [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-r", str(fs_hz),
        "-c", str(channels),
        "-d", _arecord_duration_arg(seconds),
        out_wav,
    ]


def record_arecord(out_wav: str, *, device: str, fs_hz: int, seconds: float, channels: int = 2) -> None:
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-r", str(fs_hz),
        "-c", str(channels),
        "-d", _arecord_duration_arg(seconds),
        out_wav,
    ]
    subprocess.check_call(cmd)
