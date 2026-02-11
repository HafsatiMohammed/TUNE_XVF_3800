from __future__ import annotations
import subprocess

def record_arecord(out_wav: str, *, device: str, fs_hz: int, seconds: float, channels: int = 2) -> None:
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-r", str(fs_hz),
        "-c", str(channels),
        "-d", str(seconds),
        out_wav,
    ]
    subprocess.check_call(cmd)
