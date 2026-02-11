from __future__ import annotations

import subprocess
import time
import threading
from typing import List, Optional, Tuple

Cue = Tuple[float, str]  # (time offset seconds, message)


def _run_cues(cues: List[Cue], t0: float) -> None:
    for t_offset, msg in sorted(cues, key=lambda x: x[0]):
        while True:
            dt = (t0 + float(t_offset)) - time.time()
            if dt <= 0:
                break
            time.sleep(min(dt, 0.05))
        stamp = time.time() - t0
        print(f"[{stamp:6.2f}s] {msg}", flush=True)


def play_and_record_with_cues(
    *,
    aplay_cmd: List[str],
    arecord_cmd: List[str],
    cues: List[Cue],
    record_start_delay_s: float = 0.2,
    timeout_s: Optional[float] = None,
) -> None:
    rec = subprocess.Popen(arecord_cmd)
    time.sleep(record_start_delay_s)

    t0 = time.time()
    cue_thread = threading.Thread(target=_run_cues, args=(cues, t0), daemon=True)
    cue_thread.start()

    play = subprocess.Popen(aplay_cmd)
    try:
        play.wait(timeout=timeout_s)
        rec.wait(timeout=timeout_s)
    finally:
        if play.poll() is None:
            play.terminate()
        if rec.poll() is None:
            rec.terminate()
        cue_thread.join(timeout=1.0)


def record_with_cues(
    *,
    arecord_cmd: List[str],
    cues: List[Cue],
    timeout_s: Optional[float] = None,
) -> None:
    t0 = time.time()
    cue_thread = threading.Thread(target=_run_cues, args=(cues, t0), daemon=True)
    cue_thread.start()

    rec = subprocess.Popen(arecord_cmd)
    try:
        rec.wait(timeout=timeout_s)
    finally:
        if rec.poll() is None:
            rec.terminate()
        cue_thread.join(timeout=1.0)
