from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np

from .core.mux import apply_mux
from .audio.io import read_wav, write_wav
from .audio.packing import make_packed_stereo_reference
from .audio.play import build_aplay_cmd
from .audio.record import build_arecord_cmd
from .audio.streaming import play_and_record_with_cues, record_with_cues

from .phase_tools import unpack_capture_to_logical, save_json
from .reporting.analyzer import (
    analyze_ref_gain,
    analyze_mic_gain,
    analyze_aec_silencelevel,
    analyze_sys_delay,
    analyze_phase2_echo,
    analyze_phase3_usability,
)
from .reporting.report_builder import build_report


def _load_mono(path: Path, expected_fs: int) -> np.ndarray:
    x, fs = read_wav(str(path))
    if fs != expected_fs:
        raise ValueError(f"{path} must be {expected_fs} Hz, got {fs}")
    return x[:, 0].astype(np.float32, copy=False)


def _make_ref_playback_wav(ctx, ref_logical: np.ndarray, out_wav: Path) -> float:
    fs_phys = int(ctx.cfg["io"]["physical_fs_hz"])
    pf = int(ctx.cfg["io"]["pack_factor"])

    stereo48 = make_packed_stereo_reference(
        ref_logical,
        pack_factor=pf,
        ref_physical=ctx.cfg["audio"]["reference_physical"],
        ref_slot=int(ctx.cfg["audio"]["reference_slot"]),
        fs_physical_hz=fs_phys,
    )
    write_wav(str(out_wav), stereo48, fs_phys)
    return float(stereo48.shape[0]) / float(fs_phys)


def _play_and_record(ctx, *, playback_wav: Path, capture_wav: Path, seconds: float, cues=None, extra_guard_s: float = 0.6) -> None:
    fs_phys = int(ctx.cfg["io"]["physical_fs_hz"])
    aplay_cmd = build_aplay_cmd(str(playback_wav), device=ctx.cfg["audio"]["playback_device"])
    arecord_cmd = build_arecord_cmd(
        str(capture_wav),
        device=ctx.cfg["audio"]["capture_device"],
        fs_hz=fs_phys,
        seconds=float(seconds + extra_guard_s),
        channels=2,
    )
    play_and_record_with_cues(
        aplay_cmd=aplay_cmd,
        arecord_cmd=arecord_cmd,
        cues=cues or [],
        record_start_delay_s=0.2,
        timeout_s=float(seconds + extra_guard_s + 10.0),
    )


def _record_only(ctx, *, capture_wav: Path, seconds: float, cues=None) -> None:
    fs_phys = int(ctx.cfg["io"]["physical_fs_hz"])
    arecord_cmd = build_arecord_cmd(
        str(capture_wav),
        device=ctx.cfg["audio"]["capture_device"],
        fs_hz=fs_phys,
        seconds=float(seconds),
        channels=2,
    )
    record_with_cues(arecord_cmd=arecord_cmd, cues=cues or [], timeout_s=float(seconds + 10.0))


def _apply_decision(ctx, d) -> bool:
    """
    Auto-apply for phase1: returns True if changed.
    """
    if d.recommendation == "keep":
        return False
    if d.suggested_value is not None:
        ctx.host.set_param(d.name, f"{float(d.suggested_value):.8f}")
        return True
    if d.suggested_int is not None:
        ctx.host.set_param(d.name, str(int(d.suggested_int)))
        return True
    return False


# -------------------------
# PHASE 1 (auto-iterate)
# -------------------------

def run_phase1_gain_delay(ctx, assets_dir: Path, step: int | None = None) -> None:
    """
    Phase 1 in MANUAL mode (no automatic parameter changes).
    Optional: run a single step with step=1..4.
    """
    phase_dir = ctx.run_dir / "phase1_gain_delay"
    (phase_dir / "decisions").mkdir(parents=True, exist_ok=True)

    pf = int(ctx.cfg["io"]["pack_factor"])
    fs_log = int(ctx.cfg["io"]["logical_fs_hz"])

    if step is not None and step not in (1, 2, 3, 4):
        raise ValueError("phase1 step must be one of: 1,2,3,4")

    def _run_step(n: int) -> bool:
        return (step is None) or (step == n)

    # stimuli
    wn = _load_mono(assets_dir / "white_noise_0dbfs_16k_mono.wav", fs_log)
    sil = _load_mono(assets_dir / "silence_16k_mono.wav", fs_log)
    corr = _load_mono(assets_dir / "silence_white_noise_silence_16k_mono.wav", fs_log)

    if _run_step(1):
        print("[PHASE1][STEP 1/4] REF_GAIN analysis")
        preset = ctx.mux_presets["phase1_ref_pre_post_only"]
        apply_mux(ctx.host, preset)

        play_wav = phase_dir / "01_refgain_play.wav"
        cap_wav = phase_dir / "01_refgain_cap.wav"
        dur = _make_ref_playback_wav(ctx, wn, play_wav)
        _play_and_record(ctx, playback_wav=play_wav, capture_wav=cap_wav, seconds=dur)

        sigs = unpack_capture_to_logical(str(cap_wav), pack_factor=pf, labels=preset.labels, out_dir=phase_dir / "signals_refgain")
        d = analyze_ref_gain(
            ref_pre=sigs["ref_pregain"],
            ref_post=sigs["ref_postgain"],
            ref_gain_current=ctx.host.get_float(ctx.cfg["params"]["ref_gain_param"]),
            target_peak_dbfs=float(ctx.cfg["tuning"]["ref_peak_target_dbfs"]),
            max_peak_dbfs=float(ctx.cfg["tuning"]["ref_peak_max_dbfs"]),
            out_dir=phase_dir / "plots" / "ref_gain",
            min_ref_pre_peak_dbfs=float(ctx.cfg.get("tuning", {}).get("ref_pre_min_peak_dbfs", -20.0)),
            max_gain_suggestion=float(ctx.cfg.get("tuning", {}).get("ref_gain_max_suggestion_x", 8.0)),
        )
        save_json(phase_dir / "decisions" / "01_ref_gain.json", d.to_dict())
        print(f"[PHASE1][STEP 1/4] status={d.status} recommendation={d.recommendation}")

    # use phase1_gain_delay preset for steps 2/3/4
    preset = ctx.mux_presets["phase1_gain_delay"]

    if _run_step(2):
        print("[PHASE1][STEP 2/4] MIC_GAIN analysis")
        apply_mux(ctx.host, preset)

        play_wav = phase_dir / "02_micgain_play.wav"
        cap_wav = phase_dir / "02_micgain_cap.wav"
        dur = _make_ref_playback_wav(ctx, wn, play_wav)
        _play_and_record(ctx, playback_wav=play_wav, capture_wav=cap_wav, seconds=dur)

        sigs = unpack_capture_to_logical(str(cap_wav), pack_factor=pf, labels=preset.labels, out_dir=phase_dir / "signals_micgain")
        mics = {k: sigs[k] for k in ["mic0_postgain", "mic1_postgain", "mic2_postgain", "mic3_postgain"]}
        d = analyze_mic_gain(
            mics_post=mics,
            ref_post=sigs["ref_postgain"],
            mic_gain_current=ctx.host.get_float(ctx.cfg["params"]["mic_gain_param"]),
            required_margin_db=float(ctx.cfg["tuning"]["mic_margin_db"]),
            out_dir=phase_dir / "plots" / "mic_gain",
        )
        save_json(phase_dir / "decisions" / "02_mic_gain.json", d.to_dict())
        print(f"[PHASE1][STEP 2/4] status={d.status} recommendation={d.recommendation}")

    if _run_step(3):
        print("[PHASE1][STEP 3/4] AEC_AECSILENCELEVEL analysis")
        apply_mux(ctx.host, preset)

        play_wav = phase_dir / "03_silence_play.wav"
        cap_wav = phase_dir / "03_silence_cap.wav"
        dur = _make_ref_playback_wav(ctx, sil, play_wav)
        _play_and_record(ctx, playback_wav=play_wav, capture_wav=cap_wav, seconds=dur)

        sigs = unpack_capture_to_logical(str(cap_wav), pack_factor=pf, labels=preset.labels, out_dir=phase_dir / "signals_silence")
        d = analyze_aec_silencelevel(
            ref_post_silence=sigs["ref_postgain"],
            logical_fs_hz=fs_log,
            margin_percent=float(ctx.cfg["tuning"]["silence_level_margin_percent"]),
            out_dir=phase_dir / "plots" / "silence_level",
        )
        save_json(phase_dir / "decisions" / "03_aec_silencelevel.json", d.to_dict())
        print(f"[PHASE1][STEP 3/4] status={d.status} recommendation={d.recommendation} suggested={d.suggested_value}")

    if _run_step(4):
        print("[PHASE1][STEP 4/4] SYS_DELAY analysis")
        apply_mux(ctx.host, preset)

        play_wav = phase_dir / "04_sysdelay_play.wav"
        cap_wav = phase_dir / "04_sysdelay_cap.wav"
        dur = _make_ref_playback_wav(ctx, corr, play_wav)
        _play_and_record(ctx, playback_wav=play_wav, capture_wav=cap_wav, seconds=dur)

        sigs = unpack_capture_to_logical(str(cap_wav), pack_factor=pf, labels=preset.labels, out_dir=phase_dir / "signals_sysdelay")
        mics = {k: sigs[k] for k in ["mic0_postgain", "mic1_postgain", "mic2_postgain", "mic3_postgain"]}
        d = analyze_sys_delay(
            mics_post=mics,
            ref_loopback=sigs["ref_loopback"],
            logical_fs_hz=fs_log,
            sys_delay_current=ctx.host.get_int(ctx.cfg["params"]["sys_delay_param"]),
            target_samples=int(ctx.cfg["tuning"]["sys_delay_target_samples_logical"]),
            max_allowed_samples=int(ctx.cfg["tuning"]["sys_delay_max_samples_logical"]),
            out_dir=phase_dir / "plots" / "sys_delay",
        )
        save_json(phase_dir / "decisions" / "04_sys_delay.json", d.to_dict())
        print(f"[PHASE1][STEP 4/4] status={d.status} recommendation={d.recommendation} suggested={d.suggested_int}")

    current = {
        "AUDIO_MGR_REF_GAIN": ctx.host.get_raw("AUDIO_MGR_REF_GAIN"),
        "AUDIO_MGR_MIC_GAIN": ctx.host.get_raw("AUDIO_MGR_MIC_GAIN"),
        "AEC_AECSILENCELEVEL": ctx.host.get_raw("AEC_AECSILENCELEVEL"),
        "AUDIO_MGR_SYS_DELAY": ctx.host.get_raw("AUDIO_MGR_SYS_DELAY"),
    }
    save_json(phase_dir / "phase1_current_params.json", current)

    build_report(ctx.run_dir)


# -------------------------
# PHASE 2 (cues + objective echo decisions using beam+mic+resid)
# -------------------------

def run_phase2_aec_effectiveness(ctx, assets_dir: Path) -> None:
    phase_dir = ctx.run_dir / "phase2_aec_effectiveness"
    (phase_dir / "decisions").mkdir(parents=True, exist_ok=True)

    pf = int(ctx.cfg["io"]["pack_factor"])
    fs_log = int(ctx.cfg["io"]["logical_fs_hz"])

    preset = ctx.mux_presets["phase2_aec_effectiveness"]
    apply_mux(ctx.host, preset)

    speech = _load_mono(assets_dir / "far_end_speech_16k_mono.wav", fs_log)

    play_wav = phase_dir / "play_farend.wav"
    cap_wav = phase_dir / "cap_farend.wav"
    dur = _make_ref_playback_wav(ctx, speech, play_wav)

    seg = ctx.cfg["segments"]["phase2"]
    cues = [
        (seg["far_end_only_1"][0], "FAR-END only: stay QUIET."),
        (seg["double_talk"][0], "DOUBLE TALK: TALK NOW near the robot!"),
        (seg["double_talk"][1], "STOP talking: FAR-END only again."),
        (max(dur - 3.0, 0.0), "Finishing in 3 seconds..."),
    ]

    _play_and_record(ctx, playback_wav=play_wav, capture_wav=cap_wav, seconds=dur, cues=cues, extra_guard_s=1.0)

    sigs = unpack_capture_to_logical(str(cap_wav), pack_factor=pf, labels=preset.labels, out_dir=phase_dir / "signals")

    decisions = analyze_phase2_echo(
        signals=sigs,
        logical_fs_hz=fs_log,
        segments={k: tuple(v) for k, v in seg.items()},
        targets=ctx.cfg["tuning"]["phase2"],
        out_dir=phase_dir / "plots" / "phase2",
    )

    for i, d in enumerate(decisions):
        save_json(phase_dir / "decisions" / f"{i:02d}_{d.name}.json", d.to_dict())

    build_report(ctx.run_dir)


# -------------------------
# PHASE 3 (talk/silence cues + objective usability decisions)
# -------------------------

def run_phase3_agc_noise_attns(ctx, assets_dir: Path | None = None) -> None:
    phase_dir = ctx.run_dir / "phase3_agc_noise_attns"
    (phase_dir / "decisions").mkdir(parents=True, exist_ok=True)

    pf = int(ctx.cfg["io"]["pack_factor"])
    fs_log = int(ctx.cfg["io"]["logical_fs_hz"])

    preset = ctx.mux_presets["phase3_agc_noise_attns"]
    apply_mux(ctx.host, preset)

    seg = ctx.cfg["segments"]["phase3"]
    total = seg["talk2"][1] if seg["talk2"][1] > 0 else 35.0

    cues = [
        (seg["talk1"][0], "TALK NOW (normal voice)."),
        (seg["silence"][0], "STOP. Be silent (fan/noise measurement)."),
        (seg["talk2"][0], "TALK NOW (quiet voice / farther)."),
        (max(total - 1.0, 0.0), "Finishing..."),
    ]

    cap_wav = phase_dir / "cap_phase3.wav"
    _record_only(ctx, capture_wav=cap_wav, seconds=float(total), cues=cues)

    sigs = unpack_capture_to_logical(str(cap_wav), pack_factor=pf, labels=preset.labels, out_dir=phase_dir / "signals")

    decisions = analyze_phase3_usability(
        signals=sigs,
        logical_fs_hz=fs_log,
        segments={k: tuple(v) for k, v in seg.items()},
        targets=ctx.cfg["tuning"]["phase3"],
        out_dir=phase_dir / "plots" / "phase3",
    )

    for i, d in enumerate(decisions):
        save_json(phase_dir / "decisions" / f"{i:02d}_{d.name}.json", d.to_dict())

    build_report(ctx.run_dir)
