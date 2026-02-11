from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..metrics.levels import compute_levels
from ..metrics.noise import estimate_silence_level_for_aec, power_dbfs, short_term_rms_db
from ..metrics.delay import estimate_delay
from ..metrics.echo import erle_db, resid_to_mic_db, power_db
from .segments import segment_indices
from .plots import (
    plot_waveform_with_dbfs_lines,
    plot_bar_with_threshold,
    plot_segments_rms,
    plot_correlation,
)


@dataclass
class Decision:
    name: str
    status: str                      # PASS/FAIL
    reason: str
    recommendation: str              # increase/decrease/keep
    suggested_value: Optional[float] = None
    suggested_int: Optional[int] = None
    metrics: dict = None
    plot_files: List[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "recommendation": self.recommendation,
            "suggested_value": self.suggested_value,
            "suggested_int": self.suggested_int,
            "metrics": self.metrics or {},
            "plot_files": self.plot_files or [],
        }


def _gain_ratio_db(delta_db: float) -> float:
    return float(10 ** (delta_db / 20.0))


# --------------------------
# Phase 1 analyzers
# --------------------------

def analyze_ref_gain(
    *,
    ref_pre: np.ndarray,
    ref_post: np.ndarray,
    ref_gain_current: float,
    target_peak_dbfs: float,
    max_peak_dbfs: float,
    out_dir: Path,
) -> Decision:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    m_pre = compute_levels(ref_pre)
    m_post = compute_levels(ref_post)

    p = out_dir / "ref_post_waveform.png"
    plot_waveform_with_dbfs_lines(
        ref_post,
        title="Reference post-gain (logical)",
        lines_dbfs=[0.0, max_peak_dbfs, target_peak_dbfs],
        out_path=p,
    )
    plots.append(str(p))

    if m_post.clipped or (m_post.peak_dbfs > max_peak_dbfs):
        delta_db = target_peak_dbfs - m_post.peak_dbfs
        suggested = max(ref_gain_current * _gain_ratio_db(delta_db), 1e-9)
        return Decision(
            name="AUDIO_MGR_REF_GAIN",
            status="FAIL",
            reason=f"Ref post too hot: peak={m_post.peak_dbfs:.2f} dBFS clipped={m_post.clipped}.",
            recommendation="decrease",
            suggested_value=float(suggested),
            metrics={"ref_pre": asdict(m_pre), "ref_post": asdict(m_post)},
            plot_files=plots,
        )

    if m_post.peak_dbfs < target_peak_dbfs:
        delta_db = target_peak_dbfs - m_post.peak_dbfs
        suggested = ref_gain_current * _gain_ratio_db(delta_db)
        return Decision(
            name="AUDIO_MGR_REF_GAIN",
            status="FAIL",
            reason=f"Ref post too low: peak={m_post.peak_dbfs:.2f} dBFS target={target_peak_dbfs:.2f}.",
            recommendation="increase",
            suggested_value=float(suggested),
            metrics={"ref_pre": asdict(m_pre), "ref_post": asdict(m_post)},
            plot_files=plots,
        )

    return Decision(
        name="AUDIO_MGR_REF_GAIN",
        status="PASS",
        reason=f"Ref post within band: peak={m_post.peak_dbfs:.2f} dBFS.",
        recommendation="keep",
        suggested_value=float(ref_gain_current),
        metrics={"ref_pre": asdict(m_pre), "ref_post": asdict(m_post)},
        plot_files=plots,
    )


def analyze_mic_gain(
    *,
    mics_post: Dict[str, np.ndarray],
    ref_post: np.ndarray,
    mic_gain_current: float,
    required_margin_db: float,
    out_dir: Path,
) -> Decision:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    ref_m = compute_levels(ref_post)
    mic_m = {k: compute_levels(v) for k, v in mics_post.items()}

    margins = {k: ref_m.peak_dbfs - mic_m[k].peak_dbfs for k in mic_m}
    worst = min(margins, key=lambda k: margins[k])
    worst_margin = float(margins[worst])

    p = out_dir / "mic_margins.png"
    plot_bar_with_threshold(
        margins,
        threshold=required_margin_db,
        title="Mic peak margin (ref_peak - mic_peak)",
        ylabel="Margin (dB)",
        out_path=p,
    )
    plots.append(str(p))

    if worst_margin < required_margin_db:
        needed_db = required_margin_db - worst_margin
        suggested = max(mic_gain_current * _gain_ratio_db(-needed_db), 1e-9)
        return Decision(
            name="AUDIO_MGR_MIC_GAIN",
            status="FAIL",
            reason=f"{worst} violates margin: {worst_margin:.2f} dB < {required_margin_db:.2f} dB.",
            recommendation="decrease",
            suggested_value=float(suggested),
            metrics={
                "ref_post": asdict(ref_m),
                "mic_levels": {k: asdict(v) for k, v in mic_m.items()},
                "margins_db": margins,
                "worst_mic": worst,
            },
            plot_files=plots,
        )

    return Decision(
        name="AUDIO_MGR_MIC_GAIN",
        status="PASS",
        reason=f"All mics meet margin. Worst={worst} margin={worst_margin:.2f} dB.",
        recommendation="keep",
        suggested_value=float(mic_gain_current),
        metrics={
            "ref_post": asdict(ref_m),
            "mic_levels": {k: asdict(v) for k, v in mic_m.items()},
            "margins_db": margins,
            "worst_mic": worst,
        },
        plot_files=plots,
    )


def analyze_aec_silencelevel(
    *,
    ref_post_silence: np.ndarray,
    logical_fs_hz: int,
    margin_percent: float,
    out_dir: Path,
) -> Decision:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    thr = estimate_silence_level_for_aec(ref_post_silence, fs_hz=logical_fs_hz, safety_margin_percent=margin_percent)

    p = out_dir / "ref_post_silence_rms.png"
    plot_segments_rms(
        ref_post_silence,
        fs_hz=logical_fs_hz,
        segments={"silence": (0.0, -1.0)},
        title="Ref post during injected silence (RMS envelope)",
        out_path=p,
    )
    plots.append(str(p))

    return Decision(
        name="AEC_AECSILENCELEVEL",
        status="PASS",
        reason="Set from measured ref-line noise power during injected digital silence.",
        recommendation="keep",
        suggested_value=float(thr),
        metrics={"computed_threshold": float(thr), "margin_percent": float(margin_percent)},
        plot_files=plots,
    )


def analyze_sys_delay(
    *,
    mics_post: Dict[str, np.ndarray],
    ref_loopback: np.ndarray,
    logical_fs_hz: int,
    sys_delay_current: int,
    target_samples: int,
    max_allowed_samples: int,
    out_dir: Path,
) -> Decision:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    delays = {}
    worst_corr = None

    for k, mic in mics_post.items():
        d = estimate_delay(mic, ref_loopback, fs_hz=logical_fs_hz, method="gcc_phat", max_lag_seconds=0.05)
        delays[k] = int(d.lag_samples)
        if worst_corr is None:
            worst_corr = d

    worst_mic = min(delays, key=lambda k: delays[k])
    min_lag = int(delays[worst_mic])

    if worst_corr is not None:
        p = out_dir / "mic_ref_correlation.png"
        plot_correlation(
            worst_corr.lags,
            worst_corr.corr,
            band=(0, max_allowed_samples),
            title="Mic vs Ref loopback correlation (band=causal & <= max)",
            out_path=p,
        )
        plots.append(str(p))

    # Decision & suggested new SYS_DELAY (negative preferred)
    if min_lag < 0:
        change = int(round(target_samples - min_lag))
        suggested = sys_delay_current - change
        if suggested > 0:
            suggested = 0
        return Decision(
            name="AUDIO_MGR_SYS_DELAY",
            status="FAIL",
            reason=f"Acausal: worst={worst_mic} lag={min_lag} samples (mic leads ref).",
            recommendation="decrease",  # more negative
            suggested_int=int(suggested),
            metrics={"delays_samples": delays, "worst_mic": worst_mic, "min_lag": min_lag},
            plot_files=plots,
        )

    if min_lag > max_allowed_samples:
        change = int(round(min_lag - target_samples))
        suggested = sys_delay_current + change
        if suggested > 0:
            suggested = 0
        return Decision(
            name="AUDIO_MGR_SYS_DELAY",
            status="FAIL",
            reason=f"Too delayed: worst={worst_mic} lag={min_lag} > {max_allowed_samples}.",
            recommendation="increase",  # less negative
            suggested_int=int(suggested),
            metrics={"delays_samples": delays, "worst_mic": worst_mic, "min_lag": min_lag},
            plot_files=plots,
        )

    return Decision(
        name="AUDIO_MGR_SYS_DELAY",
        status="PASS",
        reason=f"Causal and within max. Worst={worst_mic} lag={min_lag}.",
        recommendation="keep",
        suggested_int=int(sys_delay_current),
        metrics={"delays_samples": delays, "worst_mic": worst_mic, "min_lag": min_lag},
        plot_files=plots,
    )


# --------------------------
# Phase 2 analyzer (BOTH: beam + mic + residual)
# --------------------------

def analyze_phase2_echo(
    *,
    signals: Dict[str, np.ndarray],
    logical_fs_hz: int,
    segments: Dict[str, Tuple[float, float]],
    targets: dict,
    out_dir: Path,
) -> List[Decision]:
    """
    Expects signals:
      - beam_autoselect
      - mic0_postgain (and optionally mic1_postgain)
      - ref_postgain
      - resid_mic0 (and optionally resid_mic1)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions: List[Decision] = []

    # segment slices
    any_sig = next(iter(signals.values()))
    idx = segment_indices(segments, logical_fs_hz, len(any_sig))

    # plots: RMS envelopes with segment shading
    if "beam_autoselect" in signals:
        p = out_dir / "beam_rms_segments.png"
        plot_segments_rms(signals["beam_autoselect"], logical_fs_hz, segments, title="Beam RMS with segments", out_path=p)
    if "mic0_postgain" in signals:
        p = out_dir / "mic0_rms_segments.png"
        plot_segments_rms(signals["mic0_postgain"], logical_fs_hz, segments, title="Mic0 RMS with segments", out_path=p)

    # Compute objective echo metrics on far-end-only
    fe = idx["far_end_only_1"]
    dt = idx["double_talk"]

    # Validate operator spoke: mic energy should rise in DT
    mic0_fe_db = power_db(signals["mic0_postgain"][fe])
    mic0_dt_db = power_db(signals["mic0_postgain"][dt])
    near_end_presence_db = mic0_dt_db - mic0_fe_db

    # Beam should also rise in DT (if not overly suppressed)
    beam_fe_db = power_db(signals["beam_autoselect"][fe])
    beam_dt_db = power_db(signals["beam_autoselect"][dt])
    beam_rise_db = beam_dt_db - beam_fe_db

    # Residual effectiveness (ERLE proxy)
    erle0 = erle_db(signals["mic0_postgain"][fe], signals["resid_mic0"][fe]) if ("resid_mic0" in signals) else None
    r2m0 = resid_to_mic_db(signals["mic0_postgain"][fe], signals["resid_mic0"][fe]) if ("resid_mic0" in signals) else None

    # thresholds
    erle_target = float(targets["erle_target_db"])
    r2m_max = float(targets["resid_to_mic_max_db"])
    near_presence_min = float(targets["near_end_presence_min_db"])
    dt_beam_rise_min = float(targets["dt_beam_rise_min_db"])

    # Decision A: Check near-end presence (operator actually spoke)
    if near_end_presence_db < near_presence_min:
        decisions.append(Decision(
            name="PHASE2_NEAR_END_PRESENCE",
            status="FAIL",
            reason=f"Near-end presence too small: mic0 DT-FE = {near_end_presence_db:.1f} dB < {near_presence_min:.1f} dB. (Operator likely didn’t speak loud enough.)",
            recommendation="keep",
            metrics={"mic0_fe_db": mic0_fe_db, "mic0_dt_db": mic0_dt_db, "near_end_presence_db": near_end_presence_db},
            plot_files=[str(out_dir / "mic0_rms_segments.png")],
        ))
        # still continue; echo metrics may be valid but DT suppression metrics are not

    else:
        decisions.append(Decision(
            name="PHASE2_NEAR_END_PRESENCE",
            status="PASS",
            reason=f"Near-end present: mic0 DT-FE = {near_end_presence_db:.1f} dB.",
            recommendation="keep",
            metrics={"near_end_presence_db": near_end_presence_db},
            plot_files=[str(out_dir / "mic0_rms_segments.png")],
        ))

    # Decision B: Echo effectiveness (ERLE / residual-to-mic)
    if erle0 is not None and r2m0 is not None:
        ok_erle = erle0 >= erle_target
        ok_r2m = r2m0 <= r2m_max
        if ok_erle and ok_r2m:
            decisions.append(Decision(
                name="PHASE2_ECHO_EFFECTIVENESS_MIC0",
                status="PASS",
                reason=f"Mic0 echo good: ERLE={erle0:.1f} dB (>= {erle_target}), resid/mic={r2m0:.1f} dB (<= {r2m_max}).",
                recommendation="keep",
                metrics={"erle_db": erle0, "resid_to_mic_db": r2m0},
            ))
        else:
            # objective recommendation direction (not selecting which gamma automatically)
            # If ERLE low or resid too high => increase suppression (GAMMA_E or GAMMA_ENL depending on residual character).
            decisions.append(Decision(
                name="PHASE2_ECHO_EFFECTIVENESS_MIC0",
                status="FAIL",
                reason=f"Mic0 echo insufficient: ERLE={erle0:.1f} dB target {erle_target}, resid/mic={r2m0:.1f} dB target {r2m_max}.",
                recommendation="increase",
                metrics={"erle_db": erle0, "resid_to_mic_db": r2m0, "recommend": "Increase PP_GAMMA_E (linear) or PP_GAMMA_ENL (distorted) and re-test."},
            ))
    else:
        decisions.append(Decision(
            name="PHASE2_ECHO_EFFECTIVENESS_MIC0",
            status="FAIL",
            reason="Missing resid_mic0 or mic0_postgain; cannot compute ERLE/resid ratio.",
            recommendation="keep",
        ))

    # Decision C: Double-talk suppression check (beam should rise when you speak)
    if near_end_presence_db >= near_presence_min:
        if beam_rise_db >= dt_beam_rise_min:
            decisions.append(Decision(
                name="PHASE2_DOUBLE_TALK_NEAR_END_PRESERVATION",
                status="PASS",
                reason=f"Beam rises in DT: {beam_rise_db:.1f} dB (>= {dt_beam_rise_min}).",
                recommendation="keep",
                metrics={"beam_rise_db": beam_rise_db},
                plot_files=[str(out_dir / "beam_rms_segments.png")],
            ))
        else:
            decisions.append(Decision(
                name="PHASE2_DOUBLE_TALK_NEAR_END_PRESERVATION",
                status="FAIL",
                reason=f"Beam rise too small in DT: {beam_rise_db:.1f} dB < {dt_beam_rise_min}. Near-end likely being over-suppressed.",
                recommendation="decrease",
                metrics={
                    "beam_rise_db": beam_rise_db,
                    "recommend": "Decrease PP_DTSENSITIVE or reduce PP_GAMMA_E/ETAIL/ENL; if far-end false-detected, adjust PP_MGSCALE(min)."
                },
                plot_files=[str(out_dir / "beam_rms_segments.png")],
            ))

    return decisions


# --------------------------
# Phase 3 analyzer (AGC + noise + ATTNS)
# --------------------------

def analyze_phase3_usability(
    *,
    signals: Dict[str, np.ndarray],
    logical_fs_hz: int,
    segments: Dict[str, Tuple[float, float]],
    targets: dict,
    out_dir: Path,
) -> List[Decision]:
    """
    Uses:
      - beam_autoselect (required)
      - mic0_postgain (optional baseline)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions: List[Decision] = []

    beam = signals["beam_autoselect"]
    idx = segment_indices(segments, logical_fs_hz, len(beam))

    # plots
    p = out_dir / "beam_rms_segments.png"
    plot_segments_rms(beam, logical_fs_hz, segments, title="Beam RMS with talk/silence segments", out_path=p)

    # compute objective metrics
    talk1 = beam[idx["talk1"]]
    silence = beam[idx["silence"]]
    talk2 = beam[idx["talk2"]]

    speech_rms_db1 = compute_levels(talk1).rms_dbfs
    speech_rms_db2 = compute_levels(talk2).rms_dbfs
    pause_noise_db = compute_levels(silence).rms_dbfs

    # pumping (std of short-term RMS in silence)
    env = short_term_rms_db(silence, window_samples=int(logical_fs_hz*0.02), hop_samples=int(logical_fs_hz*0.01))
    pumping_std = float(np.std(env)) if env.size else 0.0

    target_speech = float(targets["speech_rms_target_dbfs"])
    tol = float(targets["speech_rms_tol_db"])
    pause_max = float(targets["pause_noise_max_dbfs"])
    pump_max = float(targets["pumping_std_max_db"])

    # Decision A: speech level
    if (abs(speech_rms_db1 - target_speech) <= tol) and (abs(speech_rms_db2 - target_speech) <= tol):
        decisions.append(Decision(
            name="PHASE3_SPEECH_LEVEL",
            status="PASS",
            reason=f"Speech RMS OK: talk1={speech_rms_db1:.1f}, talk2={speech_rms_db2:.1f} target={target_speech}±{tol}.",
            recommendation="keep",
            metrics={"talk1_rms_dbfs": speech_rms_db1, "talk2_rms_dbfs": speech_rms_db2},
            plot_files=[str(p)],
        ))
    else:
        rec = "increase" if (speech_rms_db1 < target_speech - tol) else "decrease"
        decisions.append(Decision(
            name="PHASE3_SPEECH_LEVEL",
            status="FAIL",
            reason=f"Speech RMS out of band: talk1={speech_rms_db1:.1f}, talk2={speech_rms_db2:.1f} target={target_speech}±{tol}.",
            recommendation=rec,
            metrics={
                "talk1_rms_dbfs": speech_rms_db1,
                "talk2_rms_dbfs": speech_rms_db2,
                "recommend": "Adjust PP_AGCDESIREDLEVEL (and then set PP_AGCGAIN default after convergence)."
            },
            plot_files=[str(p)],
        ))

    # Decision B: pause noise
    if pause_noise_db <= pause_max:
        decisions.append(Decision(
            name="PHASE3_PAUSE_NOISE",
            status="PASS",
            reason=f"Pause noise OK: {pause_noise_db:.1f} dBFS <= {pause_max:.1f}.",
            recommendation="keep",
            metrics={"pause_noise_rms_dbfs": pause_noise_db},
            plot_files=[str(p)],
        ))
    else:
        decisions.append(Decision(
            name="PHASE3_PAUSE_NOISE",
            status="FAIL",
            reason=f"Pause noise too high: {pause_noise_db:.1f} dBFS > {pause_max:.1f}.",
            recommendation="decrease",
            metrics={
                "pause_noise_rms_dbfs": pause_noise_db,
                "recommend": "Increase attenuation in silence: PP_ATTNS_NOMINAL/SLOPE; or reduce PP_AGCMAXGAIN; or lower PP_MIN_NS (more stationary noise suppression)."
            },
            plot_files=[str(p)],
        ))

    # Decision C: pumping (AGC/ATTNS interaction)
    if pumping_std <= pump_max:
        decisions.append(Decision(
            name="PHASE3_PUMPING",
            status="PASS",
            reason=f"Silence pumping OK: std(RMS)={pumping_std:.1f} dB <= {pump_max:.1f}.",
            recommendation="keep",
            metrics={"pumping_std_db": pumping_std},
        ))
    else:
        decisions.append(Decision(
            name="PHASE3_PUMPING",
            status="FAIL",
            reason=f"Silence pumping high: std(RMS)={pumping_std:.1f} dB > {pump_max:.1f}.",
            recommendation="decrease",
            metrics={
                "pumping_std_db": pumping_std,
                "recommend": "Reduce AGC pumping of fan: lower PP_AGCMAXGAIN and/or increase ATTNS_SLOPE; avoid changing AGC time constants unless necessary."
            },
        ))

    return decisions
