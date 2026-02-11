# Tune_XVF_3800 — Tuning Guide (3 Phases)

This project is a **repeatable, measurement-driven tuning tool** for **XVF3800** deployments where:
- Your **speaker** is in the robot **chest**
- Your **mics + XMOS** are in the robot **head**
- You have **unavoidable GPU fan noise** near the microphones
- Your **physical I/O rate is 48 kHz stereo**, while each “logical” channel is **48k / N** (e.g., N=3 → 16 kHz per logical channel)

The tool runs tuning in **three phases**, producing:
- **Objective metrics**
- **PASS/FAIL decisions**
- **Recommendations** (increase / decrease / keep)
- **Suggested values** (Phase 1 auto-applies + iterates)
- **Plots** with target lines/bands
- A human-readable **report.md** per run

Contact: **hafsati.mohammed@gmail.com**


---

## Key audio terms (quick reference)

These terms are used throughout the tuning and in the reports:

- **Reference (far-end)**: the audio you **play to the speaker** (from the host) that becomes echo in the room.
- **Loopback reference**: the reference signal as observed inside the XVF path (used for correlation/delay checks).
- **Mic signal (near-end)**: what the microphones capture in the environment (your voice + fan + reverberation + echo).
- **Far-end only**: only reference audio is active; operator stays silent.
- **Near-end only**: operator speaks; reference is silent.
- **Double-talk**: far-end is playing **and** near-end speech occurs at the same time.

Why this matters:
- AEC learns to cancel echo only when the system is **causal** and gain structure is sane.
- Your **fan** makes “real world” noise suppression and AGC/ATTNS tuning essential.


---

## Sample rates and packing (important)

Your setup:
- Physical playback: **48 kHz, 2 channels** (you only use channel 0 for the speaker stimulus).
- Physical capture: **48 kHz, 2 channels** from the device output.
- Each physical channel may see **N packed logical channels** at **48k / N** (e.g. N=3 → 16 kHz logical).

In config:
- `io.physical_fs_hz = 48000`
- `io.pack_factor = N`
- `io.logical_fs_hz = 48000 / N`

The tool:
1. Generates a packed **48 kHz stereo WAV** for playback (injecting the reference in the correct “slot”).
2. Records **48 kHz stereo**.
3. Unpacks each channel back to **logical 16k** (or 24k/32k depending on N).
4. Runs metrics and produces plots/decisions.


---

## What you need installed

On the host machine running this tool (Linux recommended):
- Python 3.10+
- `numpy`, `matplotlib`, `pyyaml`
- ALSA utilities: `aplay`, `arecord`
- Your XVF control binary: `xvf_host` accessible by path (or configured)

Recommended (but optional):
- `sox` for quick listening / conversions
- Audacity for spot checks

---

## Project layout (expected)

Typical layout (simplified):
```
Tune_XVF_3800/
  configs/
    robot_default.yaml
    mux_presets.yaml
  assets/
    generated/
      N3/
        white_noise_0dbfs_16k_mono.wav
        silence_16k_mono.wav
        silence_white_noise_silence_16k_mono.wav
        far_end_speech_16k_mono.wav
  runs/
    2026-02-11_...
      phase1_gain_delay/
      phase2_aec_effectiveness/
      phase3_agc_noise_attns/
      report.md
  src/
    tune_xvf_3800/
      phases.py
      metrics/
      reporting/
      audio/
      core/
```

Outputs always go to:
- `runs/<timestamp>/...` with plots, decisions, and a `report.md`.

---

## Configuration you must set

Edit `configs/robot_default.yaml`:

### Audio devices
Set ALSA device names:
- `audio.playback_device`: speaker output device used by `aplay`
- `audio.capture_device`: capture device used by `arecord`

### Reference injection mapping
Set where the reference goes in the packed playback stream:
- `audio.reference_physical`: `"L"` or `"R"` (which physical channel carries packed logical reference)
- `audio.reference_slot`: `0..N-1` (which packed slot within that physical channel)

> Example: If you use physical **Left** channel and slot **0** to carry your reference, set:
> - `reference_physical: "L"`
> - `reference_slot: 0`

### Pack factor
Set `io.pack_factor: 3` if using 16 kHz logical channels.

### Mux presets
Make sure `configs/mux_presets.yaml` contains presets for:
- `phase1_ref_pre_post_only`
- `phase1_gain_delay`
- `phase2_aec_effectiveness`
- `phase3_agc_noise_attns`

These control what the device outputs so the tool can measure the right signals.


---

# How tuning works (3 phases)

## Phase 1 — Gain + delay (make AEC possible and stable)
**Goal:** remove clipping risk, enforce mic/ref ratio, ensure causality and usable delay headroom.

Phase 1 is **automatic**:
- It records stimulus captures
- Computes objective metrics
- Produces PASS/FAIL decisions
- **Auto-applies** suggested changes (within safe bounds)
- Repeats until PASS (or max iterations)

### Phase 1 steps
1) **Reference gain**
   - Tunes `AUDIO_MGR_REF_GAIN` so ref post-gain peaks near the target (e.g. -1.5 dBFS) without clipping.
2) **Mic gain**
   - Tunes `AUDIO_MGR_MIC_GAIN` so **all mics** peak at least **6 dB below reference**.
3) **AEC silence level**
   - Sets `AEC_AECSILENCELEVEL` from measured noise power in the reference line during injected digital silence.
4) **System delay / causality**
   - Tunes `AUDIO_MGR_SYS_DELAY` so the system remains **causal** for all mics and the lag is **<= 40 logical samples**.

**What you do physically:**  
Put the robot in a “worst-case” position (near walls/corners, head rotation) and keep it steady during captures.

**Expected output:**  
- `phase1_final_params.json` with tuned values
- Plots: waveform with dBFS lines, mic margin bars, correlation with acceptable band


---

## Phase 2 — AEC settings + NL model (make echo removal effective)
**Goal:** improve far-end echo removal while keeping near-end speech usable during double-talk.

Phase 2 is **measure → adjust → re-run**:
- The tool plays a far-end reference speech sample.
- It prints timed cues including **“TALK NOW”** (double-talk window).
- It objectively measures:
  - ERLE proxy (echo reduction) during far-end-only
  - Residual-to-mic ratio
  - Near-end preservation during double-talk (beam rise)

### What you tune in Phase 2
Common parameters involved:
- `PP_GAMMA_E` (linear echo oversubtraction)
- `PP_GAMMA_ENL` (non-linear echo oversubtraction)
- `PP_GAMMA_ETAIL` (tail/reverb oversubtraction)
- `PP_DTSENSITIVE` (double-talk balance)
- `PP_MGSCALE` (far-end activity detection tradeoff)

**NL model training (recommended before Phase 2):**
Run the provided `nl_model_training` process (2–3 times) in a stable, quiet environment and bake the chosen model into firmware.

**Expected output:**  
- PASS/FAIL decisions such as:
  - `PHASE2_ECHO_EFFECTIVENESS_MIC0`
  - `PHASE2_DOUBLE_TALK_NEAR_END_PRESERVATION`
- Plots: RMS envelopes with shaded segments (far-end-only, double-talk)

**How to iterate:**
- If echo is not suppressed enough → increase `PP_GAMMA_E` (or `PP_GAMMA_ENL` for distorted residuals).
- If near-end is crushed in double-talk → decrease `PP_DTSENSITIVE` and/or reduce gamma.
- Re-run Phase 2 and compare reports.


---

## Phase 3 — AGC + noise + ATTNS (make it usable with fan)
**Goal:** stable speech level plus acceptable pause noise under fan conditions.

Phase 3 is **measure → adjust → re-run**:
- The tool records a scripted sequence with cues:
  - “TALK NOW (normal)”, then silence, then “TALK NOW (quiet/far)”.

### What you tune in Phase 3
- `PP_AGCDESIREDLEVEL` (target output level)
- `PP_AGCMAXGAIN` (cap amplification so fan doesn’t dominate)
- `PP_MIN_NS` / `PP_MIN_NN` (stationary / non-stationary noise suppression)
- `PP_ATTNS_MODE`, `PP_ATTNS_NOMINAL`, `PP_ATTNS_SLOPE` (reduce pause noise when AGC is high)

**What the tool measures (objective):**
- Speech RMS level in talk segments vs target band
- Pause noise RMS vs max
- “Pumping” (variance of RMS envelope in silence)

**Expected output:**  
- PASS/FAIL decisions:
  - `PHASE3_SPEECH_LEVEL`
  - `PHASE3_PAUSE_NOISE`
  - `PHASE3_PUMPING`
- Plots: RMS envelope with shaded segments


---

# How to run

## Basic commands
Assuming your CLI entrypoint is `tune-xvf` (adapt to your runner if different):

### Phase 1 (auto-tunes gain/delay)
```bash
tune-xvf phase1 --config configs/robot_default.yaml --assets assets/generated/N3
```

### Phase 2 (AEC effectiveness + double-talk cues)
```bash
tune-xvf phase2 --config configs/robot_default.yaml --assets assets/generated/N3
```

### Phase 3 (AGC/noise/ATTNS with fan)
```bash
tune-xvf phase3 --config configs/robot_default.yaml
```

## Where results go
After each run you get:
- `runs/<timestamp>/report.md`
- `runs/<timestamp>/phase*/decisions/*.json`
- `runs/<timestamp>/phase*/plots/**/*.png`
- audio captures under each phase folder

Open the report:
- `runs/<timestamp>/report.md`


---

# How many times to run each phase?

Recommended iteration counts:
- **Phase 1:** usually 1 run (it auto-iterates internally)
- **Phase 2:** 3–6 runs (after adjusting see PP_GAMMA_ / DTSENSITIVE)
- **Phase 3:** 2–6 runs (until pause noise and pumping are acceptable with fan)

Keep each run folder; you’ll compare plots/metrics between iterations.


---

# Getting tuned values into firmware defaults

Phase 1 writes a JSON snapshot:
- `runs/<timestamp>/phase1_gain_delay/phase1_final_params.json`

To bake values into firmware defaults (YAML-based build):
1) Copy final values into the appropriate `control_param_values.yaml` (product specifier folder).
2) Rebuild and flash.

> Tip: Keep a “tuned_<date>.yaml” copy so you can diff changes over time.


---

# Troubleshooting

- **Near-end presence FAIL in Phase 2:** You didn’t speak loud enough during “TALK NOW”. Re-run and speak closer/louder.
- **SYS_DELAY keeps failing:** The system might be truly acausal (reference arrives too late). Ensure the reference path into XVF is as early as possible; otherwise SYS_DELAY must be made more negative (adds mic delay).
- **Pause noise too high (fan):** Reduce `PP_AGCMAXGAIN`, increase `PP_ATTNS_SLOPE`, and/or increase stationary suppression (lower `PP_MIN_NS`).
- **Clipping:** Always fix clipping first (REF_GAIN then MIC_GAIN). Clipping breaks everything downstream.


---

## Contact
For questions or improvements:
- **hafsati.mohammed@gmail.com**