from __future__ import annotations
import argparse
from pathlib import Path
import datetime as dt

from .core.xvf_host import XvfHost
from .core.mux import load_mux_presets
from .run_context import RunContext, load_yaml
from .phases import run_phase1_gain_delay, run_phase2_aec_effectiveness, run_phase3_agc_noise_attns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="configs/robot_default.yaml")
    ap.add_argument("--mux", default="configs/mux_presets.yaml")
    ap.add_argument("--assets", default="assets/generated/N3")  # put your 16k mono files here
    ap.add_argument("--runs", default="runs")
    ap.add_argument("cmd", choices=["phase1", "phase2", "phase3"])
    args = ap.parse_args()

    cfg = load_yaml(args.robot)
    mux_presets = load_mux_presets(args.mux)

    run_stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.runs) / f"{run_stamp}_{args.cmd}"
    run_dir.mkdir(parents=True, exist_ok=True)

    host = XvfHost(
        path=cfg["device"]["xvf_host_path"],
        protocol=cfg["device"].get("protocol", "usb"),
        sudo=bool(cfg["device"].get("sudo", False)),
    )

    ctx = RunContext(run_dir=run_dir, cfg=cfg, mux_presets=mux_presets, host=host)
    assets_dir = Path(args.assets)

    if args.cmd == "phase1":
        run_phase1_gain_delay(ctx, assets_dir)
    elif args.cmd == "phase2":
        run_phase2_aec_effectiveness(ctx, assets_dir)
    elif args.cmd == "phase3":
        run_phase3_agc_noise_attns(ctx, assets_dir)

    print(f"\nDone. Outputs in: {run_dir}\n")
