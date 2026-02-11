from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_report(run_dir: Path) -> Path:
    """
    Scans run_dir for decisions.json files and builds a report.md.
    Convention used by phases.py below:
      - each phase writes decisions under <phase_dir>/decisions/*.json
    """
    report = []
    report.append("# Tune_XVF_3800 Run Report\n")

    phase_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("phase")])

    for phase in phase_dirs:
        report.append(f"## {phase.name}\n")
        dec_dir = phase / "decisions"
        if not dec_dir.exists():
            report.append("_No decisions found._\n")
            continue

        dec_files = sorted(dec_dir.glob("*.json"))
        for df in dec_files:
            d = _load_json(df)
            name = d.get("name", df.stem)
            status = d.get("status", "UNKNOWN")
            rec = d.get("recommendation", "")
            reason = d.get("reason", "")

            report.append(f"### {name}\n")
            report.append(f"- **Status:** {status}\n")
            report.append(f"- **Recommendation:** {rec}\n")
            report.append(f"- **Reason:** {reason}\n")

            if d.get("suggested_value") is not None:
                report.append(f"- **Suggested value:** {d['suggested_value']}\n")
            if d.get("suggested_int") is not None:
                report.append(f"- **Suggested int:** {d['suggested_int']}\n")

            # show key metrics (not huge)
            metrics = d.get("metrics", {})
            if metrics:
                report.append("\n**Key metrics:**\n")
                for k, v in list(metrics.items())[:12]:
                    report.append(f"- `{k}`: `{v}`\n")

            # embed plots
            plots = d.get("plot_files", [])
            for p in plots:
                # make relative
                try:
                    rel = Path(p).relative_to(run_dir)
                except Exception:
                    rel = p
                report.append(f"\n![]({rel})\n")

            report.append("\n---\n")

    out = run_dir / "report.md"
    out.write_text("\n".join(report), encoding="utf-8")
    return out
