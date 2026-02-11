from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

from .core.xvf_host import XvfHost

@dataclass
class RunContext:
    run_dir: Path
    cfg: dict
    mux_presets: dict
    host: XvfHost

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
