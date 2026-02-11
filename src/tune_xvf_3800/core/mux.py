from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import yaml
from .xvf_host import XvfHost

@dataclass(frozen=True)
class MuxPreset:
    name: str
    commands: list[list[Any]]
    labels: dict[str, str]

def load_mux_presets(path: str) -> dict[str, MuxPreset]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    presets: dict[str, MuxPreset] = {}
    for name, obj in raw.items():
        presets[name] = MuxPreset(name=name, commands=obj["commands"], labels=obj.get("labels", {}))
    return presets

def apply_mux(host: XvfHost, preset: MuxPreset) -> None:
    for cmd in preset.commands:
        # each cmd = ["AUDIO_MGR_OP_L", 3, 0] etc
        name = str(cmd[0])
        args = [str(x) for x in cmd[1:]]
        host.set_param(name, *args)
