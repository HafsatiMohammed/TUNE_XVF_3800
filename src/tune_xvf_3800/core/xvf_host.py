from __future__ import annotations
import subprocess
from dataclasses import dataclass
from typing import Optional

@dataclass
class XvfHost:
    path: str
    protocol: str = "usb"
    sudo: bool = False

    def _cmd(self, *args: str) -> list[str]:
        cmd: list[str] = []
        if self.sudo:
            cmd.append("sudo")
        cmd.append(self.path)
        # if your xvf_host needs protocol flag, uncomment:
        # cmd += ["--protocol", self.protocol]
        cmd += list(args)
        return cmd

    def run(self, *args: str) -> str:
        cmd = self._cmd(*args)
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()

    def set_param(self, name: str, *values: str) -> None:
        self.run(name, *values)

    def get_raw(self, name: str) -> str:
        return self.run(name)

    def get_float(self, name: str) -> float:
        s = self.get_raw(name)
        # common formats: "PARAM 1.23" or "PARAM: 1.23"
        toks = s.replace(":", " ").split()
        for t in toks[::-1]:
            try:
                return float(t.replace("f", ""))
            except ValueError:
                continue
        raise ValueError(f"Could not parse float from: {s}")

    def get_int(self, name: str) -> int:
        s = self.get_raw(name)
        toks = s.replace(":", " ").split()
        for t in toks[::-1]:
            try:
                return int(float(t))
            except ValueError:
                continue
        raise ValueError(f"Could not parse int from: {s}")
