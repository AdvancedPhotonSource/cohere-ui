"""Diagnostic for the macOS libomp conflict between torch's bundled
``libomp.dylib`` and the Homebrew ``libomp`` that xrayutilities links.

Both load when the GUI imports torch and then calls into xrayutilities (e.g.
``QConversion.area()`` from ``instr.get_geometry``), and the OpenMP runtime
calls ``abort()``, killing the Jupyter kernel with no Python traceback.
The fix (per CLAUDE.md) is to replace torch's bundled libomp with a symlink
to the Homebrew copy. This module DETECTS the misconfiguration so the GUI
can surface a remediation banner instead of waiting for the kernel to die.

Must not import torch or xrayutilities at module top level, which would
defeat the purpose if their import order is part of the bug.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


Severity = Literal["ok", "warning", "error"]


_REMEDIATION = """brew install libomp
cd "$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib"
mv libomp.dylib libomp.dylib.bundled
ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib libomp.dylib"""


@dataclass(frozen=True)
class EnvCheckResult:
    ok: bool
    severity: Severity
    message: str
    remediation: str

    @classmethod
    def ok_result(cls) -> "EnvCheckResult":
        return cls(ok=True, severity="ok", message="libomp check passed", remediation="")


def _torch_libomp_path() -> Optional[Path]:
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return None
    torch_pkg = Path(spec.origin).parent
    candidate = torch_pkg / "lib" / "libomp.dylib"
    return candidate if candidate.exists() else None


def _xrayutilities_present() -> bool:
    return importlib.util.find_spec("xrayutilities") is not None


def _homebrew_libomp() -> Optional[Path]:
    candidate = Path("/opt/homebrew/opt/libomp/lib/libomp.dylib")
    return candidate if candidate.exists() else None


def check_libomp_consistency() -> EnvCheckResult:
    """Return :class:`EnvCheckResult` describing the current state.

    Non-darwin platforms, or environments without xrayutilities/torch, are
    treated as OK because the conflict only affects macOS with torch and xrayutilities.
    """
    if sys.platform != "darwin":
        return EnvCheckResult.ok_result()
    if not _xrayutilities_present():
        return EnvCheckResult.ok_result()
    torch_libomp = _torch_libomp_path()
    if torch_libomp is None:
        return EnvCheckResult.ok_result()
    if torch_libomp.is_symlink():
        try:
            target = torch_libomp.resolve()
        except OSError:
            return EnvCheckResult(
                ok=False,
                severity="warning",
                message=(
                    f"torch's libomp is a symlink at {torch_libomp} but the "
                    "target cannot be resolved"
                ),
                remediation=_REMEDIATION,
            )
        brew = _homebrew_libomp()
        if brew is not None and target == brew.resolve():
            return EnvCheckResult.ok_result()
        return EnvCheckResult(
            ok=False,
            severity="warning",
            message=(
                f"torch's libomp is a symlink but points to {target}, not the "
                "Homebrew libomp at /opt/homebrew/opt/libomp/lib/libomp.dylib"
            ),
            remediation=_REMEDIATION,
        )
    return EnvCheckResult(
        ok=False,
        severity="error",
        message=(
            "torch ships a bundled libomp.dylib that will conflict with "
            "xrayutilities on macOS and crash the kernel during Postprocess. "
            "Replace it with a symlink to Homebrew's libomp."
        ),
        remediation=_REMEDIATION,
    )
