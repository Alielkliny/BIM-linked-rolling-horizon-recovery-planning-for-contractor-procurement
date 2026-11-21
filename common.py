"""Shared paths, hashing and manifest helpers for the recovery-planning pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
PROBE = ROOT / "probe"
INST = ROOT / "instances"
LOGS = ROOT / "logs"
for _p in (PROC, RESULTS, PROBE, INST, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

SCMS_CSV = RAW / "scms_wayback" / "scms.csv"
SCMS_URL = ("https://web.archive.org/web/2023id_/https://data.usaid.gov/api/views/"
            "a3rc-nmf6/rows.csv?accessType=DOWNLOAD")
SCHEP = RAW / "schependomlaan"
SCHEP_ASPLANNED_IFC = SCHEP / "IFC Schependomlaan incl planningsdata.ifc"
SCHEP_DESIGN_IFC = SCHEP / "IFC Schependomlaan.ifc"
SCHEP_XML = SCHEP / "Uitvoering Schependomlaan 18-02-2015.xml"
SCHEP_EVENTLOG_PLAN = SCHEP / "eventlog IFC schependomlaan.csv"
SCHEP_EVENTLOG_BUILT = SCHEP / "Schependomlaan eventlog as planned data incl as built data.csv"


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_record() -> dict:
    mods = {}
    for m in ("numpy", "scipy", "pandas", "networkx", "simpy", "highspy", "ifcopenshell", "matplotlib", "numba"):
        try:
            mod = __import__(m)
            mods[m] = getattr(mod, "__version__", "unknown")
        except Exception:  # pragma: no cover
            mods[m] = "missing"
    try:
        import highspy
        mods["highs"] = highspy.Highs().version()
    except Exception:
        pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": mods}


def write_json(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_json(path: Path | str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def stage_manifest(stage: str, inputs: list, outputs: list, code: list, extra: dict | None = None) -> dict:
    inputs = [_abs(p) for p in inputs]
    outputs = [_abs(p) for p in outputs]
    code = [_abs(p) for p in code]
    rec = {
        "stage": stage,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {str(Path(p).relative_to(ROOT)): sha256(p) for p in inputs},
        "outputs": {str(Path(p).relative_to(ROOT)): sha256(p) for p in outputs if Path(p).exists()},
        "code": {str(Path(p).relative_to(ROOT)): sha256(p) for p in code},
        "environment": env_record(),
    }
    if extra:
        rec.update(extra)
    write_json(RESULTS / f"manifest_{stage}.json", rec)
    return rec
