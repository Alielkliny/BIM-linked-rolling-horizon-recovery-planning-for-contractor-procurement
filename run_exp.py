"""Parallel experiment runner: every (split, cell, stream) episode is run for every requested policy under common
random numbers. Results are appended to results/raw/<tag>.csv with one row per episode and policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from common import INST, RESULTS, read_json

SPLITS = {
    # name: (instances, environment pool, seed base)
    "probe": (["schependomlaan"], "calib", 500),
    "dev": (["schependomlaan"], "calib", 1000),
    "check": (["schependomlaan"], "check", 2000),
    "check2": (["schependomlaan"], "check", 2500),
    "cf0": (["schependomlaan"], "cf0_test", 5000),
    "cf1": (["schependomlaan"], "cf1_test", 5100),
    "cf2": (["schependomlaan"], "cf2_test", 5200),
    "reserve1": (["schependomlaan"], "reserve1", 3000),
    "reserve2": (["schependomlaan"], "reserve2", 4000),
}
CELLS = [(q, v) for q in ("single", "sequential", "compound") for v in ("low", "medium", "high")]

_CACHE = {}


def _inst(name):
    if name not in _CACHE:
        _CACHE[name] = read_json(INST / f"{name}.json")
    return _CACHE[name]


def stream_seed(split: str, inst: str, cell: tuple, k: int) -> int:
    base = SPLITS[split][2]
    h = int(hashlib.sha256(f"{split}|{inst}|{cell[0]}|{cell[1]}|{k}".encode()).hexdigest()[:8], 16)
    return base * 100000 + (h % 99991)


def _low_priority() -> None:
    """Run workers below normal priority so the laptop stays responsive."""
    try:
        import ctypes
        BELOW_NORMAL = 0x00004000
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(h, BELOW_NORMAL)
    except Exception:
        try:
            os.nice(10)
        except Exception:
            pass


def run_one(task: dict) -> dict:
    from env import Episode, draw_exogenous
    from policies import ROSTER
    inst = _inst(task["inst"])
    ov = task.get("overrides") or {}
    ex = draw_exogenous(inst, task["pool"], task["seq"], task["sev"], task["seed"], overrides=ov)
    fac, role, pooling = ROSTER[task["policy"]]
    t0 = time.time()
    ep = Episode(inst, ex, None, pooling=pooling, overrides=ov)
    pol = fac(inst, task["seed"], ep)
    for k, v in (task.get("policy_kw") or {}).items():
        setattr(pol, k, v)
    if getattr(pol, "sib_oracle", False):   # diagnostic ceiling: the planner may read the episode's true arrivals
        pol.episode = ep
    if task["split"].startswith("cf") and hasattr(pol, "model"):
        from uncertainty import PlannerModel
        pol.model = PlannerModel(task["split"] + "_train")
    elif task["split"] in ("reserve1", "reserve2") and hasattr(pol, "model"):
        from uncertainty import PlannerModel   # Amendment 2, item 5: planners fit on all 2007-2011 shipments
        pol.model = PlannerModel("dev_all")
    ep.policy = pol
    r = ep.run()
    r.pop("first_error", None)
    r.update({k: task[k] for k in ("split", "inst", "seq", "sev", "stream", "seed", "policy")})
    r["variant"] = task.get("variant", "")
    r["sec"] = round(time.time() - t0, 3)
    r["solves"] = getattr(pol, "solves", 0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--policies", required=True)
    ap.add_argument("--streams", default="0-29")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--policy-kw", default="{}")
    ap.add_argument("--variant", default="")
    a = ap.parse_args()
    lo, hi = [int(x) for x in a.streams.split("-")]
    insts, pool, _ = SPLITS[a.split]
    cells = CELLS if a.cells == "all" else [tuple(c.split(":")) for c in a.cells.split(",")]
    pols = a.policies.split(",")
    ov = json.loads(a.overrides)
    pkw = json.loads(a.policy_kw)
    tasks = [{"split": a.split, "inst": i, "pool": pool, "seq": c[0], "sev": c[1], "stream": k,
              "seed": stream_seed(a.split, i, c, k), "policy": p, "overrides": ov, "policy_kw": pkw, "variant": a.variant}
             for i in insts for c in cells for k in range(lo, hi + 1) for p in pols]
    tag = a.tag or f"{a.split}"
    out = RESULTS / "raw" / f"{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old.inst, old.seq, old.sev, old.stream, old.policy, old.variant.fillna("")))
    tasks = [t for t in tasks if (t["inst"], t["seq"], t["sev"], t["stream"], t["policy"], t["variant"]) not in done]
    # slow policies first so the pool stays busy
    slow = {"proposed": 0, "abl_risk_neutral": 0, "abl_no_approval": 0, "abl_no_state": 0, "abl_no_bim": 0}
    tasks.sort(key=lambda t: 0 if t["policy"] in slow else 1)
    print(f"{len(tasks)} episodes to run -> {out}", flush=True)
    t0 = time.time()
    buf = []
    _low_priority()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_low_priority) as exe:
        futs = [exe.submit(run_one, t) for t in tasks]
        for n, f in enumerate(as_completed(futs), 1):
            buf.append(f.result())
            if len(buf) >= 50 or n == len(futs):
                df = pd.DataFrame(buf)
                df.to_csv(out, mode="a", header=not out.exists(), index=False)
                buf = []
                print(f"{n}/{len(futs)} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
