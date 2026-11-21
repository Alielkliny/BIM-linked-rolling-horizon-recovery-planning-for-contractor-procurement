"""Amendment 4: extended rival grids on the development tuning streams (dev split, streams 0-9); framework frozen."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

EXT = [("reactive_expediting", {"theta": 30}), ("reactive_expediting", {"theta": 40}),
       ("buffer_management", {"zone_scale": 2.0}), ("buffer_management", {"zone_scale": 3.0}),
       ("reported_rescheduling", {"money_weight": 4.0}), ("reported_rescheduling", {"money_weight": 8.0}),
       ("budgeted_robust", {"money_weight": 1.0, "dev_q": 0.75}), ("budgeted_robust", {"money_weight": 1.0, "dev_q": 0.95})]


def tag(kw):
    return "_".join(f"{k}{v}" for k, v in kw.items())


def one(job):
    pol, kw = job
    cmd = [sys.executable, "run_exp.py", "--split", "dev", "--policies", pol, "--streams", "0-9", "--tag", f"tune_ext_{pol}_{tag(kw)}",
           "--policy-kw", json.dumps(kw), "--variant", tag(kw)]
    return subprocess.run(cmd, capture_output=True, text=True).returncode


if __name__ == "__main__":
    with ThreadPoolExecutor(1) as ex:
        print(list(ex.map(one, EXT)))
