"""Equal-size tuning grids on the development tuning streams (dev split, streams 0-9), and the pre-declared selection
rule as coded in select(): lowest mean total disruption cost, mean delay as the tie-break within 1%."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from common import RESULTS, write_json

GRIDS = {
    "reactive_expediting": [{"theta": v} for v in (5, 10, 20)],
    "buffer_management": [{"zone_scale": v} for v in (0.5, 1.0, 1.5)],
    "reported_rescheduling": [{"money_weight": v} for v in (0.5, 1.0, 2.0)],
    "budgeted_robust": [{"money_weight": v} for v in (0.5, 1.0, 2.0)],
    "proposed": [{"money_weight": v} for v in (0.5, 1.0, 2.0)],
}


def tag(kw):
    return "_".join(f"{k}{v}" for k, v in kw.items())


def run(policies, grids, streams="0-9", out_tag="tune_dev"):
    for pol in policies:
        for kw in grids[pol]:
            cmd = [sys.executable, "run_exp.py", "--split", "dev", "--policies", pol, "--streams", streams, "--tag", out_tag,
                   "--policy-kw", json.dumps(kw), "--variant", tag(kw)]
            subprocess.run(cmd, check=True)


def select(out_tag="tune_dev"):
    d = pd.read_csv(RESULTS / "raw" / f"{out_tag}.csv")
    d["variant"] = d.variant.fillna("")
    g = d.groupby(["policy", "variant"]).agg(mean_delay=("delay", "mean"), total_cost=("total_cost", "mean"), n=("delay", "size")).reset_index()
    chosen = {}
    for pol, x in g.groupby("policy"):
        best = x.total_cost.min()
        cand = x[x.total_cost <= best * 1.01].sort_values("mean_delay")
        chosen[pol] = cand.iloc[0].variant
    write_json(RESULTS / f"{out_tag}_selection.json", {"table": g.to_dict(orient="records"), "chosen": chosen,
                                                        "rule": "lowest mean total disruption cost on the tuning streams; mean delay tie-break within 1%"})
    print(g.round(2).to_string())
    print(chosen)
    return chosen


if __name__ == "__main__":
    pols = sys.argv[1].split(",")
    grids = dict(GRIDS)
    if len(sys.argv) > 2:
        grids.update(json.loads(sys.argv[2]))
    run(pols, grids)
    select()
