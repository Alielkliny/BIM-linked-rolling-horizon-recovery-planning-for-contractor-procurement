"""Amendment 5: paired bootstrap test of the mean total-cost difference against each rival strategy, Holm-adjusted across
the four rivals on each episode set. Writes results/scores_bootstrap_mean.json."""
import json

import numpy as np
import pandas as pd

from common import RESULTS

K = ["inst", "seq", "sev", "stream"]
RIV = ["reactive_expediting", "buffer_management", "reported_rescheduling", "budgeted_robust"]
SETS = {"cf": "crossfit_fresh_a7_final.csv", "r1": "reserve1_a7.csv", "r2": "reserve2_a7.csv", "dev": None}


def holm(p):
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


out = {}
B = 10000
for key, f in SETS.items():
    if f is None:
        d = pd.concat([pd.read_csv(RESULTS / "raw" / "dev_a7.csv"), pd.read_csv(RESULTS / "raw" / "dev_final.csv").query("policy != 'proposed'")])
    else:
        d = pd.read_csv(RESULTS / "raw" / f)
    P = d[d.policy == "proposed"].set_index(K).total_cost
    rng = np.random.default_rng(20261109)
    idx = rng.integers(0, len(P), (B, len(P)))
    rows = []
    for r in RIV:
        R = d[d.policy == r].set_index(K).total_cost.reindex(P.index).to_numpy()
        diff = P.to_numpy() - R
        bm = diff[idx].mean(1)
        p = min(1.0, 2 * min((bm >= 0).mean(), (bm <= 0).mean()))
        p = max(p, 1.0 / B)
        rows.append({"rival": r, "mean_diff": float(diff.mean()), "saving_pct": float(-diff.mean() / R.mean() * 100),
                     "lo_pct": float(-np.quantile(bm, 0.975) / R.mean() * 100), "hi_pct": float(-np.quantile(bm, 0.025) / R.mean() * 100),
                     "p_boot": float(p), "n": int(len(diff))})
    adj = holm(np.array([x["p_boot"] for x in rows]))
    for x, a in zip(rows, adj):
        x["p_boot_holm"] = float(a)
    out[key] = rows
json.dump(out, open(RESULTS / "scores_bootstrap_mean.json", "w"), indent=1)
for k, rows in out.items():
    print(k, [(x["rival"][:10], round(x["saving_pct"], 2), round(x["lo_pct"], 2), x["p_boot_holm"]) for x in rows])
