"""Cross-fitted development scores (protocol Amendment 1): merges the three fold files of one design tag and applies
the acceptance rule. Usage: python s07_crossfit.py <tag> (fold files results/raw/cf_<tag>.csv, cf1_<tag>.csv, cf2_<tag>.csv)."""
import json
import sys

import numpy as np
import pandas as pd

from common import RESULTS
from s05_scores import score

RIVALS = ["reactive_expediting", "buffer_management", "reported_rescheduling", "budgeted_robust", "no_recovery"]


def load(tag, rival_tag=None):
    """Episodes of one design tag; rival rows come from rival_tag (same episodes and draws) when given."""
    parts = []
    for t in ([rival_tag] if rival_tag else []) + [tag]:
        for k, f in enumerate([f"cf_{t}.csv", f"cf1_{t}.csv", f"cf2_{t}.csv"]):
            p = RESULTS / "raw" / f
            if p.exists():
                d = pd.read_csv(p)
                if t == rival_tag and rival_tag != tag:
                    d = d[d.policy != "proposed"]
                d["fold"] = k
                parts.append(d)
    d = pd.concat(parts)
    d = d.drop_duplicates(subset=["policy", "split", "inst", "seq", "sev", "stream"], keep="last")
    d["stream"] = d["stream"] + 1000 * d["fold"]  # unique pairing key across folds
    return d


def accept(detail, others):
    """Amendment 1 rule: >=5% lower mean total cost than every rival (Holm p<0.05); no rival ahead on mean delay by
    more than 0.5 wd or on CVaR90 delay by more than 1.0 wd with Holm p<0.05."""
    get = {(r["metric"], r["rival"]): r for r in detail}
    out, ok = [], True
    for o in others:
        tc, md, cv = get[("total_cost", o)], get[("mean_delay", o)], get[("cvar90_delay", o)]
        c1 = tc["rel_gain"] >= 0.05 and tc["p_holm"] < 0.05
        c2 = not (md["diff"] > 0.5 and md["p_holm"] < 0.05)
        c3 = not (cv["diff"] > 1.0 and cv["p_holm"] < 0.05)
        ok &= c1 and c2 and c3
        out.append(dict(rival=o, gain_pct=round(100 * tc["rel_gain"], 2), p_holm=round(tc["p_holm"], 5),
                        wins=tc["wins"], losses=tc["losses"], delay_diff=round(md["diff"], 2),
                        cvar_diff=round(cv["diff"], 2), cost_ok=c1, delay_ok=c2, cvar_ok=c3))
    return ok, out


if __name__ == "__main__":
    tag = sys.argv[1]
    d = load(tag, sys.argv[2] if len(sys.argv) > 2 else None)
    print(d.groupby(["policy", "fold"]).size().unstack())
    others = [o for o in RIVALS if o in set(d.policy)]
    rows, detail = score(d, tag, "development", "crossfit", "proposed", others)
    ok, table = accept(detail, others)
    print(pd.DataFrame(table).to_string(index=False))
    m = d.groupby("policy")[["total_cost", "delay"]].mean().round(1)
    print(m)
    print("ACCEPT" if ok else "REJECT")
    json.dump(dict(tag=tag, accept=ok, table=table, detail=detail), open(RESULTS / f"crossfit_{tag}.json", "w"),
              indent=1, default=float)
