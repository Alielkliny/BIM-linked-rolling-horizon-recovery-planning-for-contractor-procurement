"""Comparison table (comparison_scores.csv) and paired statistics per split.

Means (total cost, mean delay): Wilcoxon signed-rank test on episode pairs. CVaR90 of delay: paired bootstrap of
episodes (2,000 replicates), two-sided p from the share of replicates on the far side of zero. Holm adjustment across
the rivals and ablations of the proposed method, one family per metric and split [Holm 1979]."""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, ROOT, write_json

KEY = ["inst", "seq", "sev", "stream"]
METRICS = {"total_cost": "total_cost", "mean_delay": "delay", "cvar90_delay": "delay"}
B = 2000


def cvar(x, a=0.9):
    x = np.sort(np.asarray(x, float))
    k = max(1, int(np.ceil((1 - a) * len(x))))
    return float(x[-k:].mean())


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    m = len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def paired_stats(P: pd.DataFrame, R: pd.DataFrame, metric: str, rng):
    col = METRICS[metric]
    x, y = P[col].to_numpy(float), R[col].to_numpy(float)
    if metric == "cvar90_delay":
        n = len(x)
        diffs = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, n)
            diffs[b] = cvar(x[idx]) - cvar(y[idx])
        d0 = cvar(x) - cvar(y)
        p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        return d0, max(p, 1.0 / B), lo, hi
    d = x - y
    if np.allclose(d, 0):
        return 0.0, 1.0, 0.0, 0.0
    p = stats.wilcoxon(x, y, zero_method="wilcox").pvalue
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return float(d.mean()), float(p), float(lo), float(hi)


def score(df: pd.DataFrame, idea: str, stage: str, split: str, proposed: str, others: list, seed=11):
    rng = np.random.default_rng(seed)
    P = df[df.policy == proposed].set_index(KEY).sort_index()
    rows, detail = [], []
    for metric in METRICS:
        col = METRICS[metric]
        val = (lambda s: cvar(s[col])) if metric == "cvar90_delay" else (lambda s: float(s[col].mean()))
        rows.append({"idea": idea, "stage": stage, "split": split, "method": proposed, "metric": metric, "value": val(P), "p_holm": ""})
        res = []
        for o in others:
            R = df[df.policy == o].set_index(KEY).sort_index()
            R = R.reindex(P.index)
            if R[col].isna().any():
                continue
            diff, p, lo, hi = paired_stats(P, R, metric, rng)
            res.append((o, val(R), diff, p, lo, hi, R))
        ph = holm([r[3] for r in res]) if res else []
        for (o, v, diff, p, lo, hi, R), pa in zip(res, ph):
            rows.append({"idea": idea, "stage": stage, "split": split, "method": o, "metric": metric, "value": v,
                         "p_holm": "" if stage == "probe" else round(float(pa), 6)})
            d = P[col].to_numpy() - R[col].to_numpy()
            detail.append({"split": split, "metric": metric, "rival": o, "proposed": val(P), "rival_value": v,
                           "diff": diff, "rel_gain": (v - val(P)) / v if v else 0.0, "ci_low": lo, "ci_high": hi,
                           "p": p, "p_holm": float(pa), "wins": int((d < 0).sum()), "ties": int((d == 0).sum()),
                           "losses": int((d > 0).sum())})
    return rows, detail


def main(spec_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    all_rows, all_detail = [], []
    for item in spec["runs"]:
        df = pd.read_csv(RESULTS / "raw" / item["file"])
        df["variant"] = df.variant.fillna("") if "variant" in df else ""
        if "variant" in item:
            df = df[df.variant.isin(item["variant"])]
        if "streams" in item:
            lo, hi = item["streams"]
            df = df[(df.stream >= lo) & (df.stream <= hi)]
        if "rename" in item:
            df["policy"] = df.policy.replace(item["rename"])
        rows, detail = score(df, item["idea"], item["stage"], item["split"], item["proposed"], item["others"])
        all_rows += rows
        all_detail += detail
    out = ROOT / spec.get("out", "comparison_scores.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    write_json(RESULTS / spec.get("detail", "scores_detail.json"), all_detail)
    dd = pd.DataFrame(all_detail)
    print(dd[["split", "metric", "rival", "proposed", "rival_value", "rel_gain", "ci_low", "ci_high", "p_holm", "wins", "ties", "losses"]].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
