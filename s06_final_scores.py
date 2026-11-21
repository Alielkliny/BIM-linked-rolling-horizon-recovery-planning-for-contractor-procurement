"""Final score tables for design A4: merges the episode files of each split and writes comparison_scores.csv
and results/scores_<split>_final_detail.json (manuscript tokens)."""
import json

import pandas as pd

from common import RESULTS, ROOT
from s05_scores import score

RIVALS = ["reactive_expediting", "buffer_management", "reported_rescheduling", "budgeted_robust", "no_recovery"]
ABL = ["abl_expected_value", "abl_no_approval", "abl_no_state", "abl_no_bim", "abl_static"]


def merged(files, keep=None):
    parts = []
    for f, pols in files:
        p = RESULTS / "raw" / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if pols:
            d = d[d.policy.isin(pols)]
        parts.append(d)
    d = pd.concat(parts).drop_duplicates(subset=["policy", "inst", "seq", "sev", "stream"], keep="last")
    return d


def main():
    rows_all = []
    splits = {
        "dev": [("dev_eval_r.csv", RIVALS), ("dev_a5.csv", ["proposed"]), ("dev_abl5.csv", ABL + ["variant_cvar"]),
                ("dev_a4_as_variant.csv", ["variant_sibling"])],
        "check2": [("check2_eval.csv", None)],
        "reserve1": [("reserve1_eval.csv", None)],
        "reserve2": [("reserve2_eval.csv", None)],
    }
    for split, files in splits.items():
        if not any((RESULTS / "raw" / f).exists() for f, _ in files):
            continue
        d = merged(files)
        d.to_csv(RESULTS / "raw" / f"{split}_final.csv", index=False)
        others = [o for o in RIVALS + ABL + ["variant_cvar", "variant_sibling"] if o in set(d.policy)]
        stage = "development" if split in ("dev", "check2") else "test"
        rows, detail = score(d, "A5", stage, split, "proposed", others)
        rows_all += rows
        json.dump(detail, open(RESULTS / f"scores_{split}_final_detail.json", "w"), indent=1, default=float)
        print(split, len(d), sorted(set(d.policy)))
    # probe rows (screening of the chosen design on fresh probe streams, design A3 = A4 without the money-weight change)
    pr = pd.read_csv(RESULTS / "raw" / "probe_fresh.csv")
    pr = pr[pr.policy.isin(RIVALS + ["abl_risk_neutral", "abl_expected_value", "abl_no_approval", "abl_no_state", "abl_no_bim", "abl_static"])]
    pr["policy"] = pr.policy.replace({"abl_risk_neutral": "proposed"})
    rows, _ = score(pr, "A5", "probe", "probe_fresh", "proposed", RIVALS + ABL)
    rows_all = rows + rows_all
    pd.DataFrame(rows_all).to_csv(ROOT / "comparison_scores.csv", index=False)
    print("comparison rows", len(rows_all))


if __name__ == "__main__":
    main()
