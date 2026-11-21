"""Run one split for the full roster with each policy's tuned configuration (results/tune_dev_selection.json)."""
import json
import subprocess
import sys

from common import RESULTS

ROSTER = ["proposed", "variant_cvar", "abl_expected_value", "abl_no_approval", "abl_no_state", "abl_no_bim", "abl_static",
          "reported_rescheduling", "budgeted_robust", "buffer_management", "reactive_expediting", "no_recovery", "oracle"]
TUNED = {"proposed", "reported_rescheduling", "budgeted_robust", "buffer_management", "reactive_expediting"}
INHERIT = {"variant_cvar": "proposed", "abl_expected_value": "proposed", "abl_no_approval": "proposed", "abl_no_state": "proposed",
           "abl_no_bim": "proposed", "abl_no_bim_planner": "proposed", "abl_static": "proposed",
           "abl_no_recourse": "proposed", "abl_no_cluster": "proposed"}


def kw_of(variant: str) -> dict:
    out = {}
    for part in variant.split("_"):
        for key in ("money_weight", "theta", "zone_scale"):
            pass
    # variants are written as key+value joined by underscores, e.g. money_weight1.0
    import re
    for key in ("money_weight", "theta", "zone_scale", "dev_q"):
        m = re.search(key + r"([0-9.]+)", variant)
        if m:
            v = float(m.group(1).rstrip("."))
            out[key] = int(v) if key == "theta" else v
    return out


def main(split, streams, tag, roster=None):
    sel = json.load(open(RESULTS / "tune_dev_selection.json", encoding="utf-8"))["chosen"]
    final = json.load(open(RESULTS / "design_final.json", encoding="utf-8"))
    for pol in (roster or ROSTER):
        src = pol if pol in TUNED else INHERIT.get(pol)
        kw = kw_of(sel[src]) if src in sel else {}
        if src == "proposed":
            kw = dict(final["proposed_kw"])
        cmd = [sys.executable, "run_exp.py", "--split", split, "--policies", pol, "--streams", streams, "--tag", tag,
               "--policy-kw", json.dumps(kw)]
        print(pol, kw, flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",") if len(sys.argv) > 4 else None)
