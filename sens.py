"""Sensitivity study: environment assumptions varied one at a time on development streams 10-17 (72 episodes); the
planners keep their calibration and tuned settings, so each setting also tests misspecification."""
import json
import subprocess
import sys

from run_split import kw_of

SETTINGS = {
    "alt_lead_0.5": {"alt_lead_scale": 0.5}, "alt_lead_1.5": {"alt_lead_scale": 1.5},
    "expedite_0.2": {"expedite_share": 0.2}, "expedite_0.6": {"expedite_share": 0.6},
    "cancel_0.05": {"cancel_fee": 0.05}, "cancel_0.20": {"cancel_fee": 0.20},
    "budget_0.5": {"budget_scale": 0.5}, "budget_2.0": {"budget_scale": 2.0},
    "approval_0.5": {"approval_scale": 0.5}, "approval_1.5": {"approval_scale": 1.5},
    "partial_notice": {"notice_fraction": [0.3, 1.0]}, "independent_lateness": {"independent_lateness": True},
    "record_lag_3": {"reclag_mean": 3}, "record_lag_8": {"reclag_mean": 8},
    "notice_window_5": {"notice_max": 5}, "notice_window_20": {"notice_max": 20},
    "submittal_250": {"approval_cost": 250.0}, "submittal_1000": {"approval_cost": 1000.0},
}
POLS = ["proposed", "buffer_management", "reported_rescheduling", "budgeted_robust"]


def main(names=None):
    sel = json.load(open("../results/tune_dev_selection.json"))["chosen"]
    for name, ov in SETTINGS.items():
        if names and name not in names:
            continue
        for pol in POLS:
            kw = kw_of(sel[pol]) if pol in sel else {}
            if pol == "proposed":
                kw = json.load(open("../results/design_final.json"))["proposed_kw"]
            subprocess.run([sys.executable, "run_exp.py", "--split", "dev", "--policies", pol, "--streams", "10-17",
                            "--tag", "sens", "--policy-kw", json.dumps(kw), "--overrides", json.dumps(ov), "--variant", name],
                           check=True)
    print("SENS_DONE")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
