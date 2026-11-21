"""Stage 03: assemble the decision instance (activities, logic, lots, procurement layer) as JSON."""
from __future__ import annotations

import numpy as np
import pandas as pd

import params as P
from common import INST, PROC, RESULTS, stage_manifest, write_json


def build_schependomlaan() -> dict:
    acts = pd.read_csv(PROC / "schep_activities.csv")
    lots = pd.read_csv(PROC / "schep_lots.csv")
    acts["preds"] = acts.preds.fillna("").astype(str)
    A = []
    for _, r in acts.sort_values("s").iterrows():
        preds = [int(x) for x in r.preds.split(";") if x]
        A.append({"uid": int(r.uid), "name": r["name"], "stream": r.stream, "r": int(r.s), "d": int(r.d), "preds": preds})
    due = int(acts.loc[acts.name == "Opleveren", "f"].iat[0]) + 1  # finish index is inclusive; completion at end of day
    # lot value: package value split by volume share (count share when the package has no volume)
    L = []
    for pkg, g in lots.groupby("package"):
        vol = g.volume_m3.fillna(0)
        w = vol / vol.sum() if vol.sum() > 0.5 else g.n_elements / g.n_elements.sum()
        for (_, r), wi in zip(g.iterrows(), w):
            p = int(r.need_s) - P.JIT_OFFSET[pkg]
            L.append({"id": r.lot, "package": pkg, "supplier": pkg, "need_uid": int(r.need_uid), "need_s": int(r.need_s),
                      "zone": int(r.zone), "planned_delivery": p, "lead_time": P.LEAD_TIME[pkg],
                      "order_time": p - P.LEAD_TIME[pkg], "value": round(float(P.PACKAGE_VALUE[pkg] * wi), 2),
                      "alt_lead": P.LEAD_TIME[pkg], "equiv_group": int(r.equiv_group), "n_elements": int(r.n_elements),
                      "volume_m3": float(r.volume_m3) if pd.notna(r.volume_m3) else 0.0})
    L.sort(key=lambda x: (x["need_s"], x["id"]))
    total_value = sum(l["value"] for l in L)
    inst = {"name": "schependomlaan", "due": due, "activities": A, "lots": L, "total_value": total_value,
            "budget": round(P.BUDGET_SHARE * total_value, 2), "params": P.as_dict(), "provenance": P.PROVENANCE}
    return inst


def check_instance(inst: dict) -> dict:
    """Structural checks: acyclic logic, every lot need task exists, planned schedule meets the due date."""
    import networkx as nx
    G = nx.DiGraph()
    for a in inst["activities"]:
        G.add_node(a["uid"])
        for p in a["preds"]:
            G.add_edge(p, a["uid"])
    acyclic = nx.is_directed_acyclic_graph(G)
    byid = {a["uid"]: a for a in inst["activities"]}
    # forward pass with release dates and no delays
    S = {}
    for u in nx.topological_sort(G):
        a = byid[u]
        S[u] = max([a["r"]] + [S[p] + byid[p]["d"] for p in a["preds"]])
    finish = max(S[u] + byid[u]["d"] for u in S)
    shifted = sum(1 for u in S if S[u] > byid[u]["r"])
    return {"acyclic": acyclic, "activities": len(byid), "links": G.number_of_edges(), "lots": len(inst["lots"]),
            "nominal_finish": finish, "due": inst["due"], "nominal_meets_due": finish <= inst["due"],
            "activities_shifted_in_nominal_pass": shifted, "lot_need_tasks_exist": all(l["need_uid"] in byid for l in inst["lots"]),
            "total_value": inst["total_value"], "budget": inst["budget"]}


def main():
    inst = build_schependomlaan()
    chk = check_instance(inst)
    write_json(INST / "schependomlaan.json", inst)
    write_json(RESULTS / "s03_instance_checks.json", {"schependomlaan": chk})
    stage_manifest("s03_instance", [PROC / "schep_activities.csv", PROC / "schep_lots.csv"], [INST / "schependomlaan.json"],
                   [__file__, "src/params.py", "src/common.py"])
    print(chk)


if __name__ == "__main__":
    main()
