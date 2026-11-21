"""Stage 02: build the BIM-linked activity and lot layer of the Schependomlaan case.

Sources (all real, CC BY 4.0, TU Eindhoven Schependomlaan dataset):
  * as-planned IFC with planning data: elements, quantities, home storeys, NL-SfB layers, IfcTask,
    IfcRelAssignsToProcess (element-task links), IfcRelSequence (logic links), IfcScheduleTimeControl;
  * Synchro/MS Project XML: working calendar with holidays and start-no-earlier-than constraints;
  * planned event log: element-task rows used to cross-check the IFC links;
  * original design IFC: used only for the GUID persistence and signature reconciliation test.

Outputs: data/processed/schep_elements.csv, schep_activities.csv, schep_lots.csv, results/s02_bim_checks.json.
"""
from __future__ import annotations

import collections
import datetime as dt
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from common import (PROC, RESULTS, SCHEP, SCHEP_ASPLANNED_IFC, SCHEP_DESIGN_IFC, SCHEP_EVENTLOG_PLAN, SCHEP_XML,
                    stage_manifest, write_json)

NS = {"p": "http://schemas.microsoft.com/project"}

# Supply-package rule table: (package, supplier model in the dataset, NL-SfB prefixes, first-task keywords).
# The first matching rule assigns an element. Elements built in situ (foundations, cast floors, rebar) are not lots.
PACKAGE_RULES = [
    ("hollowcore_floor", "WAARDO-Kanaalplaatvloer", ("2320",), ("Kanaalplaatvloer",)),
    ("filigree_floor", "GEELEN-Breedplaat", ("2320",), ("breedplaat",)),
    ("cs_elements", "XELLA (per storey)", ("2120", "2220"), ("kalkzandsteen", "lijmwerk lift")),
    ("prefab_stairs", "MULTICOM-trappen+bordessen", ("2410", "2810", "3410"), ("Prefab trappen",)),
    ("steel", "BERNTS-Staalconstructie", ("2810",), ()),
    ("window_frames", "LINDEN-Kozijnen", ("3120", "3130"), ()),
    ("facade_brick", "facade masonry supplier", ("2110",), ()),
    ("roof_elements", "WILLEMSEN-Hoogbouw/Laagbouw", ("2710",), ("scharnierkap", "Zinken", "Dakramen")),
    ("concrete_bands", "MULTICOM-banden", ("4110",), ("betonband",)),
    ("roof_windows", "roof window supplier", ("3720",), ()),
    ("railings", "FEK-hekwerken", ("3410",), ()),
    ("lift", "lift supplier", ("6610",), ()),
]


def wd_calendar(xml_root) -> list[dt.date]:
    """Working dates of the project base calendar (Mon-Fri minus listed exceptions)."""
    cal = xml_root.find("p:Calendars/p:Calendar", NS)
    off = set()
    for wd in cal.findall("p:WeekDays/p:WeekDay", NS):
        if wd.find("p:DayType", NS).text == "0" and wd.find("p:DayWorking", NS).text == "0":
            tp = wd.find("p:TimePeriod", NS)
            a = dt.date.fromisoformat(tp.find("p:FromDate", NS).text[:10])
            b = dt.date.fromisoformat(tp.find("p:ToDate", NS).text[:10])
            d = a
            while d <= b:
                off.add(d)
                d += dt.timedelta(days=1)
    days, d = [], dt.date(2015, 2, 16)
    while d <= dt.date(2017, 12, 31):
        if d.weekday() < 5 and d not in off:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


class WDIndex:
    def __init__(self, days: list[dt.date], origin: dt.date):
        self.days = days
        self.pos = {d: i for i, d in enumerate(days)}
        self.o = self.idx(origin)

    def idx(self, d: dt.date) -> int:
        while d not in self.pos:  # roll forward to the next working day
            d += dt.timedelta(days=1)
        return self.pos[d]

    def __call__(self, d: dt.date) -> int:
        return self.idx(d) - self.o


def parse_xml_tasks(root) -> pd.DataFrame:
    rows = []
    for t in root.findall("p:Tasks/p:Task", NS):
        g = lambda k: (t.find("p:" + k, NS).text if t.find("p:" + k, NS) is not None else None)
        if not g("Name"):
            continue
        dur_h = 0.0
        m = re.match(r"PT(\d+)H(\d+)M", g("Duration") or "PT0H0M")
        if m:
            dur_h = int(m.group(1)) + int(m.group(2)) / 60.0
        rows.append(dict(uid=int(g("UID")), name=g("Name").strip(), start=g("Start")[:10], finish=g("Finish")[:10],
                         dur_d=dur_h / 8.0, ctype=int(g("ConstraintType") or 0),
                         preds=[int(p.find("p:PredecessorUID", NS).text) for p in t.findall("p:PredecessorLink", NS)]))
    return pd.DataFrame(rows)


def ifc_tables():
    import ifcopenshell
    import ifcopenshell.util.element as ue
    f = ifcopenshell.open(str(SCHEP_ASPLANNED_IFC))
    # tasks with schedule times
    tasks = []
    for t in f.by_type("IfcTask"):
        tc = None
        for rel in getattr(t, "HasAssignments", []) or []:
            pass
        tasks.append(dict(task_gid=t.GlobalId, task_id=t.TaskId, name=(t.Name or "").strip(), order=int(t.Description or 0)))
    tdf = pd.DataFrame(tasks)
    # schedule time controls are attached through IfcRelAssignsTasks (IFC2x3)
    times = {}
    for r in f.by_type("IfcRelAssignsTasks"):
        stc = r.TimeForTask
        for obj in r.RelatedObjects:
            if obj.is_a("IfcTask") and stc is not None:
                def conv(dtt):
                    if dtt is None:
                        return None
                    d = dtt.DateComponent
                    return dt.date(d.YearComponent, d.MonthComponent, d.DayComponent)
                times[obj.GlobalId] = (conv(stc.ScheduleStart), conv(stc.ScheduleFinish),
                                       (stc.ScheduleDuration or 0) / 28800.0)
    tdf["start"] = tdf.task_gid.map(lambda g: times.get(g, (None, None, None))[0])
    tdf["finish"] = tdf.task_gid.map(lambda g: times.get(g, (None, None, None))[1])
    tdf["dur_d"] = tdf.task_gid.map(lambda g: times.get(g, (None, None, None))[2])
    seq = [(r.RelatingProcess.TaskId, r.RelatedProcess.TaskId, r.SequenceType, float(r.TimeLag or 0))
           for r in f.by_type("IfcRelSequence")]
    links = []
    for r in f.by_type("IfcRelAssignsToProcess"):
        tid = r.RelatingProcess.TaskId
        for o in r.RelatedObjects:
            links.append((o.GlobalId, tid))
    els = []
    for e in f.by_type("IfcProduct"):
        if e.is_a() in ("IfcBuilding", "IfcBuildingStorey", "IfcSite"):
            continue
        ps = ue.get_psets(e).get("SynchroResourceProperty", {})
        def q(*keys):
            for k in keys:
                for kk, vv in ps.items():
                    if kk.endswith(k) and isinstance(vv, (int, float)):
                        return float(vv)
            return np.nan
        els.append(dict(guid=e.GlobalId, ifc_class=e.is_a(), name=e.Name, storey=ps.get("[ArchiCADProperties]Home Story"),
                        layer=ps.get("[ArchiCADProperties]Layer", ""), material=ps.get("[ArchiCADProperties]Building Material / Composite", ""),
                        volume=q("GrossVolume", "Gross Volume of the Slab", "]Volume", "Net Volume"), area=q("]Area", "Surface Area"),
                        length=q("]Length", "3D Length")))
    return tdf, pd.DataFrame(seq, columns=["pred", "succ", "type", "lag"]), pd.DataFrame(links, columns=["guid", "task_id"]), pd.DataFrame(els)


def design_signature_match(asp: pd.DataFrame) -> dict:
    """GUID persistence and signature matching between the design export and the as-planned export."""
    import ifcopenshell
    import ifcopenshell.util.element as ue
    f = ifcopenshell.open(str(SCHEP_DESIGN_IFC))
    rows = []
    for e in f.by_type("IfcBuildingElement") + f.by_type("IfcBuildingElementProxy"):
        ps = ue.get_psets(e)
        vol = np.nan
        for pname, pv in ps.items():
            for k, v in pv.items():
                if k in ("GrossVolume", "Volume", "NetVolume") and isinstance(v, (int, float)):
                    vol = float(v)
                    break
            if not np.isnan(vol):
                break
        st = None
        for rel in getattr(e, "ContainedInStructure", []) or []:
            st = rel.RelatingStructure.Name
        rows.append(dict(guid=e.GlobalId, ifc_class=e.is_a().replace("StandardCase", ""), storey=st, volume=vol, name=e.Name))
    d = pd.DataFrame(rows)
    a = asp.copy()
    a["ifc_class"] = a["ifc_class"].str.replace("StandardCase", "")
    guid_overlap = len(set(d.guid) & set(a.guid))
    a["name"] = a["name"].astype(str).str.replace(r"_\(#\d+\)$", "", regex=True)
    key = lambda x: x["ifc_class"] + "|" + x["storey"].astype(str) + "|" + x["name"].astype(str) + "|" + x["volume"].round(2).astype(str)
    d["sig"], a["sig"] = key(d), key(a)
    dc, ac = d.sig.value_counts(), a.sig.value_counts()
    unique_match = sum(min(dc.get(s, 0), c) for s, c in ac.items())
    return {"design_elements": int(len(d)), "asplanned_elements": int(len(a)), "guid_overlap": int(guid_overlap),
            "signature_matched_elements": int(unique_match), "signature_match_share": float(unique_match / len(a))}


def interval_logic(acts: pd.DataFrame, stream_col: str = "stream") -> list[tuple[int, int]]:
    """Immediate finish-to-start predecessors implied by planned dates inside each work stream."""
    out = []
    for _, g in acts.groupby(stream_col):
        g = g.sort_values("s")
        rec = g[["uid", "s", "f"]].to_numpy()
        for j in range(len(rec)):
            uj, sj, _ = rec[j]
            cand = [(ui, si, fi) for ui, si, fi in rec if fi < sj]
            for ui, si, fi in cand:
                # immediate: no other candidate k starting after i finishes and finishing before j starts
                if not any((sk > fi) and (fk < sj) for uk, sk, fk in cand if uk != ui):
                    out.append((int(ui), int(uj)))
    return out


STREAMS = [  # (stream, inclusive XML UID ranges)
    ("site", [(9, 14)]), ("substructure", [(16, 42)]), ("shell", [(44, 123)]), ("facade", [(126, 147), (181, 183)]),
    ("roof", [(151, 179)]), ("services", [(186, 187)]), ("finishing", [(190, 191), (195, 204)]), ("handover", [(209, 211)]),
]


def stream_of(uid: int) -> str | None:
    for s, ranges in STREAMS:
        for a, b in ranges:
            if a <= uid <= b:
                return s
    return None


def main() -> None:
    root = ET.parse(SCHEP_XML).getroot()
    days = wd_calendar(root)
    W = WDIndex(days, dt.date(2015, 2, 19))
    xt = parse_xml_tasks(root)
    xt["stream"] = xt.uid.map(stream_of)
    xt = xt[xt.stream.notna() & (xt.dur_d > 0)].copy()
    xt["s"] = xt.start.map(lambda x: W(dt.date.fromisoformat(x)))
    xt["f"] = xt.finish.map(lambda x: W(dt.date.fromisoformat(x)))
    xt["d"] = np.maximum(1, np.round(xt.dur_d)).astype(int)

    tdf, seq, links, els = ifc_tables()
    # map IFC tasks to XML tasks by name and planned start
    tdf["start_s"] = tdf.start.map(lambda x: x.isoformat() if x else None)
    m = tdf.merge(xt[["uid", "name", "start"]], left_on=["name", "start_s"], right_on=["name", "start"], how="left")
    ifc2uid = dict(zip(m.task_id, m.uid))
    n_ifc_tasks_mapped = int(m.uid.notna().sum())

    # logic: IfcRelSequence + XML explicit links + date-implied immediate predecessors within streams
    lk = set()
    for _, r in seq.iterrows():
        a, b = ifc2uid.get(r.pred), ifc2uid.get(r.succ)
        if pd.notna(a) and pd.notna(b):
            lk.add((int(a), int(b)))
    n_ifc_seq = len(lk)
    xml_links = {(p, int(r.uid)) for _, r in xt.iterrows() for p in r.preds if p in set(xt.uid)}
    lk |= xml_links
    date_links = set(interval_logic(xt))
    lk |= date_links
    # stream entry tasks depend on the latest-finishing task of other streams before their start
    for s, g in xt.groupby("stream"):
        first = g[g.s == g.s.min()]
        for _, r in first.iterrows():
            prev = xt[(xt.f < r.s) & (xt.stream != s)]
            if len(prev):
                fmax = prev.f.max()
                for u in prev[prev.f == fmax].uid:
                    lk.add((int(u), int(r.uid)))
    # handover inspection follows the last task of every stream
    voor = int(xt[xt.name == "Voorschouw"].uid.iat[0])
    for s, g in xt.groupby("stream"):
        if s == "handover":
            continue
        for u in g[g.f == g.f.max()].uid:
            lk.add((int(u), voor))
    # drop links that contradict planned dates (successor planned to start before predecessor finishes)
    pos = xt.set_index("uid")
    links_ok = sorted((a, b) for a, b in lk if a in pos.index and b in pos.index and a != b and pos.at[a, "f"] < pos.at[b, "s"] + 1)
    dropped = sorted(set((a, b) for a, b in lk if a in pos.index and b in pos.index) - set(links_ok))

    # elements, packages and lots
    ev = pd.read_csv(SCHEP_EVENTLOG_PLAN)
    ev_links = set(zip(ev.GUID, ev.Task))
    ifc_links = set(zip(links.guid, links.task_id))
    link_agree = len(ev_links & ifc_links) / max(1, len(ev_links | ifc_links))
    links["uid"] = links.task_id.map(ifc2uid)
    links = links.dropna(subset=["uid"])
    links["uid"] = links.uid.astype(int)
    afbouw = set(xt[xt.stream == "finishing"].uid)
    lk_el = links.merge(xt[["uid", "name", "s", "stream"]], on="uid")
    nonfin = lk_el[~lk_el.uid.isin(afbouw)].sort_values("s")
    tasks_of = nonfin.groupby("guid")[["uid", "name"]].apply(lambda x: list(zip(x["uid"], x["name"])))
    els = els.set_index("guid")
    els["code"] = els.layer.str[:4]
    els["first_uid"] = pd.Series(dtype="float64")
    els["first_task"] = None
    els["package"] = None
    for gid, r in els.iterrows():
        tl = tasks_of.get(gid, [])
        for p, sup, codes, kws in PACKAGE_RULES:
            if r.code not in codes:
                continue
            hit = [t for t in tl if (not kws) or any(k.lower() in t[1].lower() for k in kws)]
            if kws and not hit:
                continue
            els.at[gid, "package"] = p
            if hit:
                els.at[gid, "first_uid"], els.at[gid, "first_task"] = hit[0]
            break
    sup_of = {p: sup for p, sup, _, _ in PACKAGE_RULES}
    storeys = {"-1 fundering": -1, "00 begane grond": 0, "01 eerste verdieping": 1, "02 tweede verdieping": 2,
               "03 derde verdieping": 3, "04 dak": 4}
    els["level"] = els.storey.map(storeys)
    # packages whose elements carry no installation task in the IFC use the task named in the schedule
    task_for_pkg = {"railings": [182, 183], "lift": [187], "roof_windows": [163, 164]}
    for p, uids in task_for_pkg.items():
        sel = els.package.eq(p) & els.first_uid.isna()
        if sel.any():
            # split by storey half: lower storeys to the first task, upper to the second when two tasks exist
            if len(uids) == 2:
                lv = els.loc[sel, "level"].fillna(0)
                els.loc[sel, "first_uid"] = np.where(lv <= 1, uids[0], uids[1])
            else:
                els.loc[sel, "first_uid"] = uids[0]
    els["first_uid"] = els.first_uid.astype("Int64")
    lots_src = els[els.package.notna() & els.first_uid.notna()].reset_index()
    lots_src["zone"] = lots_src.level.fillna(0).astype(int)
    # one lot per package and consuming task (each task instance is one storey/zone of a repeated trade)
    lots_src["lot"] = lots_src.package + "@" + lots_src.first_uid.astype(int).astype(str)
    g = lots_src.groupby("lot")
    lot = pd.DataFrame({
        "package": g.package.first(), "supplier": g.package.first().map(sup_of),
        "need_uid": g.first_uid.first().astype(int), "zone": g.zone.agg(lambda x: int(pd.Series(x).mode().iat[0])),
        "n_elements": g.size(), "volume_m3": g.volume.sum(min_count=1), "area_m2": g.area.sum(min_count=1),
        "signature": g[["ifc_class", "volume"]].apply(lambda x: "|".join(sorted(f"{c}:{round(v, 2) if pd.notna(v) else 'na'}" for c, v in zip(x.ifc_class, x.volume)))),
        "guids": g.guid.agg(lambda x: ";".join(x)),
    }).reset_index()
    # consuming activities: every non-finishing task linked to any element of the lot
    cons = lk_el.merge(lots_src[["guid", "lot"]], on="guid")
    cons = cons[~cons.uid.isin(afbouw)]
    lot["consumers"] = lot.lot.map(cons.groupby("lot").uid.agg(lambda x: ";".join(map(str, sorted(set(x))))))
    lot["consumers"] = lot.apply(lambda r: r.consumers if isinstance(r.consumers, str) else str(r.need_uid), axis=1)
    lot["need_s"] = lot.need_uid.map(pos["s"])
    lot = lot.sort_values(["need_s", "package"]).reset_index(drop=True)
    # equivalence groups: identical element signatures within a package (BIM type equivalence)
    lot["equiv_group"] = lot.groupby(["package", "signature"]).ngroup()
    lot["equiv_size"] = lot.groupby("equiv_group").lot.transform("size")

    # activities table
    acts = xt[["uid", "name", "stream", "s", "f", "d", "ctype"]].copy()
    acts["preds"] = acts.uid.map(lambda u: ";".join(str(a) for a, b in links_ok if b == u))
    due = int(xt[xt.name == "Opleveren"].f.iat[0])

    elements_out = els.reset_index()[["guid", "ifc_class", "name", "storey", "level", "layer", "material", "volume", "area",
                                      "first_uid", "first_task", "package"]]
    elements_out.to_csv(PROC / "schep_elements.csv", index=False)
    acts.to_csv(PROC / "schep_activities.csv", index=False)
    lot.to_csv(PROC / "schep_lots.csv", index=False)

    # checks (RQ1, computational)
    assigned = els.package.notna().sum()
    dup = lots_src.guid.duplicated().sum()
    vol_el = lots_src.volume.sum()
    vol_lot = lot.volume_m3.sum()
    checks = {
        "ifc_elements": int(len(els)), "ifc_tasks": int(len(tdf)), "ifc_tasks_mapped_to_xml": n_ifc_tasks_mapped,
        "xml_activities_used": int(len(acts)), "ifc_sequence_links_mapped": n_ifc_seq, "xml_explicit_links": len(xml_links),
        "date_implied_links": len(date_links), "links_total": len(links_ok), "links_dropped_contradicting_dates": len(dropped),
        "eventlog_ifc_link_jaccard": float(link_agree), "eventlog_link_rows": int(len(ev_links)), "ifc_link_rows": int(len(ifc_links)),
        "elements_assigned_to_packages": int(assigned), "elements_in_lots": int(len(lots_src)), "duplicate_element_assignments": int(dup),
        "volume_reconciliation_abs_diff_m3": float(abs(vol_el - vol_lot)), "lots": int(len(lot)), "packages": int(lot.package.nunique()),
        "lots_with_equivalent_partner": int((lot.equiv_size > 1).sum()),
        "due_wd": due, "planned_start": "2015-02-19", "planned_finish": str(xt[xt.name == "Opleveren"].finish.iat[0]),
        "elements_by_package": {k: int(v) for k, v in els.package.value_counts().items()},
        "design_vs_asplanned": design_signature_match(els.reset_index()),
    }
    write_json(RESULTS / "s02_bim_checks.json", checks)
    stage_manifest("s02_bim", [SCHEP_ASPLANNED_IFC, SCHEP_DESIGN_IFC, SCHEP_XML, SCHEP_EVENTLOG_PLAN],
                   [PROC / "schep_elements.csv", PROC / "schep_activities.csv", PROC / "schep_lots.csv", RESULTS / "s02_bim_checks.json"],
                   [__file__, "src/common.py"])
    print({k: v for k, v in checks.items() if not isinstance(v, dict)})
    print(checks["elements_by_package"]); print(checks["design_vs_asplanned"])
    print(lot[["lot", "package", "need_uid", "zone", "n_elements", "volume_m3", "need_s", "equiv_size", "consumers"]].to_string())


if __name__ == "__main__":
    main()
