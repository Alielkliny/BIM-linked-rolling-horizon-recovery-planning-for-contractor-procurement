"""Scenario-based recovery MILP solved with HiGHS (highspy), and the schedule passes the rule-based policies use.

Two schedule back-ends share one decision layer:
  full       one start variable per open activity and scenario (the formulation as written in the paper);
  condensed  start variables only for activities fed by actionable lots; all other activities enter through
             max-plus constants and longest-path distances of the static network. The two give the same optimum
             (tested in tests/test_condensed.py on recorded snapshots).
"""
from __future__ import annotations

import math

import highspy
import numpy as np

import params as P


_HIGHS = {}


def _highs(time_limit, gap):
    """One Highs object per process and option set; clearModel between solves (identical results, less overhead)."""
    key = (float(time_limit), float(gap))
    h = _HIGHS.get(key)
    if h is None:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", gap)
        h.setOptionValue("threads", 1)
        _HIGHS[key] = h
    else:
        h.clearModel()
    return h


class LP:
    def __init__(self):
        self.lb, self.ub, self.c, self.integ = [], [], [], []
        self.r_lo, self.r_hi, self.r_idx, self.r_val = [], [], [], []

    def var(self, lb=0.0, ub=highspy.kHighsInf, c=0.0, integer=False) -> int:
        self.lb.append(lb)
        self.ub.append(ub)
        self.c.append(c)
        self.integ.append(1 if integer else 0)
        return len(self.lb) - 1

    def row(self, idx, val, lo=-highspy.kHighsInf, hi=highspy.kHighsInf) -> None:
        self.r_idx.append(idx)
        self.r_val.append(val)
        self.r_lo.append(lo)
        self.r_hi.append(hi)

    def solve(self, time_limit=20.0, gap=1e-4):
        h = _highs(time_limit, gap)
        lp = highspy.HighsLp()
        n, m = len(self.lb), len(self.r_lo)
        lp.num_col_, lp.num_row_ = n, m
        lp.col_cost_ = np.asarray(self.c, dtype=float)
        lp.col_lower_ = np.asarray(self.lb, dtype=float)
        lp.col_upper_ = np.asarray(self.ub, dtype=float)
        lp.row_lower_ = np.asarray(self.r_lo, dtype=float)
        lp.row_upper_ = np.asarray(self.r_hi, dtype=float)
        lens = np.fromiter((len(ii) for ii in self.r_idx), dtype=np.int64, count=m)
        starts = np.zeros(m + 1, dtype=np.int32)
        np.cumsum(lens, out=starts[1:])
        idx = np.fromiter((j for ii in self.r_idx for j in ii), dtype=np.int32, count=int(starts[-1]))
        val = np.fromiter((v for vv in self.r_val for v in vv), dtype=float, count=int(starts[-1]))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = starts
        lp.a_matrix_.index_ = idx
        lp.a_matrix_.value_ = val
        if any(self.integ):
            lp.integrality_ = [highspy.HighsVarType.kInteger if z else highspy.HighsVarType.kContinuous for z in self.integ]
        h.passModel(lp)
        h.run()
        st = h.getModelStatus()
        info = h.getInfo()
        ok = st in (highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kTimeLimit) and info.primal_solution_status == 2
        x = np.array(h.getSolution().col_value) if ok else None
        return x, str(st), (info.mip_gap if any(self.integ) else 0.0)


# ------------------------------------------------------------------------------------------- schedule structure
class Net:
    """Static activity network of an instance: topological order, successors, sinks and longest paths."""

    def __init__(self, inst: dict):
        import networkx as nx
        self.acts = {a["uid"]: a for a in inst["activities"]}
        G = nx.DiGraph()
        for a in inst["activities"]:
            G.add_node(a["uid"])
            for p in a["preds"]:
                G.add_edge(p, a["uid"])
        self.order = list(nx.topological_sort(G))
        self.succ = {u: list(G.successors(u)) for u in G}
        self.sinks = [u for u in G if G.out_degree(u) == 0]
        self.due = inst["due"]
        self.lot_pkg = {l["id"]: l["package"] for l in inst["lots"]}
        # start-to-start longest path lengths (minus infinity when unreachable) and tails to project completion
        self.idx = {u: k for k, u in enumerate(self.order)}
        n = len(self.order)
        D = np.full((n, n), -np.inf)
        np.fill_diagonal(D, 0.0)
        for u in reversed(self.order):
            ku = self.idx[u]
            du = self.acts[u]["d"]
            for v in self.succ[u]:
                D[ku] = np.maximum(D[ku], du + D[self.idx[v]])
        self.D = D
        sink_idx = [self.idx[w] for w in self.sinks]
        dsink = np.array([self.acts[w]["d"] for w in self.sinks], dtype=float)
        self.tail = np.max(D[:, sink_idx] + dsink[None, :], axis=1)

    def dist(self, a, b) -> float:
        return float(self.D[self.idx[a], self.idx[b]])


def forward(net: Net, snap, arr: dict) -> tuple[dict, dict]:
    """Earliest starts given lot arrival estimates (arr: need_uid -> latest arrival). Started activities fixed."""
    t = math.ceil(snap.t)
    S, F = {}, {}
    for u in net.order:
        a = net.acts[u]
        if u in snap.started:
            S[u] = snap.started[u]
        else:
            S[u] = max([a["r"], t] + [F[p] for p in a["preds"]] + [arr.get(u, -1)])
        F[u] = S[u] + a["d"]
    return S, F


def backward(net: Net, F: dict) -> dict:
    """Latest starts that keep the due date."""
    LS = {}
    for u in reversed(net.order):
        a = net.acts[u]
        lf = min([LS[v] for v in net.succ[u]] + [net.due])
        LS[u] = lf - a["d"]
    return LS


def need_map(snap, bim_linked: bool) -> dict:
    """Activity each open lot is planned to feed. Without BIM linkage only the package's earliest open need is known."""
    if bim_linked:
        return {lv.id: lv.need_uid for lv in snap.lots}
    acts = {a["uid"]: a for a in snap.activities}
    first = {}
    for lv in snap.lots:
        if lv.received or lv.need_uid in snap.started:
            continue
        u = first.get(lv.package)
        if u is None or acts[lv.need_uid]["r"] < acts[u]["r"]:
            first[lv.package] = lv.need_uid
    return {lv.id: first.get(lv.package, lv.need_uid) for lv in snap.lots}


# ------------------------------------------------------------------------------------------- recovery MILP
def build_and_solve(net: Net, snap, scen: dict, lam: float, alpha: float = P.CVAR_ALPHA, need: dict | None = None,
                    delay_cost: float = P.DELAY_COST, money_weight: float = 1.0, reserve: float = 0.0,
                    time_limit: float = 20.0, backend: str = "condensed"):
    """Two-stage scenario MILP (objective in working days).

    First stage (now): expedite e, request approval q, order the approved alternative o.
    Second stage (each scenario s): expedite once the delay is revealed e2_s, order the alternative when its approval
    arrives r_s, arrival A_s, starts S_s, completion C_s, delay D_s and the CVaR excess xi_s.
      min  sum_s p D_s + lam (eta + 1/(1-alpha) sum_s p xi_s) + mu/c_d (first-stage cost + sum_s p second-stage cost)
    with mu = money_weight, c_d = delay_cost, and first-stage spending capped at (1 - reserve) of the remaining budget.
    """
    t = math.ceil(snap.t)
    need = need or {}
    S_n = next(iter(scen.values()))["orig"].shape[0] if scen else 1
    p = 1.0 / S_n
    M = net.due + 600
    lp = LP()
    cost_w = money_weight / delay_cost
    first, first_cost_terms = {}, []
    for lid, sc in scen.items():
        d = {}
        if sc["can_exp"]:
            d["e"] = lp.var(0, 1, sc["c_exp"] * cost_w, True); first_cost_terms.append((d["e"], sc["c_exp"]))
        if sc["can_req"]:
            d["q"] = lp.var(0, 1, sc["c_req"] * cost_w, True); first_cost_terms.append((d["q"], sc["c_req"]))
        if sc["can_order"]:
            d["o"] = lp.var(0, 1, sc["c_alt"] * cost_w, True); first_cost_terms.append((d["o"], sc["c_alt"]))
        first[lid] = d
    if first_cost_terms and reserve > 0:
        lp.row([v for v, _ in first_cost_terms], [c for _, c in first_cost_terms], hi=max(0.0, (1 - reserve) * snap.budget_remaining))
    eta = lp.var(0, highspy.kHighsInf, lam, False)
    started, acts = snap.started, net.acts
    # lots: per scenario either constant (folded into the need activity's bound) or modelled with variables
    const_arr, lot_vars = {}, []
    for lid, sc in scen.items():
        u = need.get(lid, sc["need"])
        if u in started:
            continue
        fd = first[lid]
        act_s = np.logical_or(bool(fd) or sc["pending"], sc["rec_ok"])
        cst = np.where(act_s, -1.0, sc["orig"])
        const_arr[u] = np.maximum(const_arr.get(u, np.full(S_n, -1.0)), cst)
        if act_s.any():
            lot_vars.append((lid, u, sc, fd, act_s))
    if backend == "condensed":
        c = {}
        for u in net.order:
            a = acts[u]
            if u in started:
                c[u] = np.full(S_n, float(started[u]))
                continue
            v = np.full(S_n, float(max(a["r"], t)))
            for pp in a["preds"]:
                v = np.maximum(v, c[pp] + acts[pp]["d"])
            if u in const_arr:
                v = np.maximum(v, const_arr[u])
            c[u] = v
        c_end = np.max(np.stack([c[w] + acts[w]["d"] for w in net.sinks]), axis=0)
        K = sorted({u for _, u, _, _, _ in lot_vars}, key=lambda u: net.idx[u])
        pairs = [(a_, b_, net.dist(a_, b_)) for a_ in K for b_ in K if a_ != b_ and net.dist(a_, b_) > -np.inf]
        tails = {a_: float(net.tail[net.idx[a_]]) for a_ in K}
    else:
        started_fin = {u: s0 + acts[u]["d"] for u, s0 in started.items()}
        open_acts = [u for u in net.order if u not in started]
        sinks_open = [u for u in net.sinks if u not in started]
        sink_const = max([started_fin[u] for u in net.sinks if u in started_fin] + [0])
    for s in range(S_n):
        if backend == "condensed":
            Sv = {u: lp.var(float(c[u][s]), M, 0.0, False) for u in K}
            for a_, b_, dab in pairs:
                lp.row([Sv[b_], Sv[a_]], [1.0, -1.0], lo=dab)
        else:
            Sv = {}
            for u in open_acts:
                a = acts[u]
                lb = max([a["r"], t, float(const_arr[u][s]) if u in const_arr else -1.0] +
                         [started_fin[pp] for pp in a["preds"] if pp in started_fin])
                Sv[u] = lp.var(lb, M, 0.0, False)
            for u in open_acts:
                for pp in acts[u]["preds"]:
                    if pp in Sv:
                        lp.row([Sv[u], Sv[pp]], [1.0, -1.0], lo=acts[pp]["d"])
        second_cost = []
        for lid, u, sc, fd, act_s in lot_vars:
            if not act_s[s]:
                continue
            A = lp.var(t, M, 0.0, False)
            r = None
            if (sc["can_req"] or sc["pending"]) and not sc["alt_ordered"] and ("r_ok" not in sc or bool(sc["r_ok"][s])):
                r = lp.var(0, 1, p * sc["c_alt"] * cost_w, True); second_cost.append((r, sc["c_alt"]))
            e2 = None
            if bool(sc["rec_ok"][s]) and sc["exp_rec"][s] < sc["orig"][s]:
                e2 = lp.var(0, 1, p * sc["c_exp"] * cost_w, True); second_cost.append((e2, sc["c_exp"]))
                if "e" in fd:
                    lp.row([fd["e"], e2], [1.0, 1.0], hi=1.0)
            # request-at-notice recourse: file the approval request when the delay is revealed, then order the
            # alternative once approved (only for lots whose approval has not been requested)
            q2 = None
            if bool(sc["rec_ok"][s]) and sc.get("req_later_ok", False) and sc["altrec2"][s] < sc["orig"][s] and (
                    "q2_ok" not in sc or bool(sc["q2_ok"][s])):
                q2 = lp.var(0, 1, p * (sc["c_req"] + sc["c_alt"]) * cost_w, True)
                second_cost.append((q2, sc["c_req"] + sc["c_alt"]))
                if "q" in fd:
                    lp.row([fd["q"], q2], [1.0, 1.0], hi=1.0)
                if e2 is not None:
                    lp.row([e2, q2], [1.0, 1.0], hi=1.0)
            # tight big-M values: each switch only needs to relax its row down to the arrival floor t
            m_orig = max(0.0, float(sc["orig"][s]) - t)
            idx, val = [A], [1.0]
            if "e" in fd:
                idx.append(fd["e"]); val.append(float(sc["orig"][s] - sc["exp"][s]))
            if e2 is not None:
                idx.append(e2); val.append(float(sc["orig"][s] - sc["exp_rec"][s]))
            if "o" in fd:
                idx.append(fd["o"]); val.append(m_orig)
            if r is not None:
                idx.append(r); val.append(m_orig)
            if q2 is not None:
                idx.append(q2); val.append(m_orig)
            lp.row(idx, val, lo=float(sc["orig"][s]))
            if q2 is not None:
                m_q2 = max(0.0, float(sc["altrec2"][s]) - t)
                lp.row([A, q2], [1.0, -m_q2], lo=float(sc["altrec2"][s]) - m_q2)
            if "o" in fd:
                m_alt = max(0.0, float(sc["altnow"][s]) - t)
                lp.row([A, fd["o"]], [1.0, -m_alt], lo=float(sc["altnow"][s]) - m_alt)
            if r is not None:
                m_rec = max(0.0, float(sc["altrec"][s]) - t)
                lp.row([A, r], [1.0, -m_rec], lo=float(sc["altrec"][s]) - m_rec)
                if "o" in fd:
                    lp.row([fd["o"], r], [1.0, 1.0], hi=1.0)
                if "q" in fd:
                    lp.row([r, fd["q"]], [1.0, -1.0], hi=1.0 if sc["pending"] else 0.0)
                elif not sc["pending"]:
                    lp.row([r], [1.0], hi=0.0)
            lp.row([Sv[u], A], [1.0, -1.0], lo=0.0)
        if backend == "condensed":
            C = lp.var(float(c_end[s]), M, 0.0, False)
            for a_ in K:
                lp.row([C, Sv[a_]], [1.0, -1.0], lo=tails[a_])
        else:
            C = lp.var(sink_const, M, 0.0, False)
            for u in sinks_open:
                lp.row([C, Sv[u]], [1.0, -1.0], lo=acts[u]["d"])
        D = lp.var(0, M, p, False)
        lp.row([D, C], [1.0, -1.0], lo=-float(net.due))
        xi = lp.var(0, M, lam * p / (1 - alpha), False)
        lp.row([xi, D, eta], [1.0, -1.0, 1.0], lo=0.0)
        idx = [v for v, _ in first_cost_terms] + [v for v, _ in second_cost]
        val = [cc for _, cc in first_cost_terms] + [cc for _, cc in second_cost]
        if idx:
            lp.row(idx, val, hi=max(0.0, snap.budget_remaining))
    x, status, gap = lp.solve(time_limit=time_limit)
    out = []
    if x is None:
        return out, {"status": status, "gap": gap}
    for lid, d in first.items():
        for k, kind in (("e", "expedite"), ("q", "request"), ("o", "order_alt")):
            if k in d and x[d[k]] > 0.5:
                out.append((kind, lid))
    return out, {"status": status, "gap": gap, "n_vars": len(lp.lb), "n_rows": len(lp.r_lo),
                 "n_int": int(sum(lp.integ)), "obj": float(np.dot(np.array(lp.c), x))}
