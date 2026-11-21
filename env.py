"""Discrete-event evaluation environment (SimPy) for supply recovery policies.

The environment holds the true state: lot arrival times, approval outcomes and activity progress. It reveals
information to the planner only through time-stamped reports in the temporal dependency graph, validates every
recommended action against the true state and the remaining budget with its own checker, and records outcomes.
All exogenous randomness is drawn before the run (common random numbers across policies).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import simpy

import params as P
from tdg import Fact, TemporalDependencyGraph
from uncertainty import pool

SEQS = ("single", "sequential", "compound")
SEVS = {"low": (0.10, 0.50), "medium": (0.50, 0.90), "high": (0.90, 0.99)}


# ----------------------------------------------------------------------------------------------- exogenous draws
def draw_exogenous(inst: dict, pool_name: str, seq: str, sev: str, seed: int, overrides: dict | None = None) -> dict:
    o = overrides or {}
    rng = np.random.default_rng(seed)
    pl = pool(pool_name)
    s_late = np.clip(pl["late"], 0, None)
    lots = inst["lots"]
    n = len(lots)
    # supplier-month groups: lots of one supplier with planned delivery in the same month (21 wd) draw their lateness
    # jointly from one resampled SCMS vendor-month block of the pool, so lateness clusters as in the record; a group
    # with one lot draws a single shipment row. Record lags come from shipment rows of the same pool.
    from uncertainty import vendor_blocks
    sh = _pool_rows(pool_name)
    blocks, _ = vendor_blocks(pool_name)
    grp_key = [(l["supplier"], l["planned_delivery"] // 21) for l in lots]
    groups = {}
    for i, k in enumerate(grp_key):
        groups.setdefault(k, []).append(i)
    base_late = np.zeros(n)
    receipt_lag = np.zeros(n)
    for k, idx in groups.items():
        if len(idx) == 1 or o.get("independent_lateness", False):
            rows = rng.integers(0, len(sh), size=len(idx))
            base_late[idx] = sh[rows, 0]
        else:
            b = blocks[int(rng.integers(0, len(blocks)))]
            base_late[idx] = rng.choice(b, size=len(idx))
        receipt_lag[idx] = sh[rng.integers(0, len(sh), size=len(idx)), 1]
    nf_lo, nf_hi = o.get("notice_fraction", P.NOTICE_FRACTION)
    ex = {
        "base_late": base_late.astype(int),
        "notice_lead": rng.integers(0, o.get("notice_max", P.NOTICE_MAX) + 1, size=n),
        "notice_frac": rng.uniform(nf_lo, nf_hi, size=n),
        "reclag": rng.choice(np.clip(pl["reclag"], 0, None), size=(n, 8)).astype(int) * int(o.get("reclag_scale", 1)),
        "receipt_lag": receipt_lag.astype(int),
        "approval_lag": np.maximum(1, np.round(rng.choice(pl["approval"], size=n) * o.get("approval_scale", 1.0))).astype(int),
        "alt_late": rng.choice(s_late, size=n).astype(int),
        "disruptions": [],
    }
    lo, hi = SEVS[sev]
    pos = pl["late_pos"]
    def mag():
        return int(max(1, round(float(np.quantile(pos, rng.uniform(lo, hi))))))
    sups = sorted({l["supplier"] for l in lots})
    def pick_supplier(tau, exclude=()):
        cand = [s for s in sups if s not in exclude and any(l["supplier"] == s and l["planned_delivery"] > tau + P.TRANSIT for l in lots)]
        return cand[int(rng.integers(0, len(cand)))] if cand else None
    tau1 = int(rng.integers(10, 141))
    k1 = pick_supplier(tau1)
    d1 = {"tau": tau1, "kind": "supplier", "supplier": k1, "delta": mag(), "frac": float(rng.uniform(nf_lo, nf_hi)),
          "lag": int(rng.choice(pl["reclag"]))}
    ex["disruptions"].append(d1)
    if seq == "sequential":
        tau2 = tau1 + int(rng.integers(3, 16))
        k2 = pick_supplier(tau2, exclude=(k1,))
        if k2 is not None:
            ex["disruptions"].append({"tau": tau2, "kind": "supplier", "supplier": k2, "delta": mag(),
                                      "frac": float(rng.uniform(nf_lo, nf_hi)), "lag": int(rng.choice(pl["reclag"]))})
    elif seq == "compound":
        tau2 = tau1 + int(rng.integers(0, 11))
        ex["disruptions"].append({"tau": tau2, "kind": "transport", "window": 20, "delta": max(1, mag() // 2),
                                  "frac": float(rng.uniform(nf_lo, nf_hi)), "lag": int(rng.choice(pl["reclag"]))})
    ex["meta"] = {"pool": pool_name, "seq": seq, "sev": sev, "seed": seed}
    if o.get("reclag_mean"):
        # sensitivity case (declared, Amendment 2 note 6): record lags of notices and receipts drawn from a geometric
        # distribution with the given mean in working days, on a separate stream so every other draw is unchanged
        rng2 = np.random.default_rng(seed + 7919)
        pg = 1.0 / (1.0 + float(o["reclag_mean"]))
        ex["reclag"] = (rng2.geometric(pg, size=(n, 8)) - 1).astype(int)
        ex["receipt_lag"] = (rng2.geometric(pg, size=n) - 1).astype(int)
        for d in ex["disruptions"]:
            d["lag"] = int(rng2.geometric(pg) - 1)
    return ex


_ROWS = {}


def _pool_rows(name: str) -> np.ndarray:
    if name not in _ROWS:
        import pandas as pd
        from common import PROC
        from uncertainty import select
        s, _ = select(name)
        _ROWS[name] = np.column_stack([np.clip(s.lateness_wd.to_numpy(float), 0, None), np.clip(s.reclag_wd.fillna(0).to_numpy(float), 0, None)])
    return _ROWS[name]


# ----------------------------------------------------------------------------------------------- snapshot types
@dataclass
class LotView:
    id: str
    idx: int
    package: str
    need_uid: int
    P: int
    eta: float
    noticed: bool
    received: bool
    expedited: bool
    appr_state: str          # none | pending | approved
    requested_at: float | None
    alt_ordered: bool
    alt_eta: float | None
    value: float
    alt_lead: int
    equiv_group: int


@dataclass
class Snapshot:
    t: float
    due: int
    budget_remaining: float
    lots: list
    started: dict            # uid -> start time (observed)
    activities: list         # instance activities (static)
    bim_linked: bool = True


@dataclass
class Action:
    kind: str                # expedite | request | order_alt
    lot: str


# ----------------------------------------------------------------------------------------------- environment
class Episode:
    def __init__(self, inst: dict, ex: dict, policy, pooling: bool = True, overrides: dict | None = None,
                 log_actions: bool = False):
        self.inst, self.ex, self.policy = inst, ex, policy
        self.o = overrides or {}
        self.env = simpy.Environment()
        self.lots = inst["lots"]
        self.lidx = {l["id"]: i for i, l in enumerate(self.lots)}
        self.acts = {a["uid"]: a for a in inst["activities"]}
        n = len(self.lots)
        self.P = np.array([l["planned_delivery"] for l in self.lots], dtype=float)
        self.T = self.P + ex["base_late"]                     # true arrival from the current source
        self.ver = np.zeros(n, dtype=int)
        self.dispatched = np.zeros(n, dtype=bool)
        self.arrived = np.zeros(n, dtype=bool)
        self.arrival_time = np.full(n, np.nan)
        self.source = ["orig"] * n
        self.expedited = np.zeros(n, dtype=bool)
        self.requested_at = [None] * n
        self.approved_at = [None] * n
        self.alt_ordered_at = [None] * n
        self.rep_eta = self.P.copy()                          # last reported ETA (environment side)
        self.tdg = TemporalDependencyGraph(inst)
        self.spent = 0.0
        self.budget = inst["budget"] * self.o.get("budget_scale", 1.0)
        self.infeasible = 0
        self.n_actions = {"expedite": 0, "request": 0, "order_alt": 0}
        self.cost_by = {"expedite": 0.0, "request": 0.0, "order_alt": 0.0, "cancel": 0.0}
        self.started, self.finished = {}, {}
        self.pending_reviews = set()
        self.pooling = pooling
        self.log = [] if log_actions else None
        self.n_reviews = 0
        self.exp_share = self.o.get("expedite_share", P.EXPEDITE_SHARE)
        self.alt_scale = self.o.get("alt_lead_scale", 1.0)
        self.cancel_fee = self.o.get("cancel_fee", P.CANCEL_FEE)
        self.appr_cost = self.o.get("approval_cost", P.APPROVAL_COST)
        # material-ready events per activity (lot assignment through equivalence groups when pooling)
        self.need_lots = {}
        for i, l in enumerate(self.lots):
            self.need_lots.setdefault(l["need_uid"], []).append(i)
        self.ready_evt = {}
        self.assigned = {}   # lot index -> activity uid it serves
        self.served = {}     # activity uid -> set of lots assigned

    # ------------------------------------------------------------------ helpers
    def _record(self, i: int, te: float, lag: float, kind: str, value) -> None:
        rec = te + max(0, lag)
        self.tdg.add_fact(("lot", self.lots[i]["id"]), Fact(te, rec, kind, value))
        if kind in ("eta", "approval"):
            self._schedule_review(rec)

    def _schedule_review(self, t: float) -> None:
        day = math.floor(t)
        if day not in self.pending_reviews:
            self.pending_reviews.add(day)
            self.env.process(self._review_at(day + 0.5))

    def _reschedule(self, i: int) -> None:
        self.ver[i] += 1
        self.env.process(self._delivery(i, self.ver[i]))

    # ------------------------------------------------------------------ processes
    def _delivery(self, i: int, v: int):
        while True:
            t_arr = self.T[i]
            disp = t_arr - P.TRANSIT
            if not self.dispatched[i] and self.env.now < disp:
                yield self.env.timeout(disp - self.env.now)
                if self.ver[i] != v:
                    return
                self.dispatched[i] = True
                continue
            if self.env.now < t_arr:
                yield self.env.timeout(t_arr - self.env.now)
                if self.ver[i] != v:
                    return
            break
        if self.arrived[i]:
            return
        self.arrived[i] = True
        self.arrival_time[i] = self.env.now
        self._record(i, self.env.now, self.ex["receipt_lag"][i], "receipt", self.env.now)
        self._assign_material(i)

    def _assign_material(self, i: int) -> None:
        l = self.lots[i]
        if self.pooling:
            grp = [j for j, m in enumerate(self.lots) if m["equiv_group"] == l["equiv_group"]]
            # serve the earliest-need activity of the group that still lacks a lot
            needs = sorted({self.lots[j]["need_uid"] for j in grp}, key=lambda u: self.acts[u]["r"])
            for u in needs:
                req = len([j for j in grp if self.lots[j]["need_uid"] == u])
                got = len([j for j in self.served.get(u, set()) if self.lots[j]["equiv_group"] == l["equiv_group"]])
                if got < req and u not in self.started:
                    self._serve(i, u)
                    return
        self._serve(i, l["need_uid"])

    def _serve(self, i: int, u: int) -> None:
        self.assigned[i] = u
        self.served.setdefault(u, set()).add(i)
        if len(self.served[u]) >= len(self.need_lots.get(u, [])) and not self.ready_evt[u].triggered:
            self.ready_evt[u].succeed()

    def _activity(self, u: int):
        a = self.acts[u]
        evs = [self.fin_evt[p] for p in a["preds"]]
        if u in self.need_lots:
            evs.append(self.ready_evt[u])
        if evs:
            yield simpy.AllOf(self.env, evs)
        if self.env.now < a["r"]:
            yield self.env.timeout(a["r"] - self.env.now)
        self.started[u] = self.env.now
        yield self.env.timeout(a["d"])
        self.finished[u] = self.env.now
        self.fin_evt[u].succeed()

    def _disruption(self, d: dict):
        yield self.env.timeout(d["tau"])
        now = self.env.now
        for i, l in enumerate(self.lots):
            if self.arrived[i]:
                continue
            hit = False
            if d["kind"] == "supplier" and l["supplier"] == d["supplier"] and self.source[i] == "orig" and not self.dispatched[i]:
                hit = True
            if d["kind"] == "transport" and now < self.T[i] <= now + d["window"]:
                hit = True
            if hit:
                self.T[i] += d["delta"]
                self._reschedule(i)
                # first notice reports a share of the new delay; the full revision follows midway to arrival
                self.env.process(self._notice(i, now, d["lag"], d["frac"], full_at=now + max(2, (self.T[i] - now) // 2)))

    def _notice(self, i: int, te: float, lag: int, frac: float, full_at: float | None):
        if te > self.env.now:
            yield self.env.timeout(te - self.env.now)
        if self.arrived[i]:
            return
        prev = self.rep_eta[i]
        rep = prev + round(frac * max(0.0, self.T[i] - prev)) if self.T[i] > prev else self.T[i]
        self.rep_eta[i] = rep
        self._record(i, self.env.now, lag, "eta", float(rep))
        if full_at is not None:
            yield self.env.timeout(max(0, full_at - self.env.now))
            if self.arrived[i]:
                return
            self.rep_eta[i] = self.T[i]
            self._record(i, self.env.now, self.ex["reclag"][i][1], "eta", float(self.T[i]))

    def _base_notices(self):
        for i, l in enumerate(self.lots):
            b = self.ex["base_late"][i]
            if b > 0:
                te = max(0.0, self.P[i] - self.ex["notice_lead"][i], float(l["order_time"]))
                full = max(te, self.P[i])
                self.env.process(self._notice(i, te, self.ex["reclag"][i][0], self.ex["notice_frac"][i], full_at=full))
        yield self.env.timeout(0)

    def _weekly(self):
        t = 0
        while t < self.inst["due"] + 200:
            self._schedule_review(t)
            t += P.REVIEW_EVERY
            yield self.env.timeout(P.REVIEW_EVERY)

    def _review_at(self, t: float):
        yield self.env.timeout(max(0, t - self.env.now))
        if self.done_evt.triggered:
            return
        self.n_reviews += 1
        snap = self.snapshot(self.env.now)
        try:
            acts = self.policy.decide(snap)
        except Exception:  # a planner failure is recorded and treated as no recommendation
            import traceback
            acts = []
            self.policy_errors = getattr(self, "policy_errors", 0) + 1
            if not hasattr(self, "first_error"):
                self.first_error = traceback.format_exc()
        for a in acts or []:
            self.apply(a)

    # ------------------------------------------------------------------ snapshot (as-of query)
    def snapshot(self, t: float) -> Snapshot:
        views = []
        for i, l in enumerate(self.lots):
            node = ("lot", l["id"])
            r = self.tdg.latest(node, t, "receipt")
            e = self.tdg.latest(node, t, "eta")
            eta = float(e.value) if e else float(self.P[i])
            ap = self.tdg.latest(node, t, "approval")
            if self.requested_at[i] is None:
                st = "none"
            elif ap is not None:
                st = "approved"
            else:
                st = "pending"
            alt_eta = None
            if self.alt_ordered_at[i] is not None:
                alt_eta = self.alt_ordered_at[i] + l["alt_lead"] * self.alt_scale
                if e and e.event_time >= self.alt_ordered_at[i]:
                    alt_eta = float(e.value)
                eta = alt_eta
            views.append(LotView(l["id"], i, l["package"], l["need_uid"], int(self.P[i]), eta, e is not None, r is not None,
                                 bool(self.expedited[i]), st, self.requested_at[i], self.alt_ordered_at[i] is not None,
                                 alt_eta, l["value"], int(round(l["alt_lead"] * self.alt_scale)), l["equiv_group"]))
        started = {u: s for u, s in self.started.items() if s <= t}
        return Snapshot(t, self.inst["due"], self.budget - self.spent, views, started, self.inst["activities"])

    # ------------------------------------------------------------------ independent checker and execution
    def apply(self, a: Action) -> bool:
        i = self.lidx.get(a.lot)
        now = self.env.now
        if i is None or self.arrived[i]:
            return self._reject(a, "unknown or received lot")
        l = self.lots[i]
        if a.kind == "expedite":
            cost = P.EXPEDITE_COST * l["value"]
            if self.expedited[i] or self.dispatched[i] or self.source[i] != "orig":
                return self._reject(a, "not expeditable")
            if self.spent + cost > self.budget + 1e-6:
                return self._reject(a, "budget")
            self.expedited[i] = True
            rem = self.T[i] - now
            self.T[i] = math.ceil(now) + max(P.TRANSIT, math.ceil(rem * (1 - self.exp_share)))
            self._reschedule(i)
            self._charge("expedite", cost)
            # confirmation of the expedited date, computed from the last reported date (not the true state)
            conf = math.ceil(now) + max(P.TRANSIT, math.ceil(max(0.0, self.rep_eta[i] - now) * (1 - self.exp_share)))
            self.rep_eta[i] = conf
            self.tdg.add_fact(("lot", l["id"]), Fact(now, now, "eta", float(conf), "expedite confirmation"))
        elif a.kind == "request":
            if self.requested_at[i] is not None:
                return self._reject(a, "already requested")
            if self.spent + self.appr_cost > self.budget + 1e-6:
                return self._reject(a, "budget")
            self.requested_at[i] = now
            self.approved_at[i] = now + self.ex["approval_lag"][i]
            self.env.process(self._approval(i))
            self._charge("request", self.appr_cost)
        elif a.kind == "order_alt":
            if self.approved_at[i] is None or self.approved_at[i] > now or self.alt_ordered_at[i] is not None:
                return self._reject(a, "no approval or already ordered")
            cost = P.ALT_PREMIUM * l["value"] + self.cancel_fee * l["value"]
            if self.spent + cost > self.budget + 1e-6:
                return self._reject(a, "budget")
            self.alt_ordered_at[i] = now
            self.source[i] = "alt"
            self.dispatched[i] = False
            self.T[i] = math.ceil(now) + int(round(l["alt_lead"] * self.alt_scale)) + int(self.ex["alt_late"][i])
            self._reschedule(i)
            self._charge("order_alt", cost)
            # the alternative supplier's own lateness is reported like a base delay
            if self.ex["alt_late"][i] > 0:
                self.rep_eta[i] = self.T[i] - self.ex["alt_late"][i]
                te = max(now, self.T[i] - self.ex["alt_late"][i] - self.ex["notice_lead"][i])
                self.env.process(self._notice(i, te, self.ex["reclag"][i][2], self.ex["notice_frac"][i],
                                              full_at=self.T[i] - self.ex["alt_late"][i]))
            else:
                self.rep_eta[i] = self.T[i]
        else:
            return self._reject(a, "unknown action")
        self.n_actions[a.kind] += 1
        if self.log is not None:
            self.log.append((now, a.kind, a.lot, "ok"))
        return True

    def _approval(self, i: int):
        yield self.env.timeout(max(0, self.approved_at[i] - self.env.now))
        self._record(i, self.env.now, 0, "approval", self.env.now)

    def _charge(self, kind: str, cost: float) -> None:
        self.spent += cost
        self.cost_by[kind] += cost

    def _reject(self, a: Action, why: str) -> bool:
        self.infeasible += 1
        if self.log is not None:
            self.log.append((self.env.now, a.kind, a.lot, "rejected: " + why))
        return False

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        env = self.env
        self.fin_evt = {u: env.event() for u in self.acts}
        for u in self.need_lots:
            self.ready_evt[u] = env.event()
        sinks = [u for u in self.acts if not any(u in self.acts[v]["preds"] for v in self.acts)]
        self.terminal = max(sinks, key=lambda u: self.acts[u]["r"])
        self.done_evt = simpy.AllOf(env, [self.fin_evt[u] for u in sinks])
        for i in range(len(self.lots)):
            self._reschedule(i)
        for u in self.acts:
            env.process(self._activity(u))
        env.process(self._base_notices())
        for d in self.ex["disruptions"]:
            if d.get("supplier", "x") is not None:
                env.process(self._disruption(d))
        env.process(self._weekly())
        env.run(until=self.done_evt)
        completion = max(self.finished.values())
        delay = max(0.0, completion - self.inst["due"])
        dc = self.o.get("delay_cost", P.DELAY_COST)
        return {"delay": delay, "completion": completion, "spent": self.spent, "total_cost": self.spent + dc * delay,
                "infeasible": self.infeasible, "n_expedite": self.n_actions["expedite"], "n_request": self.n_actions["request"],
                "n_order_alt": self.n_actions["order_alt"], "reviews": self.n_reviews,
                "policy_errors": getattr(self, "policy_errors", 0), "first_error": getattr(self, "first_error", ""),
                "cost_expedite": self.cost_by["expedite"], "cost_request": self.cost_by["request"],
                "cost_order_alt": self.cost_by["order_alt"]}
