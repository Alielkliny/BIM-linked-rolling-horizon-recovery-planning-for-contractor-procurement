"""Recovery policies: the proposed framework, its ablations, established alternatives, practice proxy and references.

Every policy receives the same snapshot (information recorded by the decision time), the same action library and the
same budget, and returns actions that the environment's checker validates.
"""
from __future__ import annotations

import math

import numpy as np

import params as P
from env import Action
from milp import Net, backward, build_and_solve, forward, need_map
from uncertainty import PlannerModel

THETA = 10  # practice rule: observed lateness (wd) that triggers an approval request for an alternative source


def _x(lv, t: float) -> tuple[float, bool]:
    """Observed lateness against the promised date, and whether any delay information exists."""
    base = lv.alt_eta if lv.alt_ordered and lv.alt_eta is not None else lv.P
    x = max(0.0, lv.eta - lv.P) if not lv.alt_ordered else 0.0
    noticed = lv.noticed and lv.eta > lv.P
    # an overdue lot without a receipt is at least as late as today
    if not lv.received and math.ceil(t) > lv.eta:
        x = max(x, math.ceil(t) - lv.P)
        noticed = True
    return x, noticed


class Policy:
    name = "policy"

    def __init__(self, inst: dict, seed: int = 0):
        self.inst = inst
        self.net = Net(inst)
        self.seed = seed

    def decide(self, snap):
        raise NotImplementedError


class Naive(Policy):
    name = "no_recovery"

    def decide(self, snap):
        return []


# ------------------------------------------------------------------------------- rule-based comparators
class _RuleBase(Policy):
    def _plan(self, snap):
        need = need_map(snap, True)
        arr = {}
        for lv in snap.lots:
            if lv.received:
                continue
            u = need[lv.id]
            arr[u] = max(arr.get(u, -1), math.ceil(lv.eta))
        S, F = forward(self.net, snap, arr)
        return need, arr, S, F

    @staticmethod
    def _need_date(net, snap, u, F):
        a = net.acts[u]
        return max([a["r"], math.ceil(snap.t)] + [F[p] for p in a["preds"]])


class ReactiveExpediting(_RuleBase):
    """Practice proxy: expedite a lot whose reported date delays its installation; request an alternative after a
    reported lateness of theta wd; order it once approved when its nominal arrival beats the reported date."""
    name = "reactive_expediting"
    theta = THETA

    def decide(self, snap):
        need, arr, S, F = self._plan(snap)
        acts, budget = [], snap.budget_remaining
        t = math.ceil(snap.t)
        for lv in snap.lots:
            if lv.received or lv.alt_ordered or need[lv.id] in snap.started:
                continue
            nd = self._need_date(self.net, snap, need[lv.id], F)
            gap = lv.eta - nd
            x, _ = _x(lv, snap.t)
            if gap > 0 and not lv.expedited and lv.eta - P.TRANSIT > t and budget >= P.EXPEDITE_COST * lv.value:
                acts.append(Action("expedite", lv.id)); budget -= P.EXPEDITE_COST * lv.value
            if x >= self.theta and lv.appr_state == "none" and budget >= P.APPROVAL_COST:
                acts.append(Action("request", lv.id)); budget -= P.APPROVAL_COST
            c_alt = (P.ALT_PREMIUM + P.CANCEL_FEE) * lv.value
            if lv.appr_state == "approved" and gap > 0 and t + lv.alt_lead < lv.eta and budget >= c_alt:
                acts.append(Action("order_alt", lv.id)); budget -= c_alt
        return acts


class BufferManagement(_RuleBase):
    """Critical-chain style buffer management: penetration of each lot's float buffer sets green, yellow and red
    zones; yellow requests an alternative, red expedites, a consumed buffer orders an approved alternative."""
    name = "buffer_management"
    zone_scale = 1.0

    def decide(self, snap):
        need, arr, S, F = self._plan(snap)
        LS = backward(self.net, F)
        acts, budget = [], snap.budget_remaining
        t = math.ceil(snap.t)
        for lv in snap.lots:
            if lv.received or lv.alt_ordered or need[lv.id] in snap.started:
                continue
            u = need[lv.id]
            x, _ = _x(lv, snap.t)
            if x <= 0:
                continue
            buf = max(1.0, (LS[u] - lv.P))   # time between the promised date and the latest start that keeps the due date
            pen = x / buf
            c_alt = (P.ALT_PREMIUM + P.CANCEL_FEE) * lv.value
            z = self.zone_scale
            if pen >= z / 3 and lv.appr_state == "none" and budget >= P.APPROVAL_COST:
                acts.append(Action("request", lv.id)); budget -= P.APPROVAL_COST
            if pen >= 2 * z / 3 and not lv.expedited and lv.eta - P.TRANSIT > t and budget >= P.EXPEDITE_COST * lv.value:
                acts.append(Action("expedite", lv.id)); budget -= P.EXPEDITE_COST * lv.value
            if pen >= z and lv.appr_state == "approved" and t + lv.alt_lead < lv.eta and budget >= c_alt:
                acts.append(Action("order_alt", lv.id)); budget -= c_alt
        return acts


# ------------------------------------------------------------------------------- optimisation-based policies
class RecoveryMILP(Policy):
    """Rolling-horizon recovery optimisation. mode: stochastic (scenarios), mean (expected values), quantile
    (robust counterpart at the 0.9 quantile) or oracle (true future values)."""
    name = "milp"

    def __init__(self, inst, seed=0, mode="stochastic", lam=1.0, n_scen=P.N_SCEN, approval_aware=True,
                 carry_state=True, bim_linked=True, static=False, model_pool="calib", episode=None, q=0.9,
                 name=None, time_limit=20.0, window=30, money_weight=1.0, reserve=0.0, alpha=P.CVAR_ALPHA,
                 backend="condensed", record=None, oracle_info="all", fixed_scenarios=False, gate_all=True,
                 sibling_rule=True, recourse=True, shrink=1.0, lone_rows=False, alt_gate=False, no_news=False,
                 sib_oracle=False, early_requests=False):
        super().__init__(inst, seed)
        self.sibling_rule, self.recourse, self.shrink = sibling_rule, recourse, shrink
        # lone_rows (A7): a lot alone in its supplier-month group draws lateness from shipment rows, as in the record
        # alt_gate (A8): substitute orders only when the substitute beats the current source at the 0.9 quantile
        self.lone_rows, self.alt_gate = lone_rows, alt_gate
        # no_news (A9): an unreported lot inside the notice window is conditioned on the absence of a recorded notice
        self.no_news = no_news
        # sib_oracle (diagnostic ceiling only): true current arrival of unreported lots with a late reported sibling
        self.sib_oracle = sib_oracle
        # early_requests (screen B1): approval requests are eligible for any open lot inside the planning window,
        # reported or not (expediting and substitute orders keep the reported-or-overdue gate)
        self.early_requests = early_requests
        self.oracle_info, self.fixed_scenarios, self.gate_all = oracle_info, fixed_scenarios, gate_all
        self.window, self.money_weight, self.reserve, self.alpha = window, money_weight, reserve, alpha
        self.backend, self.record = backend, record
        self.mode, self.lam, self.n_scen = mode, lam, n_scen
        self.approval_aware, self.carry_state, self.bim_linked = approval_aware, carry_state, bim_linked
        self.static, self.q, self.episode, self.time_limit = static, q, episode, time_limit
        self.model = PlannerModel(model_pool)
        self.done_static = False
        self.solves, self.fail = 0, 0
        self.gaps = []
        if name:
            self.name = name

    def _samples(self, snap, S, rng):
        """Joint samples of every open lot's arrival, approval residual and alternative lateness (S draws)."""
        t = math.ceil(snap.t)
        lots = [lv for lv in snap.lots if not lv.received]
        sib_late = {}
        info = {}
        for lv in lots:
            x, noticed = _x(lv, snap.t)
            info[lv.id] = (x, noticed)
            if noticed and x > 0 and not lv.alt_ordered:
                sib_late[(lv.package, lv.P // 21)] = True
        unnoticed_by_sup = {}
        for lv in lots:
            x, noticed = info[lv.id]
            if not noticed and not lv.alt_ordered:
                unnoticed_by_sup.setdefault((lv.package, lv.P // 21), []).append(lv.id)
        Lmat = {}
        gsize = {}
        for lv in snap.lots:
            gsize[(lv.package, lv.P // 21)] = gsize.get((lv.package, lv.P // 21), 0) + 1
        hz = {lv.id: lv.P - snap.t for lv in lots}
        for k, ids in unnoticed_by_sup.items():
            lone = self.lone_rows and gsize.get(k, 0) == 1
            n_draw = 4 * S if self.no_news else S
            if getattr(self, "no_cluster", False):   # ablation: independent lateness per lot, no supplier-month conditioning
                draws = rng.choice(self.model.late, size=(n_draw, len(ids))).astype(float)
            elif lone:
                draws = rng.choice(self.model.late, size=(n_draw, 1)).astype(float)
            else:
                draws = np.array([self.model.sample_supplier_block(rng, sib_late.get(k, False), len(ids)) for _ in range(n_draw)])
            if self.no_news:
                # P(notice already recorded | late) for each lot, from the notice window and the record lags;
                # candidate draws are resampled in proportion to the likelihood of the observed silence
                v = np.array([self.model.p_notice_seen(hz[i]) for i in ids])
                w = np.prod(np.where(draws > 0, 1.0 - v[None, :], 1.0), axis=1)
                w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
                draws = draws[rng.choice(len(draws), size=S, p=w)]
            for j, lid in enumerate(ids):
                Lmat[lid] = draws[:, j].astype(float)
        out = {}
        for lv in lots:
            state_exp, state_appr, state_alt, req_at = lv.expedited, lv.appr_state, lv.alt_ordered, lv.requested_at
            if not self.carry_state:          # ablation: the planner forgets commitments and pending approvals
                state_exp, state_appr, state_alt, req_at = False, "none", False, None
            x, noticed = info[lv.id]
            fac = (1 - P.EXPEDITE_SHARE) if (state_exp and not lv.alt_ordered) else 1.0
            if lv.alt_ordered:
                orig = np.maximum(t + 1.0, lv.eta + self.model.sample_alt_late(rng, S).astype(float))
            elif not noticed:
                orig = np.maximum(t + 1.0, lv.eta + Lmat[lv.id] * fac * self.shrink)
            elif math.ceil(snap.t) <= lv.eta:
                # a reported date is taken as the supplier's current commitment
                orig = np.full(S, max(t + 1.0, float(lv.eta)))
            else:
                # overdue without a recorded receipt: final lateness L >= elapsed lateness x from the record
                L = self.model.sample_final_late(rng, x, True, S).astype(float)
                orig = np.maximum(t + 1.0, lv.P + L)
            if state_appr == "pending":
                el = max(0.0, snap.t - (req_at or snap.t))
                a_res = self.model.sample_approval_residual(rng, el, S)
            elif state_appr == "approved":
                a_res = np.zeros(S)
            else:
                a_res = self.model.sample_approval_residual(rng, 0.0, S)
            la = self.model.sample_alt_late(rng, S).astype(float)
            if not noticed:   # revelation time of the remaining delay, used by the wait-and-see expedite
                rev = np.maximum(t + 1.0, np.minimum(lv.P, orig) - P.NOTICE_MAX / 2)
            else:
                rev = np.full(S, t + float(P.REVIEW_EVERY))
            out[lv.id] = dict(lv=lv, orig=orig, a_res=a_res, la=la, rev=rev, x=x, noticed=noticed,
                              sib=bool(sib_late.get((lv.package, lv.P // 21), False)), state=(state_exp, state_appr, state_alt))
            if self.sib_oracle and out[lv.id]["sib"] and not noticed and not lv.alt_ordered and self.episode is not None:
                tru = max(t + 1.0, float(self.episode.T[lv.idx]))
                out[lv.id].update(orig=np.full(S, tru), noticed=True, rev=np.full(S, t + float(P.REVIEW_EVERY)))
        return out

    def _oracle_samples(self, snap):
        t = math.ceil(snap.t)
        ep = self.episode
        out = {}
        for lv in snap.lots:
            if lv.received:
                continue
            i = lv.idx
            fut = 0.0
            for d in ep.ex["disruptions"]:
                if d["tau"] <= snap.t:
                    continue
                if d["kind"] == "supplier" and d.get("supplier") == lv.package and ep.source[i] == "orig" and ep.T[i] - P.TRANSIT > d["tau"]:
                    fut += d["delta"]
                if d["kind"] == "transport" and d["tau"] < ep.T[i] <= d["tau"] + d["window"]:
                    fut += d["delta"]
            a = (ep.approved_at[i] - t) if ep.approved_at[i] is not None else float(ep.ex["approval_lag"][i])
            out[lv.id] = dict(lv=lv, orig=np.array([max(t + 1.0, float(ep.T[i]) + fut)]), a_res=np.array([max(0.0, a)]),
                              la=np.array([float(ep.ex["alt_late"][i])]), rev=np.array([t + 1.0]), x=0.0, noticed=True, sib=True,
                              state=(lv.expedited, lv.appr_state, lv.alt_ordered))
        return out

    def _partial_oracle(self, snap, rng):
        """Expected-value planner that knows one class of future information exactly (headroom decomposition)."""
        t = math.ceil(snap.t)
        ep = self.episode
        big = self._samples(snap, 200, rng)
        red = lambda v: np.array([float(np.mean(v))])
        full = self._oracle_samples(snap)
        out = {}
        for k, d in big.items():
            d = dict(d, orig=red(d["orig"]), a_res=red(d["a_res"]), la=red(d["la"]), rev=red(d["rev"]))
            i = d["lv"].idx
            if self.oracle_info == "base":
                d["orig"] = np.array([max(t + 1.0, float(ep.T[i]))]); d["noticed"] = True
            elif self.oracle_info == "disr":
                fut = full[k]["orig"][0] - max(t + 1.0, float(ep.T[i]))
                if fut > 0:
                    d["orig"] = d["orig"] + fut; d["noticed"] = True
            elif self.oracle_info == "appr":
                d["a_res"], d["la"] = full[k]["a_res"], full[k]["la"]
            out[k] = d
        return out

    def _established(self, snap):
        """Inputs of the established rivals: reported dates only ('reported'), or a budgeted interval set ('budgeted')."""
        t = math.ceil(snap.t)
        acts_r = {a["uid"]: a["r"] for a in snap.activities}
        lots = [lv for lv in snap.lots if not lv.received]
        info, sib = {}, {}
        for lv in lots:
            x, noticed = _x(lv, snap.t)
            info[lv.id] = (x, noticed)
            if noticed and x > 0 and not lv.alt_ordered:
                sib[(lv.package, lv.P // 21)] = True
        if self.mode == "reported":
            dhat, a_nom = 0.0, float(P.NOMINAL_APPROVAL)
            risk = []
        else:
            pos = self.model.late_pos
            dq = float(getattr(self, "dev_q", 0.9))  # bound quantile of the budgeted set (0.9 unless tuned)
            dhat = float(np.quantile(pos, dq))
            a_nom = float(np.quantile(self.model.appr, dq))
            risk = [lv.id for lv in lots if not info[lv.id][1] and not lv.alt_ordered and acts_r.get(lv.need_uid, 0) - t <= self.window]
        S = 1 + len(risk)
        out = {}
        for lv in lots:
            x, noticed = info[lv.id]
            base = max(t + 1.0, float(lv.eta)) if not (noticed and math.ceil(snap.t) > lv.eta) else t + 1.0
            orig = np.full(S, base)
            if lv.id in risk:
                orig[1 + risk.index(lv.id)] = base + dhat
            el = max(0.0, snap.t - (lv.requested_at or snap.t))
            a_res = np.full(S, 0.0 if lv.appr_state == "approved" else max(1.0, a_nom - el))
            out[lv.id] = dict(lv=lv, orig=orig, a_res=a_res, la=np.zeros(S), rev=np.full(S, t + 1.0), x=x, noticed=noticed,
                              sib=bool(sib.get((lv.package, lv.P // 21), False)),
                              state=(lv.expedited, lv.appr_state, lv.alt_ordered))
        return out

    def _scen(self, snap):
        t = math.ceil(snap.t)
        if self.mode in ("reported", "budgeted"):
            smp = self._established(snap)
            return self._finish_scen(snap, smp, t)
        if self.fixed_scenarios:
            rng = np.random.default_rng(self.seed % (2 ** 32))
        else:
            rng = np.random.default_rng((self.seed * 1_000_003 + int(snap.t * 2)) % (2 ** 32))
        if self.mode == "oracle" and self.oracle_info != "all":
            smp = self._partial_oracle(snap, rng)
        elif self.mode == "oracle":
            smp = self._oracle_samples(snap)
        elif self.mode == "stochastic":
            smp = self._samples(snap, self.n_scen, rng)
        else:  # expected-value or quantile reduction of the same joint distribution
            big = self._samples(snap, 200, rng)
            if self.mode == "mean":
                red = lambda v: np.array([float(np.mean(v))])
            else:
                red = lambda v: np.array([float(np.quantile(v, self.q))])
            smp = {k: dict(d, orig=red(d["orig"]), a_res=red(d["a_res"]), la=red(d["la"]), rev=red(d["rev"])) for k, d in big.items()}
        return self._finish_scen(snap, smp, t)

    def _finish_scen(self, snap, smp, t):
        acts_r = {a["uid"]: a["r"] for a in snap.activities}
        out = {}
        for lid, d in smp.items():
            lv = d["lv"]
            state_exp, state_appr, state_alt = d["state"]
            orig, a_res, la, rev = d["orig"], d["a_res"], d["la"], d["rev"]
            if not self.approval_aware:    # ablation: alternatives treated as available once requested
                a_res = np.zeros_like(a_res)
            in_window = d["noticed"] or (acts_r.get(lv.need_uid, 0) - t <= self.window) or state_appr == "pending"
            # action library eligibility: expediting answers a reported or elapsed delay (shared by every planner)
            can_exp = (d["noticed"] or (d["sib"] and self.sibling_rule)) and (not state_exp) and (not state_alt) and lv.eta - P.TRANSIT > t
            exp = t + np.maximum(P.TRANSIT, np.ceil((orig - t) * (1 - P.EXPEDITE_SHARE)))
            base_ok = (not state_exp) and (not state_alt)
            # non-anticipative recourse: a later expedite is modelled only for lots whose delay is not yet reported,
            # at the time the notice reveals it; a reported delay is answered now or not at all
            rec_ok = np.logical_and(np.logical_and(rev < orig - P.TRANSIT, orig > lv.P), base_ok and not d["noticed"] and self.recourse)
            exp_rec = rev + np.maximum(P.TRANSIT, np.ceil((orig - rev) * (1 - P.EXPEDITE_SHARE)))
            appr_ok = state_appr == "approved" or (not self.approval_aware and state_appr in ("none", "pending"))
            sib = d["sib"] and self.sibling_rule
            gate = (d["noticed"] or sib) if self.gate_all else True
            can_order = in_window and gate and (not state_alt) and appr_ok
            can_req = in_window and (gate or self.early_requests) and (not state_alt) and state_appr == "none" and self.approval_aware
            pending = (not state_alt) and state_appr == "pending" and self.approval_aware and self.recourse
            req_later_ok = (not state_alt) and state_appr == "none" and self.approval_aware and self.recourse
            altrec2 = rev + a_res + lv.alt_lead + la
            extra = {}
            if self.alt_gate and self.mode not in ("reported", "budgeted", "oracle"):
                qv = float(np.quantile(self.model.late_pos, 0.9))
                cur = np.minimum(orig, exp) if can_exp else np.asarray(orig, float)
                cur2 = np.where(rec_ok, np.minimum(orig, exp_rec), orig)
                can_order = can_order and bool(np.all(t + lv.alt_lead + qv < cur))
                extra = {"r_ok": np.asarray(t + a_res + lv.alt_lead + qv < cur, bool),
                         "q2_ok": np.asarray(rev + a_res + lv.alt_lead + qv < cur2, bool)}
            out[lid] = {"altrec2": np.asarray(altrec2, float), "req_later_ok": bool(req_later_ok), **extra,
                        "orig": np.asarray(orig, float), "exp": np.asarray(exp, float), "exp_rec": np.asarray(exp_rec, float),
                        "rec_ok": np.asarray(rec_ok, bool), "altnow": np.asarray(t + lv.alt_lead + la, float),
                        "altrec": np.asarray(t + a_res + lv.alt_lead + la, float), "can_exp": bool(can_exp),
                        "can_req": bool(can_req), "can_order": bool(can_order), "pending": bool(pending),
                        "alt_ordered": bool(state_alt), "need": lv.need_uid, "c_exp": P.EXPEDITE_COST * lv.value,
                        "c_req": P.APPROVAL_COST, "c_alt": (P.ALT_PREMIUM + P.CANCEL_FEE) * lv.value}
        return out

    def decide(self, snap):
        if self.static and self.done_static:
            return []
        self.done_static = True
        scen = self._scen(snap)
        if not scen:
            return []
        # skip the solve when no lot can delay its activity in any scenario
        need = need_map(snap, self.bim_linked)
        acts_r = {a["uid"]: a["r"] for a in snap.activities}
        risky = any((sc["orig"].max() > acts_r.get(need.get(lid, sc["need"]), 0)) or sc["pending"] for lid, sc in scen.items())
        if not risky:
            return []
        budget = snap.budget_remaining
        if not self.carry_state:
            budget = self.inst["budget"]
        snap2 = snap
        if budget != snap.budget_remaining:
            from dataclasses import replace
            snap2 = replace(snap, budget_remaining=budget)
        if self.record is not None:
            self.record.append((snap2, scen, need))
        lam, alpha = self.lam, self.alpha
        if self.mode == "budgeted":
            S_n = next(iter(scen.values()))["orig"].shape[0]
            lam, alpha = (10.0, 1.0 - 1.0 / S_n) if S_n > 1 else (0.0, self.alpha)
        res, info = build_and_solve(self.net, snap2, scen, lam, alpha=alpha, need=need,
                                    money_weight=self.money_weight, reserve=self.reserve, time_limit=self.time_limit,
                                    backend=self.backend)
        self.solves += 1
        if not res and info.get("status", "").find("Optimal") < 0 and "gap" in info and info.get("n_vars") is None:
            self.fail += 1
        self.gaps.append(info.get("gap", 0.0))
        out = []
        for kind, lid in res:
            if kind == "order_alt" and not self.approval_aware:
                lv = next(l for l in snap.lots if l.id == lid)
                if lv.appr_state == "none":
                    out.append(Action("order_alt", lid))   # rejected by the checker: no approval yet
                    out.append(Action("request", lid))
                    continue
            out.append(Action(kind, lid))
        return out


def _flag(p, **kw):
    for k, v in kw.items():
        setattr(p, k, v)
    return p


ROSTER = {
    # name: (factory, role, env pooling)
    "no_recovery": (lambda inst, seed, ep: Naive(inst, seed), "naive", True),
    "reactive_expediting": (lambda inst, seed, ep: ReactiveExpediting(inst, seed), "practice", True),
    "buffer_management": (lambda inst, seed, ep: BufferManagement(inst, seed), "strong", True),
    "reported_rescheduling": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="reported", approval_aware=False, recourse=False,
                                                                 sibling_rule=False, name="reported_rescheduling"), "strong", True),
    "budgeted_robust": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="budgeted", approval_aware=False, recourse=False,
                                                           sibling_rule=False, name="budgeted_robust"), "strong", True),
    "proposed": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="stochastic", lam=0.0, fixed_scenarios=True, name="proposed"), "proposed", True),
    "variant_cvar": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="stochastic", lam=1.0, fixed_scenarios=True, name="variant_cvar"), "variant", True),
    "abl_expected_value": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="mean", name="abl_expected_value"), "ablation", True),
    "abl_no_approval": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="stochastic", lam=0.0, approval_aware=False, fixed_scenarios=True, name="abl_no_approval"), "ablation", True),
    "abl_no_state": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="stochastic", lam=0.0, carry_state=False, fixed_scenarios=True, name="abl_no_state"), "ablation", True),
    "abl_no_bim": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="stochastic", lam=0.0, bim_linked=False, fixed_scenarios=True, name="abl_no_bim"), "ablation", False),
    "abl_no_bim_planner": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="stochastic", lam=0.0, bim_linked=False, fixed_scenarios=True, name="abl_no_bim_planner"), "ablation", True),
    "abl_no_recourse": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="stochastic", lam=0.0, recourse=False, fixed_scenarios=True, name="abl_no_recourse"), "ablation", True),
    "abl_no_cluster": (lambda inst, seed, ep: _flag(RecoveryMILP(inst, seed, mode="stochastic", lam=0.0, fixed_scenarios=True, name="abl_no_cluster"), no_cluster=True), "ablation", True),
    "abl_static": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="stochastic", lam=0.0, static=True, fixed_scenarios=True, name="abl_static"), "ablation", True),
    "oracle": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="oracle", episode=ep, name="oracle"), "diagnostic", True),
    "oracle_base": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="oracle", episode=ep, oracle_info="base", name="oracle_base"), "diagnostic", True),
    "oracle_disr": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="oracle", episode=ep, oracle_info="disr", name="oracle_disr"), "diagnostic", True),
    "oracle_appr": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="oracle", episode=ep, oracle_info="appr", name="oracle_appr"), "diagnostic", True),
    # retained for the screening record (earlier designs, probe data only)
    "deterministic_rolling": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="mean", name="deterministic_rolling"), "screening", True),
    "robust_rolling": (lambda inst, seed, ep: RecoveryMILP(inst, seed, mode="quantile", name="robust_rolling"), "screening", True),
}
