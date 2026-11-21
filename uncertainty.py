"""Empirical timing distributions from the SCMS pools: the planner's calibrated model and the environment's draws."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from common import PROC


N_FOLDS = 3


def _fold(key: str) -> int:
    import hashlib
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % N_FOLDS


def select(name: str):
    """Shipments and quotations of a named pool. Cross-fit pools cf{k}_train / cf{k}_test split the 2007-2011
    development shipments by vendor-month block into N_FOLDS folds (quotations by quotation number)."""
    s = pd.read_csv(PROC / "scms_shipments.csv")
    q = pd.read_csv(PROC / "scms_quote_lags.csv")
    if name.startswith("cf"):
        k = int(name[2])
        part = name.split("_")[1]
        s = s[s.pool.isin(["calib", "check"])].copy()
        s["ym"] = pd.to_datetime(s.sched_date).dt.to_period("M").astype(str)
        s["fold"] = [(_fold(f"{v}|{m}")) for v, m in zip(s.vendor.astype(str), s.ym)]
        q = q[q.pool.isin(["calib", "check"])].copy()
        q["fold"] = [_fold("q|" + str(x)) for x in q["pq #"]]
        if part == "test":
            return s[s.fold == k], q[q.fold == k]
        return s[s.fold != k], q[q.fold != k]
    names = {"calib": ["calib"], "check": ["check"], "reserve1": ["reserve1"], "reserve2": ["reserve2"],
             "dev_all": ["calib", "check"]}[name]
    return s[s.pool.isin(names)], q[q.pool.isin(names)]


@lru_cache(maxsize=None)
def pool(name: str) -> dict:
    s, q = select(name)
    late = np.clip(s.lateness_wd.to_numpy(dtype=float), 0, None)
    pos = np.sort(late[late > 0])
    return {
        "late": np.sort(late),                    # final lateness per shipment (wd, zero when on time or early)
        "late_pos": pos,                          # lateness of late shipments only
        "reclag": np.clip(s.reclag_wd.fillna(0).to_numpy(dtype=float), 0, None),
        "approval": np.clip(q.quote_to_po_wd.to_numpy(dtype=float), 1, None),
        "late_share": float((late > 0).mean()),
        "n_shipments": int(len(s)), "n_quotes": int(len(q)),
    }


@lru_cache(maxsize=None)
def vendor_blocks(name: str) -> tuple:
    """Lateness of shipments grouped by vendor and scheduled month (blocks with at least two shipments)."""
    s, _ = select(name)
    s = s.copy()
    s["ym"] = pd.to_datetime(s.sched_date).dt.to_period("M")
    s["lw"] = np.clip(s.lateness_wd.to_numpy(float), 0, None)
    blocks = [g.lw.to_numpy() for _, g in s.groupby(["vendor", "ym"]) if len(g) >= 2]
    late_blocks = [b for b in blocks if (b > 0).sum() >= 1]
    return blocks, late_blocks


class PlannerModel:
    """Calibrated model the optimisation-based planners share (fitted on the calibration pool only)."""

    def __init__(self, pool_name: str = "calib"):
        p = pool(pool_name)
        self.blocks, self.late_blocks = vendor_blocks(pool_name)
        self.late = p["late"]
        self.late_pos = p["late_pos"]
        self.appr = np.sort(p["approval"])
        self.reclag = np.asarray(p["reclag"], float)
        self.tail_excess = float(np.mean(self.late_pos[-max(5, len(self.late_pos) // 20):] - self.late_pos[-max(5, len(self.late_pos) // 20)]))
        self.tail_excess = max(self.tail_excess, 5.0)

    # final lateness given that at least x wd of lateness has been observed (x = 0 and no notice: unconditional)
    def sample_final_late(self, rng, x: float, noticed: bool, n: int) -> np.ndarray:
        if not noticed and x <= 0:
            return rng.choice(self.late, size=n)
        base = self.late_pos[self.late_pos >= x]
        if len(base) >= 5:
            return rng.choice(base, size=n)
        return x + rng.exponential(self.tail_excess, size=n)

    def mean_final_late(self, x: float, noticed: bool) -> float:
        if not noticed and x <= 0:
            return float(self.late.mean())
        base = self.late_pos[self.late_pos >= x]
        return float(base.mean()) if len(base) >= 5 else x + self.tail_excess

    def quantile_final_late(self, x: float, noticed: bool, q: float) -> float:
        if not noticed and x <= 0:
            return float(np.quantile(self.late, q))
        base = self.late_pos[self.late_pos >= x]
        return float(np.quantile(base, q)) if len(base) >= 5 else x + self.tail_excess * -np.log(1 - q)

    def sample_approval_residual(self, rng, elapsed: float, n: int) -> np.ndarray:
        base = self.appr[self.appr > elapsed]
        if len(base) < 5:
            return np.full(n, 5.0)
        return np.maximum(1.0, rng.choice(base, size=n) - elapsed)

    def mean_approval_residual(self, elapsed: float) -> float:
        base = self.appr[self.appr > elapsed]
        return float(base.mean() - elapsed) if len(base) >= 5 else 5.0

    def quantile_approval_residual(self, elapsed: float, q: float) -> float:
        base = self.appr[self.appr > elapsed]
        return float(np.quantile(base, q) - elapsed) if len(base) >= 5 else 5.0

    def p_notice_seen(self, h: float) -> float:
        """Probability that a late lot due in h working days already has a recorded notice: first notices arrive
        0 to NOTICE_MAX wd before the promised date (uniform) and are recorded after the pool's record lag."""
        import params as P
        if h <= 0:
            return 1.0
        lead = np.arange(P.NOTICE_MAX + 1)
        lag = self.reclag
        return float(np.mean((lead[None, :] - lag[:, None]) >= h))

    def sample_alt_late(self, rng, n: int) -> np.ndarray:
        return rng.choice(self.late, size=n)

    def sample_supplier_block(self, rng, sibling_late: bool, m: int) -> np.ndarray:
        """Joint lateness of m not-yet-reported lots of one supplier: a resampled vendor-month block, conditioned on
        a late sibling when one lot of the supplier is already reported late."""
        if sibling_late:
            b = self.late_blocks[int(rng.integers(0, len(self.late_blocks)))].copy()
            j = int(np.flatnonzero(b > 0)[0])
            b = np.delete(b, j)
        else:
            b = self.blocks[int(rng.integers(0, len(self.blocks)))]
        return rng.choice(b, size=m)
