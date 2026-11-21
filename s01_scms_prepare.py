"""Stage 01: prepare the SCMS delivery-history record as the procurement-timing calibration source.

Reads the Wayback copy of the USAID data.usaid.gov export (10,324 order lines), parses the five date
fields, derives shipment-level lateness, record-entry lag, quotation-to-order lag and price premia, and
assigns shipments (ASN/DN numbers) to the calibration, check and two sealed reserve pools.
All durations are converted from calendar days to working days with the factor 5/7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import PROC, RESULTS, SCMS_CSV, SCMS_URL, sha256, stage_manifest, write_json

WD = 5.0 / 7.0
SEED = 20261108


def _dt(s: pd.Series) -> pd.Series:
    a = pd.to_datetime(s, format="%d-%b-%y", errors="coerce")
    b = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    return a.fillna(b)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_lines() -> pd.DataFrame:
    d = pd.read_csv(SCMS_CSV, encoding="utf-8")
    d.columns = [c.strip().lower() for c in d.columns]
    for c in ["pq first sent to client date", "po sent to vendor date", "scheduled delivery date",
              "delivered to client date", "delivery recorded date"]:
        d[c + "_dt"] = _dt(d[c])
    d["lateness_cd"] = (d["delivered to client date_dt"] - d["scheduled delivery date_dt"]).dt.days
    d["reclag_cd"] = (d["delivery recorded date_dt"] - d["delivered to client date_dt"]).dt.days
    d["promised_lt_cd"] = (d["scheduled delivery date_dt"] - d["po sent to vendor date_dt"]).dt.days
    d["quote_to_po_cd"] = (d["po sent to vendor date_dt"] - d["pq first sent to client date_dt"]).dt.days
    d["sched_year"] = d["scheduled delivery date_dt"].dt.year
    d["unit_price"] = _num(d["unit price"])
    d["line_value"] = _num(d["line item value"])
    d["freight_usd"] = _num(d["freight cost (usd)"])
    d["weight_kg"] = _num(d["weight (kilograms)"])
    return d


def shipments(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("asn/dn #")
    s = pd.DataFrame({
        "n_lines": g.size(),
        "lateness_cd": g["lateness_cd"].median(),
        "lateness_mixed": g["lateness_cd"].agg(lambda x: int(x.nunique() > 1)),
        "reclag_cd": g["reclag_cd"].median(),
        "mode": g["shipment mode"].agg(lambda x: x.mode().iat[0] if x.notna().any() else "Unknown"),
        "vendor": g["vendor"].first(),
        "fulfill": g["fulfill via"].first(),
        "sched_year": g["sched_year"].min(),
        "sched_date": g["scheduled delivery date_dt"].min(),
        "promised_lt_cd": g["promised_lt_cd"].median(),
        "value_usd": g["line_value"].sum(),
    }).reset_index().rename(columns={"asn/dn #": "asn"})
    s["mode"] = s["mode"].fillna("Unknown")
    for c in ["lateness", "reclag", "promised_lt"]:
        s[c + "_wd"] = np.round(s[c + "_cd"] * WD)
    s["late"] = (s["lateness_cd"] > 0).astype(int)
    return s


def assign_pools(s: pd.DataFrame) -> pd.DataFrame:
    """Calibration (75%) and check (25%) from 2007-2011; reserves R1 and R2 halves of 2012-2015.

    Stratified by scheduled year, shipment mode and the late flag so that each pool keeps the mix of the
    period it comes from. 2006 (65 lines, none late) is excluded.
    """
    rng = np.random.default_rng(SEED)
    s = s.copy()
    s["pool"] = "excluded_2006"
    for (yr, mode, late), idx in s.groupby(["sched_year", "mode", "late"]).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        if yr <= 2006:
            continue
        if yr <= 2011:
            k = int(round(0.75 * len(idx)))
            s.loc[idx[:k], "pool"] = "calib"
            s.loc[idx[k:], "pool"] = "check"
        else:
            k = int(round(0.5 * len(idx)))
            s.loc[idx[:k], "pool"] = "reserve1"
            s.loc[idx[k:], "pool"] = "reserve2"
    return s


def quote_lags(d: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Quotation-to-purchase-order lag per quotation (PQ #), the approval-delay calibration source."""
    q = d.dropna(subset=["quote_to_po_cd"]).copy()
    q = q[q["quote_to_po_cd"] >= 0]
    pool_of = s.set_index("asn")["pool"]
    q["pool"] = q["asn/dn #"].map(pool_of)
    g = q.groupby("pq #").agg(quote_to_po_cd=("quote_to_po_cd", "min"), pool=("pool", "first"),
                               year=("sched_year", "min")).reset_index()
    g["quote_to_po_wd"] = np.round(g["quote_to_po_cd"] * WD)
    return g


def price_premium(d: pd.DataFrame) -> dict:
    """Unit-price ratio of a vendor to the cheapest vendor of the same item and year (non-preferred source)."""
    x = d.dropna(subset=["unit_price"]).copy()
    x = x[x["unit_price"] > 0]
    m = x.groupby(["item description", "sched_year", "vendor"])["unit_price"].median().reset_index()
    best = m.groupby(["item description", "sched_year"])["unit_price"].transform("min")
    nv = m.groupby(["item description", "sched_year"])["vendor"].transform("nunique")
    r = (m["unit_price"] / best)[(nv >= 2) & (m["unit_price"] > best)]
    return {"n_vendor_item_years": int(len(r)), "ratio_quantiles": {str(q): float(np.quantile(r, q)) for q in
            (0.1, 0.25, 0.5, 0.75, 0.9)}, "ratio_mean": float(r.mean())}


def mode_profile(d: pd.DataFrame) -> dict:
    out = {}
    for mode, g in d.groupby("shipment mode"):
        fr = g[(g["freight_usd"] > 0) & (g["line_value"] > 0)]
        out[mode] = {
            "lines": int(len(g)),
            "promised_lt_wd_median": float(np.nanmedian(g["promised_lt_cd"]) * WD) if g["promised_lt_cd"].notna().any() else None,
            "late_share": float((g["lateness_cd"] > 0).mean()),
            "freight_to_value_median": float(np.median(fr["freight_usd"] / fr["line_value"])) if len(fr) else None,
            "freight_lines": int(len(fr)),
        }
    return out


def main() -> None:
    d = load_lines()
    s = assign_pools(shipments(d))
    q = quote_lags(d, s)
    s.to_csv(PROC / "scms_shipments.csv", index=False)
    q.to_csv(PROC / "scms_quote_lags.csv", index=False)
    prof = {
        "source": {"file": "data/raw/scms_wayback/scms.csv", "url": SCMS_URL, "sha256": sha256(SCMS_CSV),
                   "accessed": "2026-11-08",
                   "note": "GitHub mirror suheb1231/SCMS-Delivery-History-Dataset agrees on 10,324 rows, ids, delivery and"
                           " record dates, and differs in the reformatted quotation and purchase-order date strings;"
                           " the archived USAID export is the source of record."},
        "lines": int(len(d)), "columns": int(d.shape[1] - 14),
        "shipments": int(len(s)),
        "shipments_multi_line": int((s.n_lines > 1).sum()),
        "shipments_mixed_lateness_share": float(s.loc[s.n_lines > 1, "lateness_mixed"].mean()),
        "late_share_lines": float((d["lateness_cd"] > 0).mean()),
        "late_lines": int((d["lateness_cd"] > 0).sum()),
        "late_days_quantiles_cd": {str(k): float(np.quantile(d.loc[d.lateness_cd > 0, "lateness_cd"], k)) for k in
                                    (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
        "reclag_quantiles_cd": {str(k): float(np.quantile(d["reclag_cd"].dropna(), k)) for k in
                                (0.5, 0.75, 0.9, 0.95, 0.99)},
        "reclag_positive_share": float((d["reclag_cd"] > 0).mean()),
        "quote_to_po_quotations": int(len(q)),
        "quote_to_po_quantiles_cd": {str(k): float(np.quantile(q["quote_to_po_cd"], k)) for k in
                                     (0.1, 0.25, 0.5, 0.75, 0.9)},
        "late_share_by_year": {int(k): float(v) for k, v in d.groupby("sched_year")["lateness_cd"].apply(lambda x: (x > 0).mean()).items()},
        "pools": {p: {"shipments": int((s.pool == p).sum()), "late_shipments": int(s.loc[s.pool == p, "late"].sum()),
                      "late_share": float(s.loc[s.pool == p, "late"].mean()),
                      "quotations": int((q.pool == p).sum())} for p in sorted(s.pool.unique())},
        "mode_profile": mode_profile(d),
        "price_premium": price_premium(d),
    }
    write_json(RESULTS / "s01_scms_profile.json", prof)
    stage_manifest("s01_scms", [SCMS_CSV], [PROC / "scms_shipments.csv", PROC / "scms_quote_lags.csv",
                                            RESULTS / "s01_scms_profile.json"], [__file__, "src/common.py"])
    print({k: prof[k] for k in ("lines", "shipments", "late_share_lines", "shipments_mixed_lateness_share")})
    print(prof["pools"])
    print(prof["price_premium"])
    print(prof["mode_profile"])


if __name__ == "__main__":
    main()
