"""Model parameters with provenance tags (REAL, CITED, ASSUMPTION). Every ASSUMPTION has a sensitivity range.

Units: time in working days (wd), money in euro. Values tagged REAL are measured in this study's data files
(results/s01_scms_profile.json for the procurement record); ASSUMPTION values are varied in the sensitivity study.
"""
from __future__ import annotations

# Nominal supplier lead time from order to promised delivery and lead time of an approved alternative source (wd).
LEAD_TIME = {  # ASSUMPTION, sensitivity factor 0.5 to 1.5 on the alternative-source lead time
    "hollowcore_floor": 30, "filigree_floor": 25, "cs_elements": 25, "prefab_stairs": 30, "steel": 20,
    "window_frames": 40, "facade_brick": 15, "roof_elements": 35, "concrete_bands": 30, "roof_windows": 20,
    "railings": 30, "lift": 60,
}
# Delivery offset before the need date in the plan (wd): prefabricated elements arrive on the installation day.
JIT_OFFSET = {  # ASSUMPTION, sensitivity: offsets doubled
    "hollowcore_floor": 0, "filigree_floor": 0, "cs_elements": 2, "prefab_stairs": 0, "steel": 2,
    "window_frames": 5, "facade_brick": 5, "roof_elements": 0, "concrete_bands": 2, "roof_windows": 5,
    "railings": 5, "lift": 10,
}
TRANSIT = 2  # wd between dispatch and delivery; expediting is possible before dispatch (ASSUMPTION)
# Package value (euro) split over lots by volume share, or by element count where volume is absent (ASSUMPTION, x0.5 to x2).
PACKAGE_VALUE = {
    "hollowcore_floor": 60000, "filigree_floor": 120000, "cs_elements": 150000, "prefab_stairs": 30000, "steel": 40000,
    "window_frames": 250000, "facade_brick": 150000, "roof_elements": 200000, "concrete_bands": 30000,
    "roof_windows": 20000, "railings": 60000, "lift": 80000,
}
EXPEDITE_SHARE = 0.4      # share of remaining time to arrival removed by expediting (ASSUMPTION, 0.2 to 0.6)
EXPEDITE_COST = 0.067     # share of lot value: air minus truck freight-to-value medians, 0.126 - 0.059 (REAL, SCMS)
ALT_PREMIUM = 0.20        # unit-price premium of a non-preferred source: median ratio 1.20 (REAL, SCMS)
CANCEL_FEE = 0.10         # share of lot value paid when an order is cancelled (ASSUMPTION, 0.05 to 0.20)
APPROVAL_COST = 500.0     # submittal preparation for an alternative source, euro (ASSUMPTION, 250 to 1000)
DELAY_COST = 2500.0       # time-related cost of one working day of completion delay, euro (ASSUMPTION, 1000 to 5000)
BUDGET_SHARE = 0.03       # recovery budget as a share of procured package value (ASSUMPTION, 0.015 to 0.06)
NOTICE_MAX = 10           # first delay notice arrives 0 to 10 wd before the promised date (ASSUMPTION)
NOTICE_FRACTION = (1.0, 1.0)  # share of the final lateness a first notice reports (ASSUMPTION; sensitivity 0.3 to 1.0)
REVIEW_EVERY = 5          # weekly review cadence (wd) plus event-triggered reviews
NOMINAL_APPROVAL = 10     # planner-side nominal submittal review period for the reported-date rival (ASSUMPTION, 5 to 20)
ROBUST_GAMMA = 1          # budget of simultaneous lot deviations for the budgeted robust rival (ASSUMPTION, 1 to 2)
CVAR_ALPHA = 0.9
N_SCEN = 20

PROVENANCE = {
    "LEAD_TIME": "ASSUMPTION", "JIT_OFFSET": "ASSUMPTION", "TRANSIT": "ASSUMPTION", "PACKAGE_VALUE": "ASSUMPTION",
    "EXPEDITE_SHARE": "ASSUMPTION", "EXPEDITE_COST": "REAL", "ALT_PREMIUM": "REAL", "CANCEL_FEE": "ASSUMPTION",
    "APPROVAL_COST": "ASSUMPTION", "DELAY_COST": "ASSUMPTION", "BUDGET_SHARE": "ASSUMPTION", "NOTICE_MAX": "ASSUMPTION",
    "NOTICE_FRACTION": "ASSUMPTION", "REVIEW_EVERY": "ASSUMPTION", "CVAR_ALPHA": "CITED", "N_SCEN": "ASSUMPTION",
    "lateness, record lag, quotation-to-order lag": "REAL (SCMS pools)",
    "activities, durations, start-no-earlier-than dates, calendar": "REAL (Schependomlaan planning)",
    "logic links": "REAL (IfcRelSequence, XML links) plus date-implied links (ASSUMPTION)",
    "lots, quantities, need dates": "REAL (IFC element-task links)",
}


def as_dict() -> dict:
    g = globals()
    keys = ["LEAD_TIME", "JIT_OFFSET", "TRANSIT", "PACKAGE_VALUE", "EXPEDITE_SHARE", "EXPEDITE_COST", "ALT_PREMIUM",
            "CANCEL_FEE", "APPROVAL_COST", "DELAY_COST", "BUDGET_SHARE", "NOTICE_MAX", "NOTICE_FRACTION",
            "REVIEW_EVERY", "CVAR_ALPHA", "N_SCEN"]
    return {k: g[k] for k in keys}
