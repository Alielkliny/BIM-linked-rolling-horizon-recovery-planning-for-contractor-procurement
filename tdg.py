"""Temporal dependency graph: typed BIM, schedule and procurement nodes with bitemporal facts.

Every fact carries an event time (when it happened in the project) and a recorded time (when the project
team learned it). `as_of(t)` answers every planning query with facts recorded no later than t, so a historical
decision uses only information available at the decision date.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import networkx as nx

NEG_INF = -10 ** 9


@dataclass
class Fact:
    event_time: float
    recorded_at: float
    kind: str
    value: object
    source: str = ""


class TemporalDependencyGraph:
    def __init__(self, inst: dict, elements: dict | None = None):
        G = nx.MultiDiGraph()
        for a in inst["activities"]:
            G.add_node(("act", a["uid"]), type="activity", name=a["name"], r=a["r"], d=a["d"])
        for a in inst["activities"]:
            for p in a["preds"]:
                G.add_edge(("act", p), ("act", a["uid"]), key="PRECEDES", rel="PRECEDES")
        for l in inst["lots"]:
            G.add_node(("lot", l["id"]), type="order_line", package=l["package"], P=l["planned_delivery"])
            G.add_node(("sup", l["supplier"]), type="supplier")
            G.add_node(("alt", l["supplier"]), type="alternative_source")
            G.add_edge(("sup", l["supplier"]), ("lot", l["id"]), key="SUPPLIES", rel="SUPPLIES")
            G.add_edge(("alt", l["supplier"]), ("lot", l["id"]), key="APPROVED_ALTERNATIVE_TO", rel="APPROVED_ALTERNATIVE_TO")
            G.add_edge(("lot", l["id"]), ("act", l["need_uid"]), key="REQUIRED_BY", rel="REQUIRED_BY")
        if elements:
            for gid, (lot_id, uid) in elements.items():
                G.add_node(("el", gid), type="element")
                G.add_edge(("el", gid), ("lot", lot_id), key="SUPPLIED_IN", rel="SUPPLIED_IN")
                G.add_edge(("el", gid), ("act", uid), key="INSTALLED_BY", rel="INSTALLED_BY")
        self.G = G
        self._facts: dict = {}      # node -> list of Fact sorted by recorded_at
        self._keys: dict = {}

    def add_fact(self, node, fact: Fact) -> None:
        lst = self._facts.setdefault(node, [])
        keys = self._keys.setdefault(node, [])
        i = bisect.bisect_right(keys, fact.recorded_at)
        lst.insert(i, fact)
        keys.insert(i, fact.recorded_at)

    def facts_as_of(self, node, t: float, kind: str | None = None) -> list[Fact]:
        keys = self._keys.get(node, [])
        i = bisect.bisect_right(keys, t)
        out = self._facts.get(node, [])[:i]
        return [f for f in out if kind is None or f.kind == kind]

    def latest(self, node, t: float, kind: str):
        """Latest fact of a kind known at t: the highest event time among facts recorded by t."""
        fs = self.facts_as_of(node, t, kind)
        if not fs:
            return None
        return max(fs, key=lambda f: (f.event_time, f.recorded_at))

    def future_leak(self, t: float) -> int:
        """Count of facts a query at t would return that were recorded after t (must be zero)."""
        n = 0
        for node in self._facts:
            for f in self.facts_as_of(node, t):
                if f.recorded_at > t:
                    n += 1
        return n

    def affected_activities(self, lot_id: str) -> set:
        """Activities reachable from an order line through REQUIRED_BY and PRECEDES edges."""
        start = ("act", next(v for _, v, k in self.G.out_edges(("lot", lot_id), keys=True) if k == "REQUIRED_BY")[1])
        acts = {start[1]}
        for n in nx.descendants(self.G, start):
            if n[0] == "act":
                acts.add(n[1])
        return acts
