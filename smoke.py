import sys, time, json
from common import INST, read_json
from env import Episode, draw_exogenous
from policies import ROSTER
inst = read_json(INST / "schependomlaan.json")
names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ROSTER)
for seed in [1, 2]:
    ex = draw_exogenous(inst, "calib", "sequential", "medium", seed)
    print("seed", seed, "base late lots", int((ex["base_late"]>0).sum()), "disr", [(d["tau"], d.get("supplier"), d["delta"]) for d in ex["disruptions"]])
    for nm in names:
        fac, role, pooling = ROSTER[nm]
        t0 = time.time()
        ep = Episode(inst, ex, None, pooling=pooling, log_actions=True)
        pol = fac(inst, seed, ep)
        ep.policy = pol
        r = ep.run()
        r["sec"] = round(time.time() - t0, 2)
        err = r.pop("first_error")
        print(f"  {nm:24s} delay={r['delay']:6.1f} spent={r['spent']:9.0f} exp={r['n_expedite']} req={r['n_request']} alt={r['n_order_alt']} infeas={r['infeasible']} rev={r['reviews']} err={r['policy_errors']} solves={getattr(pol,'solves','-')} sec={r['sec']}")
        if err: print(err[-1500:])
