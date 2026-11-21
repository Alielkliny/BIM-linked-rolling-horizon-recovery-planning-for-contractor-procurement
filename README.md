# Procurement disruption model with a BIM-linked schedule

Recovery decisions for a contractor's procurement lots, re-planned at each decision point as supply disruptions arrive one after another. A temporal dependency graph links BIM elements, schedule activities and procurement lots with event and record times; a discrete-event environment (SimPy) holds the true state; and the proposed policy solves a scenario-based recovery MILP with HiGHS at each decision point. Procurement timing is calibrated on a public delivery-history record, and the compared policies share the same snapshots, action library and common random numbers.

## Scope of this repository

This repository holds the code of the method, the comparison methods and the evaluation, with the split,
protocol and settings files that the scripts read. It does not hold the input data, run outputs, figures or tables.

## Repository layout

| Path | Contents |
|---|---|
| `src/common.py`, `params.py` | Paths, hashing helpers and model parameters with provenance and sensitivity ranges |
| `src/s01_scms_prepare.py`, `uncertainty.py` | Delivery-history record and the empirical timing distributions |
| `src/s02_bim_schedule.py`, `s03_instance.py`, `tdg.py` | BIM-linked activity and lot layer, decision instance and temporal dependency graph |
| `src/env.py` | Discrete-event evaluation environment |
| `src/milp.py` | Scenario-based recovery MILP and schedule passes |
| `src/policies.py` | The proposed framework, its ablations, established alternatives, practice proxy and references |
| `src/run_exp.py`, `run_split.py`, `tune.py`, `tune_ext.py` | Parallel runner, split runs and equal-size tuning |
| `src/s05_scores.py`, `s06_final_scores.py`, `s07_crossfit.py`, `s11_bootstrap_mean.py` | Score tables and paired tests |
| `src/sens.py`, `smoke.py` | Sensitivity study and smoke test |
| `tests/` | Verification and validation suite of the environment and the condensed solver check |
| `results/design_final.json`, `results/tune_dev_selection.json` | Selected design and tuned settings |

## Environment

The code was run with Python 3.13 on Windows 11. Install the packages with

```
python -m pip install -r requirements.txt
```

The versions in `requirements.txt` are those of the Python environment the code was run in.

## Input data

The input data are public and are not redistributed here. Place them as follows.

- USAID Supply Chain Management System delivery-history export (Wayback copy) as `data/raw/scms_wayback/scms.csv`.
- Schependomlaan open BIM dataset (TU Eindhoven, CC BY 4.0): IFC model, MS Project XML schedule and element tables, at the paths in `src/common.py`.

## Running the code

```
cd src
python s01_scms_prepare.py
python s02_bim_schedule.py
python s03_instance.py
python tune.py
python run_exp.py --split dev --policies proposed --streams 10-39 --tag dev_a7 --policy-kw '{"money_weight":1.0,"sibling_rule":false,"lone_rows":true}'
python run_split.py dev 10-39 dev_abl variant_cvar,abl_expected_value,abl_no_approval,abl_no_state,abl_no_bim,abl_static
python run_split.py probe 10-29 probe_a7 proposed,abl_expected_value,abl_no_approval,abl_no_state,abl_no_bim,abl_static
python run_split.py check 0-29 check_eval proposed,reported_rescheduling,budgeted_robust,buffer_management,reactive_expediting,no_recovery
python run_split.py reserve1 0-89 reserve1_a7 proposed,reported_rescheduling,budgeted_robust,buffer_management,reactive_expediting,no_recovery
python run_split.py reserve2 0-89 reserve2_a7 proposed,reported_rescheduling,budgeted_robust,buffer_management,reactive_expediting,no_recovery
python sens.py
python s07_crossfit.py <tag>
python s06_final_scores.py
python s11_bootstrap_mean.py
python ../tests/test_vv.py
python ../tests/test_condensed.py
```

Run the scripts from `src/`; they read and write `../data`, `../instances` and `../results`. The score scripts name the run tags they read; `run_split.py` takes each policy's tuned settings from `results/tune_dev_selection.json`.
