#!/usr/bin/env python
"""D23(B): produce the shipped model files from the fitted models — no hand-made aliases (audit 2, I8).

Decision rule (fixed before looking at anything, and applied to val/CV/G6 evidence ONLY):
  ship the family that wins on (i) Megascale-val natural per-protein rho — both families scored on the same
  held-out rows, the only directly comparable non-test number — then (ii) grouped CV on each model's own
  training set, then (iii) G6 transfer (sign flips / undetermined / >3x). Ties go to the model with fewer
  ddG-fitted terms. Evidence: results/ladder_physics/family_decision.csv + ss_decision.csv.
Outcome (see notes/ladder_physics.md "Errata"): the Megascale-only family wins on (i) and (ii) and ties on
(iii), so B6g / B6gs1 / B6gaf ship and the joint fits become the documented multi-assay variant.

  PYTHONPATH=. .venv/bin/python scripts/57_export_final.py [--check]
Writes results/ladder_physics/models/{final_stage1,final_stage2,final_af,alt_multiassay_stage{1,2},alt_multiassay_af}.json
and models/SHIPPED.json (the provenance record). --check verifies the files match their sources and exits non-zero.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

MODELS = ROOT / "results" / "ladder_physics" / "models"
# target file -> (source model, one-line description)
SHIP = {
    "final_stage2": ("B6g", "DEFAULT. Stage 2 (site + flexible neighbours), fitted on Megascale-train only."),
    "final_stage1": ("B6gs1", "DEFAULT FAST MODE. Stage 1 (Boltzmann site, fixed environment), Megascale-train only."),
    "final_af": ("B6gaf", "AF INPUTS ONLY. Stage 2 + pLDDT attenuation; the term is identically 0 on crystal input."),
    "alt_multiassay_stage2": ("B6gj", "Multi-assay variant: joint Megascale+FireProt fit, stage 2."),
    "alt_multiassay_stage1": ("B6gjs1", "Multi-assay variant: joint fit, stage 1."),
    "alt_multiassay_af": ("B6gjaf", "Multi-assay variant: joint fit + pLDDT attenuation (AF inputs only)."),
}
RULE = ("Megascale-val natural per-protein rho (identical held-out rows) > grouped CV on each model's own "
        "training set > G6 transfer; ties to fewer fitted terms. No test set was used.")


def param_count(m: dict) -> dict:
    """ddG-fitted numbers, split the way the model card must report them (audit 2, I5)."""
    w = len(m["weights"])
    ref = len(m.get("ref") or {})
    pinned = len(m.get("inactive_features") or [])
    n = {"weights": w, "weights_pinned_at_zero": pinned, "reference_energies": ref, "intercept": 1,
         "nested_scale_base": 0}
    if m.get("scale_base"):
        b = m["scale_base"]
        n["nested_scale_base"] = len(b["weights"]) + len(b.get("ref") or {}) + 1
    n["total_ddG_fitted_at_inference"] = w + ref + 1 + n["nested_scale_base"]
    n["effective_ddG_fitted"] = n["total_ddG_fitted_at_inference"] - pinned
    nu = m.get("nuisance") or {}
    n["also_fitted_but_dropped_at_inference"] = (len(nu.get("scale", {})) + len(nu.get("offset", {}))
                                                 + sum(len(v) for v in (nu.get("delta_ref") or {}).values()))
    return n


def build(check: bool) -> int:
    rec, bad = {}, 0
    for dst, (src, desc) in SHIP.items():
        sp = MODELS / f"{src}.json"
        if not sp.exists():
            print(f"MISSING SOURCE {sp}", file=sys.stderr)
            bad += 1
            continue
        m = json.loads(sp.read_text())
        m["shipped_as"], m["source_model"], m["description"] = dst, src, desc
        m["selection_rule"] = RULE
        m["inference_note"] = ("`nuisance` (per-assay scale/offset, Megascale delta_ref) is fitted but NEVER "
                               "applied at inference; `ref` is the assay-agnostic prior + shared part.")
        m["parameter_count"] = param_count(m)
        txt = json.dumps(m, indent=1)
        dp = MODELS / f"{dst}.json"
        if check:
            if not dp.exists() or dp.read_text() != txt:
                print(f"STALE: {dp.name} does not match {src}.json", file=sys.stderr)
                bad += 1
        else:
            dp.write_text(txt)
        rec[dst] = {"source": src, "stage": m["stage"], "scale_kind": m.get("scale_kind"),
                    "n_weights": len(m["weights"]), **param_count(m)}
    if not check:
        (MODELS / "SHIPPED.json").write_text(json.dumps(
            {"decision": "D23(B): ship the Megascale-only family; joint fits are the multi-assay variant",
             "rule": RULE, "evidence": ["results/ladder_physics/family_decision.csv",
                                        "results/ladder_physics/ss_decision.csv"],
             "produced_by": "scripts/57_export_final.py", "models": rec}, indent=1))
        print(pd.DataFrame(rec).T.to_string())
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the shipped files match their sources")
    a = ap.parse_args()
    bad = build(a.check)
    if a.check:
        print("shipped files are current" if not bad else f"{bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
