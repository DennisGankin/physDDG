#!/usr/bin/env python
"""Gate G3 pre-test: stage 1 averaged over K perturbed backbones (physddg.mutate.stage1_ensemble), Megascale-val only.

Backbones: native + K = 8 copies with residues rigidly displaced by a smooth random field (GP sample, RBF length 6 Å,
RMS amplitude `amp`). Aggregates per (site, aa): mean over backbones, Boltzmann over backbones (no strain energy).
Evaluated (Megascale-val, natural domains; per-protein median and pooled Spearman, plus the buried small->large stratum,
rsa < 0.25 and dvol < -0.3) for (a) raw unit-weight dF and (b) the fitting agent's B4 model
(results/ladder_physics/models/B4.json) with its stage-1 inputs replaced by the ensemble versions (weights not refitted).
Output: results/mutate/backbone_ensemble.csv, runtime in results/mutate/backbone_ensemble_runtime.csv
Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/58_backbone_ensemble.py [amp ...]
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physddg import fit as FT  # noqa: E402
from physddg import topology as T  # noqa: E402
from physddg.mutate import MutateConfig, stage1_ensemble  # noqa: E402
from physddg.structure import annotate, load_protein  # noqa: E402

spec = importlib.util.spec_from_file_location("lp52", ROOT / "scripts" / "52_ladder_physics.py")
LP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LP)
OUT = ROOT / "results" / "mutate"


def ensemble_features(va: pd.DataFrame, amp: float, K: int = 8) -> dict[str, pd.DataFrame]:
    """Per aggregate ('native', 'mean', 'boltz'): DataFrame aligned to va with replacement s1_* columns."""
    cfg = MutateConfig(rep_soft=0.0)            # identical energy terms to the features B4 was fitted on
    terms = None
    outs = {k: {} for k in ("native", "mean", "boltz")}
    rt = []
    for st, g in va.groupby("structure"):
        p = annotate(load_protein(st if Path(st).is_absolute() else ROOT / st, keep_het=False))
        key = {(c, r): i for i, (c, r) in enumerate(zip(p.chain.tolist(), p.resnum.tolist()))}
        si = np.array([key[(c, str(r))] for c, r in zip(g.chain, g.pdb_resnum)])
        sites, inv = np.unique(si, return_inverse=True)
        t0 = time.time()
        r = stage1_ensemble(p, sites, cfg, K=K, amp=amp)
        rt.append(dict(structure=st, L=len(p), n_sites=len(sites), seconds=time.time() - t0, K=K))
        if terms is None:
            from physddg import energy as E
            terms = E.term_names(cfg.params)
        iw, im = g.wt.map(T.AA_INDEX).to_numpy(), g.mut.map(T.AA_INDEX).to_numpy()
        d = lambda a: a[inv, im] - a[inv, iw]
        aggs = {"native": (r["F_k"][0], r["E_k"][0], None, None), "mean": (r["F_mean"], r["E_mean"], r["S_mean"], r["prior_mean"]),
                "boltz": (r["F_boltz"], r["E_boltz"], r["S_boltz"], r["prior_boltz"])}
        for name, (F, Ea, S_, P_) in aggs.items():
            cols = {"s1_dF": d(F)}
            for j, t in enumerate(terms):
                cols[f"s1_dE_{t}"] = d(Ea[..., j])
            if S_ is not None:
                cols["s1_dTS"] = cfg.kT * d(S_)
                cols["s1_dprior_avg"] = d(P_)
            outs[name][st] = pd.DataFrame(cols, index=g.index)
    return {k: pd.concat(v.values()).loc[va.index] for k, v in outs.items()}, pd.DataFrame(rt)


def metrics(df: pd.DataFrame, col: str) -> dict:
    nat = df[(df.domain_origin == "natural") & df.y.notna() & df[col].notna()]
    per = [spearmanr(x[col], x.y).statistic for _, x in nat.groupby("protein_id") if len(x) >= 20]
    bsl = nat[(nat.rsa < 0.25) & (nat.dvol < -0.3)]
    return {"nat_prot": float(np.nanmedian(per)), "nat_pooled": spearmanr(nat[col], nat.y).statistic,
            "buried_small_large": spearmanr(bsl[col], bsl.y).statistic, "n_bsl": len(bsl), "n": len(nat)}


def main(amps=(0.3,)):
    _, va, _ = LP.train_tables()
    model = FT.load_model(ROOT / "results/ladder_physics/models/B4.json")
    rows = []
    base = va.copy()
    base["pred"] = LP.predict_models({"B4": model}, base)["B4"]
    rows.append({"amp": 0.0, "agg": "features_file", "raw": metrics(base, "s1_dF"), "B4": metrics(base, "pred")})
    rts = []
    for amp in amps:
        ens, rt = ensemble_features(va, amp)
        rt["amp"] = amp
        rts.append(rt)
        for name, cols in ens.items():
            d = va.copy()
            for c in cols:
                d[c] = cols[c]
            d["pred"] = LP.predict_models({"B4": model}, d)["B4"]
            rows.append({"amp": amp, "agg": name, "raw": metrics(d, "s1_dF"), "B4": metrics(d, "pred")})
            print(rows[-1], flush=True)
    flat = [{"amp": r["amp"], "agg": r["agg"], **{f"raw_{k}": v for k, v in r["raw"].items()},
             **{f"B4_{k}": v for k, v in r["B4"].items()}} for r in rows]
    res = pd.DataFrame(flat)
    res.round(4).to_csv(OUT / "backbone_ensemble.csv", index=False)
    rt = pd.concat(rts)
    rt.to_csv(OUT / "backbone_ensemble_runtime.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string())
    print(rt.groupby("amp")[["n_sites", "seconds"]].sum())


if __name__ == "__main__":
    main(tuple(float(x) for x in sys.argv[1:]) or (0.3,))
