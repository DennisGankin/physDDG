#!/usr/bin/env python
"""AF-mode Q3: build the treated prediction tables (one column per treatment) for Domainome and abundance DMS.

  PYTHONPATH=. .venv/bin/python scripts/83_af_treatments.py --stage prep      # trimmed structures + simple feats
  PYTHONPATH=. .venv/bin/python scripts/83_af_treatments.py --stage p1        # fit the reference physics model
  PYTHONPATH=. .venv/bin/python scripts/83_af_treatments.py --stage preds     # treated predictions

Predictors
  B3+   fitted simple-descriptor model (results/baselines_ladder/models/B3+.json)
  P1    reference physics model fitted HERE on Megascale-train only (10 stage-1 energy terms + 2 statistical
        terms + 20 reference energies, same objective as the baseline ladder). It exists so that the energy-level
        treatments (repulsion cap, trimming) can be judged on a calibrated physics predictor; the production
        weights will come from the fitting agent (results/ladder_physics*).
  s1dF  unit-weight stage-1 packing free energy (no fitting at all)

Treatments (columns `<pred>__<treatment>`): off, trim50, trim70, mavisp, band<w>, cap_plddt, cap_soft, s2
(shell repack), burswap. Shrinkage (d) and the flag rules (f) are applied downstream in scripts/84.

Outputs: data/features/af/treated_<dataset>.parquet (git-ignored), results/af_mode/models/P1.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.af_common as C  # noqa: E402
from physddg import afmode as A  # noqa: E402
from physddg import features_simple as FS  # noqa: E402
from physddg import fit as FT  # noqa: E402

TRIM_VARIANTS = {"trim50": A.PRESETS["trim50"].trim, "trim70": A.PRESETS["trim70"].trim,
                 "mavisp": A.PRESETS["mavisp"].trim}
STRUCT_DIR = ROOT / "data" / "structures" / "af_mode"
P1_FEATURES = ["s1_dE_lj_atr", "s1_dE_lj_rep", "s1_dE_solv_apolar", "s1_dE_solv_polar", "s1_dE_hb_sc_bb",
               "s1_dE_hb_sc_sc", "s1_dE_elec_dh", "s1_dTS", "d_env_cn_ss_bb", "d_rama"]
BAND_W = [0.0, 0.25, 0.5, 0.75]


# ----------------------------------------------------------------------------- prep: trimmed structures
def _trim_one(args):
    src, variant = args
    from physddg.structure import annotate, load_protein
    cfg = TRIM_VARIANTS[variant]
    p = annotate(load_protein(ROOT / src if not Path(src).is_absolute() else src, keep_het=False), do_sasa=False)
    keep = A.trim_mask(p, cfg)
    dst = STRUCT_DIR / variant / (Path(src).stem + ".pdb")
    keys = {k for k, m in zip(p.keys, keep) if m}
    n = A.write_trimmed_pdb(ROOT / src if not Path(src).is_absolute() else src, keys, dst)
    return src, str(dst.relative_to(ROOT)), int(len(p)), int(keep.sum()), n


def prep(datasets: list[str], n_jobs: int = 32):
    from joblib import Parallel, delayed
    rows = []
    for ds in datasets:
        src = sorted(set(C.domainome(with_feats=False).structure) if ds == "domainome"
                     else set(FS.dataset_rows(ds).structure.dropna()))
        for variant in TRIM_VARIANTS:
            res = Parallel(n_jobs=n_jobs)(delayed(_trim_one)((s, variant)) for s in src)
            m = pd.DataFrame(res, columns=["structure", "trimmed_structure", "L", "n_keep", "n_written"])
            m["dataset"], m["variant"] = ds, variant
            rows.append(m)
            print(f"{ds}/{variant}: {len(m)} structures, kept {m.n_keep.sum()}/{m.L.sum()} residues "
                  f"({1 - m.n_keep.sum() / m.L.sum():.3f} trimmed)")
            # simple features on the trimmed models
            r = (C.domainome(with_feats=False) if ds == "domainome" else FS.dataset_rows(ds)).copy()
            r = C._norm(r[C.KEYS + ["structure"]])
            r = r.merge(m[["structure", "trimmed_structure"]], on="structure", how="left")
            r["structure"] = r.trimmed_structure
            r = r.drop(columns=["trimmed_structure"]).drop_duplicates(C.KEYS + ["structure"])
            sites = FS.site_tables(r.structure.unique(), n_jobs)
            feats = FS.variant_features(r, sites)
            out = C.FEAT / "af" / f"simple_{ds}_{variant}.parquet"
            feats.to_parquet(out, index=False)
            print(f"  -> {out.name}: site_found {feats.site_found.mean():.3f}")
    f = C.OUT / "trimmed_structures.csv"
    old = [pd.read_csv(f)] if f.exists() else []
    t = pd.concat(old + rows, ignore_index=True).drop_duplicates(["dataset", "variant", "structure"], keep="last")
    t.to_csv(f, index=False)


# ----------------------------------------------------------------------------- P1 reference physics model
def fit_p1(n_boot: int = 0) -> dict:
    out = C.OUT / "models"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "P1.json"
    a = pd.read_parquet(C.PROC / "megascale.parquet")
    c = pd.read_csv(C.PROC / "megascale_censored.csv", low_memory=False)
    d = pd.concat([a, c[c.split == "train"]], ignore_index=True)
    d = d[d.split == "train"].reset_index(drop=True)
    d = C.attach(d, C.phys("megascale", 1))
    d = FT.prepare_targets(d, "megascale").assign(dataset="megascale")
    cfg = FT.FitConfig(features=P1_FEATURES, ref=True, loss="huber", censor=True, within="rows",
                       ref_l2=1e-2, ref_dataset="megascale")
    m = FT.fit(d, cfg, n_boot=n_boot, name="P1",
               provenance={"note": "AF-mode reference physics model, Megascale-train only"})
    FT.save_model(m, path)
    print("P1 weights:", {k: round(v, 3) for k, v in m["weights"].items()})
    return m


# ----------------------------------------------------------------------------- production physics models
LADDER_PHYS = ROOT / "results" / "ladder_physics"


def production_models() -> dict:
    """The fitting agent's physics models (B4 = stage 1, B5 = stage 2), if they exist yet."""
    out = {}
    for n in ("B4", "B5"):
        f = LADDER_PHYS / "models" / f"{n}.json"
        if f.exists():
            out[n] = FT.load_model(f)
    return out


def _derive():
    import importlib.util
    spec = importlib.util.spec_from_file_location("s52", ROOT / "scripts" / "52_ladder_physics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def predict_production(model: dict, simple_feat: pd.DataFrame, phys_feat: pd.DataFrame, stage: int) -> np.ndarray:
    """Apply a results/ladder_physics model to a treated feature table (same `derive` transform as scripts/52)."""
    global _S52
    try:
        _S52
    except NameError:
        _S52 = _derive()
    cols = [c for c in phys_feat.columns if c not in simple_feat.columns or c in C.KEYS]
    t = simple_feat.merge(phys_feat[cols], on=C.KEYS, how="left")
    opts = model.get("provenance", {}).get("opts", {"elec": "dh", "transform": "slog_rep", "pro": "none",
                                                    "avg": "avg", "drefp": stage == 1})
    t = _S52.derive(t, stage, opts)
    return FT.predict(model, t)


# ----------------------------------------------------------------------------- treated predictions
def treated(ds: str) -> pd.DataFrame:
    """One row per variant, one column per (predictor, treatment)."""
    base = C.domainome(with_feats=False) if ds == "domainome" else None
    if base is None:
        rows = FS.dataset_rows(ds)
        base = C._norm(rows).drop_duplicates(C.KEYS + ["structure"]).reset_index(drop=True)
        meta = pd.read_csv(C.PROC / f"{ds}.csv", low_memory=False)
        meta = C._norm(meta).drop_duplicates(C.KEYS)
        keep = [c for c in ("score", "cluster", "uniprot", "plddt", "DMS_id", "assay") if c in meta]
        base = base.merge(meta[C.KEYS + keep], on=C.KEYS, how="left")
        base["mscore"] = -base["score"]
    mdl = C.models(("B3+", "B3"))
    p1 = FT.load_model(C.OUT / "models" / "P1.json")
    prod = production_models()
    out = base.copy()
    simple_off = C.simple(ds).drop(columns=["structure"])

    def add(tag: str, simple_feat: pd.DataFrame | None, phys_feat: pd.DataFrame | None, on_structure=True):
        t = None
        if simple_feat is not None:
            t = C.attach(base[C.KEYS + (["structure"] if on_structure else [])], simple_feat,
                         on_structure=on_structure)
            for name, m in mdl.items():
                out[f"{name}__{tag}"] = FT.predict(m, t)
            if tag == "off":
                for c in ("bur", "rsa", "cn", "occn", "de_env_bb"):
                    out[c] = t[c].to_numpy()
        if phys_feat is not None:
            tp = C.attach(base[C.KEYS + (["structure"] if on_structure else [])], phys_feat,
                          on_structure=on_structure)
            out[f"P1__{tag}"] = FT.predict(p1, tp)
            out[f"s1dF__{tag}"] = tp["s1_dF"].to_numpy() if "s1_dF" in tp else np.nan
            if tag == "off":
                out["plddt_site"] = tp["ctx_bfactor"].to_numpy()
                out["hse_up"] = tp["ctx_hse_up"].to_numpy()
                out["cn_site"] = tp["ctx_cn"].to_numpy()
            if prod and "B4" in prod:                      # production stage-1 model on the same treated features
                sf = t if t is not None else C.attach(base[C.KEYS], simple_off, on_structure=False)
                out[f"B4__{tag}"] = predict_production(prod["B4"], sf, tp, 1)
            t = tp if t is None else t
        return t

    afdir = C.FEAT / "af"
    add("off", C.simple(ds), C.phys(ds, 1))
    # (c) shell repack = stage 2 (for the production models this is B5 vs B4, both fitted on their own stage)
    s2 = C.phys(ds, 2)
    t2 = C.attach(base[C.KEYS + ["structure"]], s2)
    out["s1dF__s2"] = t2["s2_dF"].to_numpy()
    ren = {c: c.replace("s2_", "s1_") for c in t2.columns if c.startswith("s2_")}
    out["P1__s2"] = FT.predict(p1, t2.rename(columns=ren))
    if prod and "B5" in prod:
        sf = C.attach(base[C.KEYS], simple_off, on_structure=False)
        out["B4__s2"] = predict_production(prod["B5"], sf, t2, 2)
    # (a) trimming
    for v in TRIM_VARIANTS:
        sf = afdir / f"simple_{ds}_{v}.parquet"
        pf = afdir / f"phys_s1_{ds}_{v}.parquet"
        add(v, C.load_feat(sf).drop(columns=["structure"]) if sf.exists() else None,
            C.load_feat(pf).drop(columns=["structure"]) if pf.exists() else None, on_structure=False)
    # (a') 50-70 band down-weighting: blend of the trim50 (band kept) and trim70 (band dropped) features
    if all((afdir / f"phys_s1_{ds}_{v}.parquet").exists() for v in ("trim50", "trim70")):
        fk = C.attach(base[C.KEYS], C.load_feat(afdir / f"phys_s1_{ds}_trim50.parquet").drop(columns=["structure"]),
                      on_structure=False)
        fd = C.attach(base[C.KEYS], C.load_feat(afdir / f"phys_s1_{ds}_trim70.parquet").drop(columns=["structure"]),
                      on_structure=False)
        num = [c for c in fk.columns if fk[c].dtype.kind == "f" and c in fd]
        for w in BAND_W:
            bl = fk.copy()
            bl[num] = A.blend(fk[num].to_numpy(float), fd[num].to_numpy(float), w)
            out[f"P1__band{w}"] = FT.predict(p1, bl)
            out[f"s1dF__band{w}"] = bl["s1_dF"].to_numpy()
    # (b) caps
    for v in ("cap_plddt", "cap_soft"):
        pf = afdir / f"phys_s1_{ds}_{v}.parquet"
        if pf.exists():
            add(v, None, C.load_feat(pf).drop(columns=["structure"]), on_structure=False)
    # (e) backbone-only burial in the 70-90 band (descriptor models only)
    t = C.attach(base[C.KEYS + ["structure"]], C.simple(ds))
    alt = A.cn_to_burial(t["cn"].to_numpy(float))
    pl = out["plddt_site"].to_numpy(float)
    for lo, hi, tag in ((70.0, 90.0, "burswap"), (0.0, 1e9, "burcn")):
        t2 = t.copy()
        t2["bur"] = A.burial_swap(t["bur"].to_numpy(float), alt, pl, lo, hi)
        for name, m in mdl.items():
            out[f"{name}__{tag}"] = FT.predict(m, t2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="preds", choices=("prep", "p1", "preds"))
    ap.add_argument("--datasets", nargs="+", default=["domainome", "abundance_dms"])
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    C.ensure_out()
    if a.stage == "prep":
        prep(a.datasets, a.jobs)
    elif a.stage == "p1":
        fit_p1()
    else:
        for ds in a.datasets:
            t = treated(ds)
            p = C.FEAT / "af" / f"treated_{ds}.parquet"
            t.to_parquet(p, index=False)
            cols = [c for c in t.columns if "__" in c]
            print(f"{p.name}: {len(t)} rows, {len(cols)} prediction columns, coverage "
                  + ", ".join(f"{c}={t[c].notna().mean():.2f}" for c in cols[:6]))


if __name__ == "__main__":
    main()
