#!/usr/bin/env python
"""Physics rungs B4 (stage-1 engine terms) / B5 (stage-2, flexible neighbours): selection on Megascale-val
(+ grouped CV on train), drop-one ablations (train CV / val only), final fits with cluster-bootstrap SEs,
evaluation on all benchmarks with FoldX / B3+ / tmol-refit / KORPM / DDGun3D on identical rows, strata,
Ssym antisymmetry study (D15), G6 transfer, report tables.

  PYTHONPATH=. .venv/bin/python scripts/52_ladder_physics.py --stage select|ablate|fit|eval|antisym|transfer|report|all
Outputs: results/ladder_physics/ (models/*.json, selection.csv, ablation.csv, metrics.csv, paired.csv, strata.csv,
stabilising.csv, antisymmetry.csv, paired_af_agreement.csv, transfer_*.csv, weights.csv, main_table.md).
"""
import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from physddg import eval as E  # noqa: E402
from physddg import features_extra as X  # noqa: E402
from physddg import fit as FT  # noqa: E402

_spec = importlib.util.spec_from_file_location("ladder51", Path(__file__).with_name("51_baseline_ladder.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

ROOT = E.ROOT
OUT = ROOT / "results" / "ladder_physics"
FEAT = ROOT / "data" / "features"
KEYS = list(E.KEYS)
KS = KEYS + ["structure"]
TERMS = ["lj_atr", "lj_rep", "solv_apolar", "solv_polar", "hb_sc_bb", "hb_sc_sc", "hb_bb_bb", "ELEC", "disulf"]
STAT = ["d_env_cn_ss_bb", "d_rama", "d_rama_prev"]
# B6 additions (residual analysis, notes/residual_analysis.md). site_scsc / site_hbsc_bur are WT-site main
# effects; ncap_pref (PDB-derived N-cap potential, scripts/53) and helix_dipole are state-function differences.
# site_hbsc_bur fails the G6 3x rule (FireProt/Megascale scaled ratio 0.25, undetermined on FireProt) and is
# therefore NOT in the headline B6; kept as the reported variant B6hb.
B6_XTAL = ["site_scsc", "ncap_pref", "helix_dipole"]
B6_HB = B6_XTAL + ["site_hbsc_bur"]
B6_D15 = ["scsc_dvol", "ncap_pref", "helix_dipole"]                    # D15-compliant variant
# D19: the shipped default must generalise across assays -> no site_scsc (hurts FireProt/S669/S461, G6 edge)
# and no site_hbsc_bur (fails G6). B6g = B5 + N-cap (PISCES-derived) + helix dipole; both are state functions.
B6G = ["ncap_pref", "helix_dipole"]
B3P_BASE = ["bur", "pro_nhhb", "gly_posphi", "occ_dvol_loss"]   # B3+ after G6 pruning (no cis-Pro, no dV+)
# D17: the headline models use the plain disulfide indicator. `ss_broken_nc` (catalytic-CXXC exception) is a
# separate reported variant because the exception was derived from two FireProt-TEST rows (audit item 1).
SS_FEAT = {False: "ss_broken", True: "ss_broken_nc"}
# physics sign constraints: every dE term and the rotamer prior destabilise when they increase; -T dS lowers F
POS = {"site_scsc", "site_hbsc_bur", "ncap_pref", "helix_dipole", "scsc_dvol", "hbsc_dpolar",
       "lj_atr", "lj_rep", "lj_rep_pro", "solv_apolar", "solv_polar", "hb_sc_bb", "hb_sc_sc", "hb_bb_bb", "elec_dh",
       "elec_bur", "elec_dd", "disulf", "dprior_avg", "dF", "d_env_cn_ss_bb", "d_rama", "d_rama_prev", "bur",
       "pro_nhhb", "occ_dvol_loss", "ss_broken", "ss_broken_nc"}
NEG = {"dTS", "scale_lowplddt", "scale_burial"}   # attenuation terms must reduce |prediction|


# ----------------------------------------------------------------------------- features
def add_phys(df: pd.DataFrame, name: str, cap: str = "") -> pd.DataFrame:
    """Merge stage-1/2 engine features onto rows with KEYS+structure. cap in {'', capA, capB, capC} selects E1's
    D9 repulsion-cap variant for stage 1 (its columns are suffixed `_<cap>` so several can coexist)."""
    out = df
    for s in (1, 2):
        p = pd.read_parquet(FEAT / f"phys_s{s}_{name}.parquet")
        keep = [c for c in p.columns if c.startswith(f"s{s}_")] + (["d_rama", "d_rama_prev", "d_env_cn_ss", "d_env_cn_ss_bb",
                                                                       "d_env_rsa_ss", "d_sc_entropy", "d_posphi", "ctx_bfactor", "ctx_rsa"] if s == 1 else [])
        p = p[KS + keep].drop_duplicates(KS)
        p["pdb_resnum"] = E._norm_resnum(p["pdb_resnum"])
        out = out.merge(p, on=KS, how="left", suffixes=("", f"_s{s}"))
    if cap:
        for s in (1, 2):
            f = FEAT / f"phys_s{s}_{name}_{cap}.parquet"
            if not f.exists():
                continue
            q = pd.read_parquet(f)
            keep = [c for c in q.columns if c.startswith(f"s{s}_")]
            q = q[KS + keep].drop_duplicates(KS).rename(columns={c: f"{c}_{cap}" for c in keep})
            q["pdb_resnum"] = E._norm_resnum(q["pdb_resnum"])
            out = out.merge(q, on=KS, how="left")
    assert len(out) == len(df)
    return out


def add_extra(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Merge physddg.features_extra site descriptors (cached per dataset) and the candidate per-variant terms."""
    st = X.site_table(df.structure.unique(), name).drop(columns=["structure_abs"]).add_prefix("x_").rename(
        columns={"x_structure": "structure", "x_chain": "chain", "x_pdb_resnum": "pdb_resnum"})
    d = df.assign(pdb_resnum=df.pdb_resnum.astype(str), chain=df.chain.astype(str))
    n0 = len(d)
    d = d.merge(st, on=["structure", "chain", "pdb_resnum"], how="left")
    assert len(d) == n0, (len(d), n0)
    assert "dvol" in d, "add_extra needs the simple features (dvol) merged first"
    cand = X.candidate_features(d)
    keep = [c for c in cand.columns if c not in d.columns]
    return pd.concat([d, cand[keep]], axis=1)


def cxxc_flags(df: pd.DataFrame) -> pd.Series:
    """True where WT Cys sits in a C-X-X-C motif (catalytic/strained redox disulfides, e.g. thioredoxin CGPC)."""
    from physddg import structure as S
    flag = pd.Series(False, index=df.index)
    cand = df[(df.wt == "C") & (df.get("ss_broken", 0) > 0)]
    for st, g in cand.groupby("structure"):
        p = S.load_protein(st if Path(st).is_absolute() else ROOT / st, keep_het=False)
        for ix, r in g.iterrows():
            i = p.index(r.chain, str(r.pdb_resnum))
            flag[ix] = any(0 <= i + k < len(p) and p.seq[i + k] == "C" and p.chain[i + k] == r.chain for k in (-3, 3))
    return flag


def derive(df: pd.DataFrame, stage: int, opts: dict, caps: dict | None = None) -> pd.DataFrame:
    """Add model columns: transformed energy terms `e_<term>`, `dTS`, `dprior_avg`, `dref_apolar/polar`, `dF`,
    ss_broken_nc. opts: elec (dh|bur|dd), transform (none|slog_rep|slog_all|clip_rep5|clip_rep20|clip_pct),
    pro (none|split), avg (avg|min; stage 1 only)."""
    d = df.copy()
    s = f"s{stage}"
    cap = opts.get("cap", "")
    sfx = f"_{cap}" if cap else ""
    mid = "dEmin" if (opts.get("avg") == "min" and stage == 1) else "dE"
    tr = opts["transform"]
    slog = lambda x: np.sign(x) * np.log1p(np.abs(x))
    for t in TERMS:
        name = f"elec_{opts['elec']}" if t == "ELEC" else t
        src = f"{s}_{mid}_{name}"
        if t == "lj_rep" and opts.get("rep") == "soft" and f"{s}_{mid}_lj_rep_soft{sfx}" in d:
            src = f"{s}_{mid}_lj_rep_soft{sfx}"          # E1's log-softened repulsion (already saturating)
        elif cap and f"{src}{sfx}" in d:
            src = f"{src}{sfx}"
        x = d[src].astype(float)
        if opts.get("rep") == "soft" and t == "lj_rep":
            pass                                           # already saturating: no extra transform
        elif tr == "slog_all" or (tr == "slog_rep" and t == "lj_rep"):
            x = slog(x)
        elif tr.startswith("clip_rep") and t == "lj_rep":
            c = float(tr[8:])
            x = x.clip(-c, c)
        elif tr == "clip_pct" and caps:
            x = x.clip(*caps[name])
        d[f"e_{name}"] = x
    if opts.get("pro") == "split":
        isp = ((d.wt == "P") | (d.mut == "P")).astype(float)
        d["e_lj_rep_pro"] = d["e_lj_rep"] * isp
        d["e_lj_rep"] = d["e_lj_rep"] * (1 - isp)
    d["dTS"], d["dprior_avg"] = d[f"{s}_dTS"], d[f"{s}_dprior_avg"]
    d["dref_apolar"], d["dref_polar"] = d[f"{s}_dref_apolar"], d[f"{s}_dref_polar"]
    fcol = f"{s}_dF{sfx}" if (cap and f"{s}_dF{sfx}" in d) else f"{s}_dF"
    d["dF"] = slog(d[fcol]) if tr != "none" else d[fcol]
    if "ss_broken" in d:
        d["ss_broken"] = d["ss_broken"].fillna(0)
        d["ss_broken_nc"] = d["ss_broken"] * (~d.get("cxxc", pd.Series(False, index=d.index)).astype(bool))
    return d


def energy_cols(opts) -> list:
    # disulf energy dropped from the model: no support on Megascale (std ~0); the CXXC-aware indicator ss_broken_nc covers it
    cols = [f"e_elec_{opts['elec']}" if t == "ELEC" else f"e_{t}" for t in TERMS if t != "disulf"]
    if opts.get("pro") == "split":
        cols.append("e_lj_rep_pro")
    return cols + ["dTS", "dprior_avg"]


def feature_set(kind: str, opts: dict) -> tuple[list, dict]:
    """kind -> (features, extra FitConfig kwargs). opts['cxxc'] picks the disulfide indicator (D17)."""
    en, refp = energy_cols(opts), (["dref_apolar", "dref_polar"] if opts.get("drefp", True) else [])
    B3P = B3P_BASE + [SS_FEAT[bool(opts.get("cxxc", False))]]
    sets = {"b6g": en + STAT + B3P + B6G + refp,
            "b6": en + STAT + B3P + B6_XTAL + refp, "b6hb": en + STAT + B3P + B6_HB + refp,
            "b6d15": en + STAT + B3P + B6_D15 + refp,
            "full": en + STAT + B3P + refp, "noenv": en + STAT[1:] + B3P + refp, "phys": en + refp,
            "dF": ["dF"] + refp, "odd": en + STAT + refp, "odd_sym": en + STAT + refp}
    extra = {"odd": {"intercept": False}, "odd_sym": {"intercept": False, "sym_pair": True}}.get(kind, {})
    return sets[kind], extra


def signs_for(feats, constrained=True) -> dict:
    if not constrained:
        return {}
    out = {}
    for f in feats:
        base = f[2:] if f.startswith("e_") else f
        out[f] = 1 if base in POS else (-1 if base in NEG else 0)
    return {k: v for k, v in out.items() if v}


def make_cfg(feats, sel, extra=None, constrained=True) -> FT.FitConfig:
    kw = dict(features=list(feats), ref=sel.get("ref", True), loss=sel.get("loss", "huber"), censor=True, within="rows",
              ref_l2=sel.get("ref_l2", 1e-2), ref_dataset="megascale", signs=signs_for(feats, constrained))
    kw.update(extra or {})
    return FT.FitConfig(**kw)


# ----------------------------------------------------------------------------- data
_CACHE = {}


def train_tables(cap: str = ""):
    if _CACHE.get("cap", "") != cap:
        _CACHE.clear()
        _CACHE["cap"] = cap
    if "train" not in _CACHE:
        cap = _CACHE.get("cap", "")
        tr = add_extra(add_phys(L.megascale_train_table("train"), "megascale", cap), "megascale")
        va = add_extra(add_phys(L.megascale_train_table("val"), "megascale", cap), "megascale")
        fp = add_extra(add_phys(L.fireprot_train_table(), "fireprot", cap), "fireprot")
        keep = KS + ["y", "lo", "hi", "sigma_meas", "dataset", "cluster", "domain_origin", "ddg", "cxxc", "rsa", "dvol",
                     "bur", "pro_nhhb", "gly_posphi", "occ_dvol_loss", "ss_broken"]
        out = []
        for d in (tr, va, fp):
            d["cxxc"] = cxxc_flags(d)
            if "domain_origin" not in d:
                d["domain_origin"] = "natural"
            ph = [c for c in d.columns if c.startswith(("s1_", "s2_", "d_", "x_"))
                  or c in set(B6_HB + B6_D15 + ["hbsc_dpolar", "flag_lowplddt", "flag_buried"])]
            out.append(d[[c for c in keep if c in d] + ph].reset_index(drop=True))
        _CACHE["train"] = tuple(out)
    return _CACHE["train"]


def clip_caps(train, stage, opts) -> dict:
    s = f"s{stage}"
    mid = "dEmin" if (opts.get("avg") == "min" and stage == 1) else "dE"
    caps = {}
    for t in TERMS:
        name = f"elec_{opts['elec']}" if t == "ELEC" else t
        caps[name] = tuple(np.nanpercentile(train[f"{s}_{mid}_{name}"], [0.5, 99.5]).tolist())
    return caps


def val_metrics(m, va):
    v = va[va.y.notna()].assign(ddg=lambda d: d.y)
    v = v.assign(pred=FT.predict(m, v))
    out = {}
    for tag, sub in (("all", v), ("natural", v[v.domain_origin == "natural"])):
        r = E.evaluate(sub.dropna(subset=["pred"]), "pred")
        out[f"val_{tag}_prot"], out[f"val_{tag}_pooled"] = r["spearman_prot_median"], r["spearman"]
    return out


def _run_config(label, stage, opts, kind, sel, tr, va, cv=True, drop=None, constrained=True):
    import torch
    torch.set_num_threads(2)
    caps = clip_caps(tr, stage, opts) if opts["transform"] == "clip_pct" else None
    trd, vad = derive(tr, stage, opts, caps), derive(va, stage, opts, caps)
    feats, extra = feature_set(kind, opts)
    if drop:
        feats = [f for f in feats if f not in drop]
        if "REF" in drop:
            extra = {**extra, "ref": False}
    cfg = make_cfg(feats, sel, extra, constrained)
    m = FT.fit(trd, cfg)
    r = {"label": label, "stage": stage, "kind": kind, **opts, "drop": "+".join(drop or []), "n_feat": len(feats),
         **val_metrics(m, vad)}
    if cv:
        oof = FT.cross_validate(trd, cfg, k=5)
        r.update({f"cv_{k}": v for k, v in FT.cv_metrics(oof).items()})
        nat = oof[(oof.domain_origin == "natural") & oof.y.notna()]
        r["cv_natural_prot"] = FT.cv_metrics(nat, by_fold=False)["spearman_prot_median"]
        per = []
        for f, g in nat.groupby("fold"):
            per.append(E.evaluate(g.assign(ddg=g.y), "oof")["spearman_prot_median"])
        r["cv_fold_prot"] = json.dumps([round(x, 4) for x in per])
    return r


# ----------------------------------------------------------------------------- stage: select
def stage_select():
    tr, va, _ = train_tables()
    base = {"elec": "dh", "transform": "slog_rep", "pro": "none", "avg": "avg", "drefp": True}
    sel = {"loss": "huber", "ref_l2": 1e-2, "ref": True}
    rows = []

    def batch(cands, stage=1, kind="full"):
        res = Parallel(n_jobs=min(len(cands), 16))(delayed(_run_config)(lab, stage, o, kind, sel, tr, va) for lab, o in cands)
        for r in res:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "cv_fold_prot"}, flush=True)
        rows.extend(res)
        return {r["label"]: r for r in res}

    def pick(res, cands):
        best = max(res, key=lambda k: res[k]["val_natural_prot"] + res[k]["cv_natural_prot"])  # val + CV jointly
        return dict(cands)[best]
    # 1) electrostatics variant x heavy-tail transform (stage 1 and stage 2 separately)
    out = {}
    for stage in (1, 2):
        c = [(f"s{stage}:elec={e}", {**base, "elec": e}) for e in ("dh", "bur", "dd")]
        o = pick(batch(c, stage), c)
        c = [(f"s{stage}:tr={t}", {**o, "transform": t}) for t in ("none", "slog_rep", "slog_all", "clip_rep5", "clip_rep20", "clip_pct")]
        o = pick(batch(c, stage), c)
        c = [(f"s{stage}:pro={p},drefp={dr}", {**o, "pro": p, "drefp": dr}) for p in ("none", "split") for dr in (True, False)]
        o = pick(batch(c, stage), c)
        if stage == 1:
            c = [(f"s1:avg={a}", {**o, "avg": a}) for a in ("avg", "min")]
            batch(c, 1)                      # reported only (Boltzmann vs min ablation); avg kept per D5
        out[stage] = o
    # 2) loss check on the chosen stage-1 setting
    for loss in ("gauss",):
        r = _run_config(f"s1:loss={loss}", 1, out[1], "full", {**sel, "loss": loss}, tr, va)
        rows.append(r)
        print({k: r[k] for k in ("label", "val_natural_prot", "cv_natural_prot")})
    pd.DataFrame(rows).to_csv(OUT / "selection.csv", index=False)
    sel.update({"opts_s1": out[1], "opts_s2": out[2]})
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print("SELECTED", sel)


def load_sel():
    return json.loads((OUT / "selection.json").read_text())


# ----------------------------------------------------------------------------- stage: ablate (train CV + val only)
def stage_ablate():
    tr, va, _ = train_tables()
    sel = load_sel()
    tasks = []
    for stage in (1, 2):
        o = sel[f"opts_s{stage}"]
        feats, _ = feature_set("full", o)
        tasks.append((f"B{stage + 3} full", stage, o, "full", None, True))
        tasks.append((f"B{stage + 3} unconstrained", stage, o, "full", None, False))
        for f in feats + ["REF"]:
            tasks.append((f"B{stage + 3} -{f}", stage, o, "full", [f], True))
        tasks.append((f"B{stage + 3} -energy", stage, o, "full", energy_cols(o), True))
        tasks.append((f"B{stage + 3} -stat", stage, o, "full", STAT, True))
        tasks.append((f"B{stage + 3} -B3+", stage, o, "full", B3P, True))
        for kind in ("noenv", "phys", "dF", "odd", "odd_sym"):
            tasks.append((f"B{stage + 3}-{kind}", stage, o, kind, None, True))
    res = Parallel(n_jobs=24)(delayed(_run_config)(lab, st, o, kind, sel, tr, va, True, drop, con)
                              for lab, st, o, kind, drop, con in tasks)
    t = pd.DataFrame(res)
    for stage in (4, 5):
        full = t[t.label == f"B{stage} full"].iloc[0]
        m = t.label.str.startswith(f"B{stage}")
        for k in ("val_natural_prot", "cv_natural_prot", "cv_spearman_prot_median", "cv_pearson"):
            t.loc[m, f"d_{k}"] = t.loc[m, k] - full[k]
    t.to_csv(OUT / "ablation.csv", index=False)
    print(t[["label", "n_feat", "val_natural_prot", "cv_natural_prot", "d_val_natural_prot", "d_cv_natural_prot",
             "cv_pearson"]].round(4).to_string(index=False))


# ----------------------------------------------------------------------------- stage: fit
EVAL_CAP = "capC"          # cap variant whose columns are merged at evaluation time (selected on train/val)
# name -> (stage, kind, joint[, attenuation]); "+C" = catalytic-CXXC exception (test-informed, D17);
# "af" models carry the pLDDT attenuation and are only meaningful where B-factors are pLDDT (AF/AF2 models).
ATTEN = {"B6gaf": "plddt"}
MODELS = {
    "B6gs1": (1, "b6g", False), "B6g": (2, "b6g", False), "B6gaf": (2, "b6g", False),   # D19 shipped models
    "B6x": (2, "b6", False),                                                          # with site_scsc (transparency)
    "B4-capC": (1, "full", False), "B4-capC-soft": (1, "full", False),
    "B4j+C": (1, "full", True), "B5j+C": (2, "full", True),
    "B4": (1, "full", False), "B4-noenv": (1, "noenv", False), "B4-phys": (1, "phys", False), "B4-dF": (1, "dF", False),
    "B4-odd": (1, "odd", False), "B4-odd+S": (1, "odd_sym", False), "B4j": (1, "full", True),
    "B5": (2, "full", False), "B5-noenv": (2, "noenv", False), "B5-phys": (2, "phys", False), "B5-dF": (2, "dF", False),
    "B5-odd": (2, "odd", False), "B5-odd+S": (2, "odd_sym", False), "B5j": (2, "full", True),
}


def stage_fit(n_boot):
    tr, _, fp = train_tables(EVAL_CAP)
    sel = load_sel()
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    caps_all = {}
    for name, (stage, kind, joint) in MODELS.items():
        o = dict(sel[f"opts_s{stage}"], cxxc=name.endswith("+C"))
        if "capC" in name:
            o.update(cap="capC", rep="soft" if name.endswith("soft") else "hard")
        caps = clip_caps(tr, stage, o) if o["transform"] == "clip_pct" else None
        caps_all[stage] = caps
        d = derive(tr, stage, o, caps)
        feats, extra = feature_set(kind, o)
        cfg = make_cfg(feats, sel, extra)
        if joint:
            d = pd.concat([d, derive(fp, stage, o, caps)], ignore_index=True)
            cfg.ref_nuisance = ("megascale",)
        prov = {"train": "megascale train (Dataset 3 + censored)" + (" + fireprot train" if joint else ""),
                "features": f"data/features/phys_s{stage}_*.parquet + simple_* + extra_sites_* (features_extra)",
                "opts": o, "caps": caps,
                "selection": "results/ladder_physics/selection.csv + b6_selection.csv (Megascale-val natural + train grouped CV)"}
        nb = n_boot if kind in ("full", "noenv", "phys", "b6", "b6g") else 0
        if name in ATTEN:
            m = fit_attenuated(d, cfg, ATTEN[name], n_boot=nb, name=name, provenance=prov)
        else:
            m = FT.fit(d, cfg, n_boot=nb, n_jobs=32, name=name, provenance=prov)
        m["derive_opts"], m["stage"], m["cxxc_exception"] = o, stage, bool(o.get("cxxc"))
        FT.save_model(m, OUT / "models" / f"{name}.json")
        se = m.get("se", {}).get("weights", {})
        print(name, {k: f"{v:.3f}" + (f"±{se[k]:.3f}" if k in se else "") for k, v in m["weights"].items()},
              "conv", m["converged"], flush=True)
        if joint:
            print("  delta_megascale", {k: round(v, 2) for k, v in m["nuisance"]["delta_ref"]["megascale"].items()},
                  m["nuisance"]["scale"], m["nuisance"]["offset"])
    (OUT / "models" / "caps.json").write_text(json.dumps({str(k): v for k, v in caps_all.items()}))


def load_models():
    return {p.stem: FT.load_model(p) for p in sorted((OUT / "models").glob("*.json")) if p.stem != "caps"}


def predict_models(models, df) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    cache = {}
    for name, m in models.items():
        key = (m["stage"], json.dumps(m["derive_opts"], sort_keys=True))
        if key not in cache:
            cache[key] = derive(df, m["stage"], m["derive_opts"], m["provenance"].get("caps"))
        out[name] = predict_attenuated(m, cache[key]) if m.get("scale_kind") else FT.predict(m, cache[key])
    return out


# ----------------------------------------------------------------------------- competitors
COMP_CLIP = (-3.0, 8.0)
COMP = {"FoldX": "results/baselines/foldx_{b}.csv", "tmol-refit": "results/tmol/refit_min_terms_ref_{b}.csv",
        "KORPM": "results/competitors/korpm_{b}.csv", "DDGun3D": "results/competitors/ddgun3d_{b}.csv"}


def attach_competitors(b: pd.DataFrame, bench: str) -> pd.DataFrame:
    for name, pat in COMP.items():
        p = ROOT / pat.format(b=bench)
        if p.exists():
            c = E._norm(pd.read_csv(p, dtype={"pdb_resnum": str}), KEYS)[KEYS + ["ddg_pred"]].drop_duplicates(KEYS)
            b = b.merge(c.rename(columns={"ddg_pred": name}), on=KEYS, how="left")
    if "FoldX" in b:                       # D17: fixed monotone clip, nothing fitted (audit item 5)
        b["FoldX-clip"] = b["FoldX"].clip(-3.0, 8.0)
    p = ROOT / "results" / "baselines_ladder" / f"preds_{bench}.parquet"
    if p.exists():
        c = pd.read_parquet(p)[KS + ["B3+"]]
        c["pdb_resnum"] = E._norm_resnum(c["pdb_resnum"])
        b = b.merge(c.drop_duplicates(KS), on=KS, how="left")
    return b


def strata(b: pd.DataFrame) -> dict:
    apolar, charged = set("AVILMFWC"), set("DEKR")
    st = {"all": np.ones(len(b), bool)}
    if "domain_origin" in b:
        st["natural"], st["designed"] = (b.domain_origin == "natural").to_numpy(), (b.domain_origin == "designed").to_numpy()
    rsa = b["rsa"].to_numpy(float)
    st["buried(rsa<0.25)"], st["exposed(rsa>=0.25)"] = rsa < 0.25, rsa >= 0.25
    st["wt_apolar"], st["wt_charged"] = b.wt.isin(apolar).to_numpy(), b.wt.isin(charged).to_numpy()
    st["wt_polar"] = ~(st["wt_apolar"] | st["wt_charged"])
    st["to_Pro"], st["from_Gly"] = (b.mut == "P").to_numpy(), (b.wt == "G").to_numpy()
    dv = b["dvol"].to_numpy(float)
    st["large->small(dV>30)"], st["small->large(dV<-30)"] = dv > 0.3, dv < -0.3
    st["buried small->large"] = st["buried(rsa<0.25)"] & st["small->large(dV<-30)"]
    pl = b["plddt"].to_numpy(float) if "plddt" in b else (
        b["ctx_bfactor"].to_numpy(float) if "ctx_bfactor" in b else np.full(len(b), np.nan))
    if np.isfinite(pl).any() and np.nanmax(pl) > 1 and ("plddt" in b or bench_is_af(b)):
        for lo, hi in ((0, 50), (50, 70), (70, 90), (90, 101)):
            st[f"pLDDT[{lo},{hi})"] = (pl >= lo) & (pl < hi)
    return st


def bench_is_af(b):
    return "struct_source" in b and (b.struct_source.astype(str).str.contains("af")).mean() > 0.5


def stabilising(df, col, k=5):
    """AUROC for ddg < -0.5 (stabilising) and mean per-protein precision@k of the k most negative predictions."""
    d = df.dropna(subset=[col, "ddg"])
    pos = (d.ddg < -0.5).to_numpy()
    au = E.auroc(-d[col].to_numpy(), pos) if pos.sum() >= 10 else np.nan
    prec = [(g.nsmallest(k, col).ddg < -0.5).mean() for _, g in d.groupby("protein_id") if len(g) >= 20]
    return {"auroc_stab": au, f"prec@{k}_per_protein": float(np.mean(prec)) if prec else np.nan, "base_rate": float(pos.mean())}


SUBSETS = dict(L.SUBSETS)
# D17(2): S461/S669 reported on strict subsets only (S461-all has 24.5% exact Megascale-train duplicate mutations)
SUBSETS["s461"] = {"strict": SUBSETS["s461"]["strict"]}
SUBSETS["s669"] = {"strict": SUBSETS["s669"]["strict"]}
SUBSETS["domainome"] = {"all": "", "plddt70": "plddt>=70", "strict": "homolog_in_megascale_train==0",
                        "flag_kept": "plddt>=70,x_dist_term>=3",
                        "plddt<50": "plddt<50", "plddt50-70": "plddt>=50,plddt<70", "plddt70-90": "plddt>=70,plddt<90",
                        "plddt>=90": "plddt>=90"}
SUBSETS.pop("s2648", None)
METRICS = L.METRICS


def _eval_task(bench, sub, model, df, n_boot, comps, rows_tag="own"):
    """rows_tag: 'own' = every row this model predicts; 'common' = rows predicted by ALL table models (D17)."""
    target = "ddg" if df.ddg.notna().any() else "score"
    r = E.evaluate(df, model, target=target)
    row = {"bench": bench, "subset": sub, "rows": rows_tag, "model": model, **{k: r[k] for k in METRICS}}
    if n_boot:
        ci = E.bootstrap_ci(df, model, target=target, n_boot=n_boot)
        for k in METRICS[2:]:
            if k in ci:
                row[f"{k}_lo"], row[f"{k}_hi"] = ci[k]
    pairs = []
    for c in (comps if rows_tag == "own" else []):
        both = df[df[c].notna() & df[model].notna()] if c in df else df.iloc[:0]
        if len(both) < 30 or c == model:
            continue
        pb = E.paired_bootstrap(both, model, c, target=target, n_boot=n_boot or 300)
        a, b = E.evaluate(both, model, target=target), E.evaluate(both, c, target=target)
        pr = {"bench": bench, "subset": sub, "model": model, "vs": c, "n_both": len(both)}
        for k in ("spearman", "spearman_prot_median", "pearson"):
            if k in pb:
                pr.update({f"{k}_model": a[k], f"{k}_vs": b[k], f"d_{k}": pb[k]["delta"], f"d_{k}_lo": pb[k]["lo"],
                           f"d_{k}_hi": pb[k]["hi"]})
        pairs.append(pr)
    site = {"bench": bench, "subset": sub, "rows": rows_tag, "model": model, **L.site_decomp(df, model, target)}
    return row, pairs, site


def stage_eval(n_boot=1000):
    models = load_models()
    comps = ["FoldX", "FoldX-clip", "B3+", "tmol-refit", "KORPM", "DDGun3D"]
    mains = ["B4", "B5", "B6g", "B6gs1", "B6gaf", "B6x"]
    tasks, cov, strata_rows, stab_rows = [], [], [], []
    for bench, subs in SUBSETS.items():
        b = add_extra(add_phys(L.bench_table(bench), bench, EVAL_CAP), bench)
        b["cxxc"] = cxxc_flags(b)
        P = predict_models(models, b)
        b = attach_competitors(pd.concat([b, P], axis=1), bench)
        cols = [c for c in list(models) + comps if c in b]
        keep = list(dict.fromkeys(KS + ["ddg", "cluster", "domain_origin", "rsa", "dvol", "plddt", "ctx_bfactor",
                                        "struct_source", "score", "direction", "pair_id"] + cols))
        b[[c for c in keep if c in b]].to_parquet(OUT / f"preds_{bench}.parquet", index=False)
        cov.append({"bench": bench, "rows": len(b), **{c: float(b[c].notna().mean()) for c in cols}})
        nb = 200 if bench == "domainome" else n_boot
        # D17(5): models covering < 95% of the benchmark are left out of the common-row intersection and
        # flagged in the report (e.g. DDGun3D runs on 50 of 518 Domainome domains).
        low_cov = {c for c in cols if b[c].notna().mean() < 0.95}
        tab_cols = [c for c in cols if c not in low_cov]
        for sub, expr in subs.items():
            d = E.apply_subset(b, expr)
            common = d[tab_cols].notna().all(axis=1) if tab_cols else pd.Series(False, index=d.index)
            for c in cols:
                cc = [x for x in (comps + mains) if x in d and x != c] if c in mains else []
                use = list(dict.fromkeys(KEYS + ["ddg", "cluster", c] + cc + (["score"] if "score" in d else [])))
                want_boot = nb if (c in mains + comps or sub in ("all", "natural")) else 0
                dd = d[d[c].notna()]
                if len(dd) >= 20:
                    tasks.append((bench, sub, c, dd[use], want_boot, cc, "own"))
                dc = d[common & d[c].notna()]
                if c in tab_cols and len(dc) >= 20 and len(dc) < len(dd):
                    tasks.append((bench, sub, c, dc[use], want_boot, [], "common"))
                elif c in tab_cols and len(dc) >= 20:
                    tasks.append((bench, sub, c, dc[use], 0, [], "common"))
        if bench in ("megascale", "fireprot", "domainome", "abundance_dms"):
            base = E.apply_subset(b, "split==test") if "split" in b and bench in ("megascale", "fireprot") else b
            for sname, mask in strata(base).items():
                g = base[mask]
                for c in cols:
                    gg = g.dropna(subset=[c])
                    if len(gg) < 30:
                        continue
                    r = E.evaluate(gg, c, target="ddg" if gg.ddg.notna().any() else "score")
                    strata_rows.append({"bench": bench, "stratum": sname, "model": c, "n": r["n"], "n_prot": r["n_prot_eval"],
                                        "spearman": r["spearman"], "spearman_prot_median": r["spearman_prot_median"],
                                        "pearson": r["pearson"]})
            if b.ddg.notna().any():
                for sname in ("all", "natural"):
                    if sname == "natural" and "domain_origin" not in base:
                        continue
                    g = base[strata(base)[sname]]
                    for c in cols:
                        stab_rows.append({"bench": bench, "subset": sname, "model": c, **stabilising(g, c)})
    pd.DataFrame(cov).to_csv(OUT / "prediction_coverage.csv", index=False)
    print(pd.DataFrame(cov).round(3).to_string())
    pd.DataFrame(strata_rows).to_csv(OUT / "strata.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(OUT / "stabilising.csv", index=False)
    res = Parallel(n_jobs=40)(delayed(_eval_task)(*t) for t in tasks)
    pd.DataFrame([r[0] for r in res]).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame([p for r in res for p in r[1]]).to_csv(OUT / "paired.csv", index=False)
    pd.DataFrame([r[2] for r in res]).to_csv(OUT / "site_decomposition.csv", index=False)
    paired_af(models)


def paired_af(models):
    p = pd.read_csv(L.PROC / "paired_af.csv", low_memory=False)
    p = p[p.usable.astype(bool) & ~p.af_is_mutant_state.fillna(False).astype(bool)].reset_index(drop=True)
    af = add_extra(add_phys(L.with_feats(p[KS].copy(), "paired_af"), "paired_af", EVAL_CAP), "paired_af")
    xt = add_extra(add_phys(L.with_feats(p.assign(structure=p.xtal_structure, chain=p.xtal_chain,
                                                  pdb_resnum=p.xtal_resnum)[KS], "paired_af_xtal"),
                            "paired_af_xtal", EVAL_CAP), "paired_af_xtal")
    for d in (af, xt):
        d["cxxc"] = cxxc_flags(d)
    pa, px = predict_models(models, af), predict_models(models, xt)
    rows = []
    for ds in ["all"] + sorted(p.dataset.unique()):
        m = np.ones(len(p), bool) if ds == "all" else (p.dataset == ds).to_numpy()
        for c in list(pa.columns) + ["s1_dF", "s2_dF", "s1_dE_lj_rep", "s2_dE_lj_rep"]:
            a = (pa[c] if c in pa else af[c])[m].to_numpy(float)
            x = (px[c] if c in px else xt[c])[m].to_numpy(float)
            y = p.ddg[m].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(x)
            rows.append({"dataset": ds, "quantity": c, "n": int(ok.sum()), "pearson_af_vs_xtal": np.corrcoef(a[ok], x[ok])[0, 1],
                         "spearman_af_vs_xtal": spearmanr(a[ok], x[ok])[0], "mean_abs_diff": float(np.mean(np.abs(a[ok] - x[ok]))),
                         "spearman_af_vs_ddg": spearmanr(a[ok], y[ok])[0], "spearman_xtal_vs_ddg": spearmanr(x[ok], y[ok])[0]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "paired_af_agreement.csv", index=False)
    print(t[t.dataset == "all"].round(3).to_string(index=False))


# ----------------------------------------------------------------------------- stage: antisymmetry (Ssym, diagnostic only)
def stage_antisym():
    models = load_models()
    b = add_extra(add_phys(L.bench_table("ssym"), "ssym", EVAL_CAP), "ssym")
    b["cxxc"] = cxxc_flags(b)
    b = attach_competitors(pd.concat([b, predict_models(models, b)], axis=1), "ssym")
    b["raw_s1_dF"], b["raw_s2_dF"] = b["s1_dF"], b["s2_dF"]
    cols = ["raw_s1_dF", "raw_s2_dF"] + list(models) + [c for c in ("B3+", "KORPM", "DDGun3D") if c in b]
    rows = []
    for c in cols:
        a = E.antisymmetry(b, c)
        # (c) two-structure symmetrisation: p_sym(dir) = (p_dir - p_rev)/2 (exactly antisymmetric by construction)
        w = b.pivot_table(index="pair_id", columns="direction", values=[c, "ddg"], aggfunc="first").dropna()
        psym = (w[(c, "direct")] - w[(c, "reverse")]) / 2
        yd = w[("ddg", "direct")]
        rows.append({"model": c, **{k: a[k] for k in ("n_pairs", "r_pearson", "bias", "pearson_direct", "pearson_reverse")},
                     "pearson_direct_symmetrised": float(np.corrcoef(psym, yd)[0, 1]),
                     "mean_pred_direct": float(w[(c, "direct")].mean()), "mean_pred_reverse": float(w[(c, "reverse")].mean())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "antisymmetry.csv", index=False)
    print(t.round(3).to_string(index=False))
    # adaptation diagnostics: per-term direct + reverse sums (0 if the term were antisymmetric across structures)
    terms = [c for c in b.columns if c.startswith(("s1_dE_", "s2_dE_")) or c in ("s1_dF", "s2_dF", "s1_dTS", "s2_dTS",
                                                                                    "d_env_cn_ss_bb", "d_rama", "bur")]
    rr = []
    for c in terms:
        w = b.pivot_table(index="pair_id", columns="direction", values=c, aggfunc="first").dropna()
        s_ = w["direct"] + w["reverse"]
        rr.append({"term": c, "mean_dir_plus_rev": float(s_.mean()), "median": float(s_.median()),
                   "r_dir_vs_minus_rev": float(np.corrcoef(w["direct"], -w["reverse"])[0, 1]) if w["direct"].std() > 0 else np.nan})
    pd.DataFrame(rr).to_csv(OUT / "antisymmetry_terms.csv", index=False)
    print(pd.DataFrame(rr).round(3).to_string(index=False))


# ----------------------------------------------------------------------------- stage: transfer (G6)
def stage_transfer(n_boot):
    tr, _, fp = train_tables(EVAL_CAP)
    sel = load_sel()
    for name, stage, kind in (("B4", 1, "full"), ("B5", 2, "full"), ("B6gs1", 1, "b6g"), ("B6g", 2, "b6g"),
                              ("B6x", 2, "b6")):
        o = dict(sel[f"opts_s{stage}"], cxxc=False)
        caps = clip_caps(tr, stage, o) if o["transform"] == "clip_pct" else None
        feats, extra = feature_set(kind, o)
        fits = {src: FT.fit(derive(d, stage, o, caps), make_cfg(feats, sel, extra), n_boot=n_boot, n_jobs=32, name=f"{name}_{src}")
                for src, d in (("megascale", tr), ("fireprot", fp))}
        rows = []
        for f in feats:
            wa, wb = fits["megascale"]["weights"][f], fits["fireprot"]["weights"][f]
            sa, sb = fits["megascale"]["se"]["weights"][f], fits["fireprot"]["se"]["weights"][f]
            ratio = wb / wa if wa != 0 else np.nan
            rows.append({"feature": f, "w_megascale": wa, "se_megascale": sa, "w_fireprot": wb, "se_fireprot": sb,
                         "ratio": ratio, "ratio_over_1.7": ratio / 1.7 if np.isfinite(ratio) else np.nan,
                         "zero_in_one": bool((wa == 0) != (wb == 0)),
                         "gt3x_scaled": bool(np.isfinite(ratio) and (abs(ratio / 1.7) > 3 or abs(ratio / 1.7) < 1 / 3)),
                         "z_diff": (wb - wa) / np.sqrt(sa ** 2 + sb ** 2) if sa + sb > 0 else np.nan})
        ra, rb = np.array(list(fits["megascale"]["ref"].values())), np.array(list(fits["fireprot"]["ref"].values()))
        rows.append({"feature": "ref_energy_pearson", "w_fireprot": float(np.corrcoef(ra, rb)[0, 1])})
        for src, d in (("megascale", tr), ("fireprot", fp)):    # sign-free refit: does the data prefer the sign?
            mu = FT.fit(derive(d, stage, o, caps), make_cfg(feats, sel, extra, constrained=False))
            for f in feats:
                rows.append({"feature": f"{f} [unconstrained]", ("w_megascale" if src == "megascale" else "w_fireprot"):
                             mu["weights"][f]})
        t = pd.DataFrame(rows)
        t.to_csv(OUT / f"transfer_{name}.csv", index=False)
        print(f"== transfer {name}\n", t.round(3).to_string(index=False))


# ----------------------------------------------------------------------------- stage: B6 selection (train/val)
def stage_b6():
    """Add the residual-analysis terms one at a time to B5/B4 and check the gains survive the corrected
    N-cap scale. Selection data only: Megascale-val (natural ρ_prot) + grouped 5-fold CV on train."""
    tr, va, _ = train_tables(EVAL_CAP)
    sel = load_sel()
    rows = []
    for stage in (1, 2):
        o = dict(sel[f"opts_s{stage}"], cxxc=False)
        trd, vad = derive(tr, stage, o), derive(va, stage, o)
        base, extra = feature_set("full", o)
        sets = {"base (B4/B5)": base, "+site_scsc": base + ["site_scsc"], "+ncap_pref": base + ["ncap_pref"],
                "+helix_dipole": base + ["helix_dipole"], "+site_hbsc_bur": base + ["site_hbsc_bur"],
                "B6 (4 terms)": base + B6_XTAL, "B6-D15 (state-function variant)": base + B6_D15,
                "B6 minus site_scsc": base + [f for f in B6_XTAL if f != "site_scsc"]}
        for lab, feats in sets.items():
            cfg = make_cfg(feats, sel, extra)
            m = FT.fit(trd, cfg)
            r = {"stage": stage, "model": lab, "n_feat": len(feats), **val_metrics(m, vad)}
            oof = FT.cross_validate(trd, cfg, k=5)
            r["cv_natural_prot"] = FT.cv_metrics(oof[oof.domain_origin == "natural"], by_fold=False)["spearman_prot_median"]
            r["cv_all_prot"] = FT.cv_metrics(oof, by_fold=False)["spearman_prot_median"]
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        # attenuation + its burial-matched control (notes/af_mode.md section 4)
        cfg6 = make_cfg(base + B6_XTAL, sel, extra)
        for kind in ("plddt", "burial"):
            oof = cv_attenuated(trd, cfg6, kind, k=5)
            m = fit_attenuated(trd, cfg6, kind)
            v = vad.assign(ddg=vad.y, pred=predict_attenuated(m, vad))
            vn = v[(v.domain_origin == "natural") & v.ddg.notna()]
            r = {"stage": stage, "model": f"B6 + {kind} attenuation", "n_feat": len(base + B6_XTAL) + 1,
                 "val_natural_prot": E.evaluate(vn, "pred")["spearman_prot_median"],
                 "val_all_prot": E.evaluate(v[v.ddg.notna()], "pred")["spearman_prot_median"],
                 "cv_natural_prot": FT.cv_metrics(oof[oof.domain_origin == "natural"], by_fold=False)["spearman_prot_median"],
                 "cv_all_prot": FT.cv_metrics(oof, by_fold=False)["spearman_prot_median"],
                 "atten_weight": m["weights"][f"scale_{kind}"]}
            rows.append(r)
            print({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in r.items()}, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "b6_selection.csv", index=False)
    print(t.round(4).to_string(index=False))


# ----------------------------------------------------------------------------- stage: final (D19) selection + ablation
GROUPS = {"env potential": ["d_env_cn_ss_bb"], "Rama (site + pre-Pro)": ["d_rama", "d_rama_prev"],
          "vdW (lj_atr, lj_rep)": ["e_lj_atr", "e_lj_rep"], "solvation (apolar, polar)": ["e_solv_apolar", "e_solv_polar"],
          "H-bonds (sc-bb, sc-sc, bb-bb)": ["e_hb_sc_bb", "e_hb_sc_sc", "e_hb_bb_bb"], "electrostatics": ["e_elec_dh"],
          "side-chain entropy + rotamer prior": ["dTS", "dprior_avg"], "all engine energies": None,
          "B3+ remainder (burial, cavity, Pro/Gly/SS)": ["bur", "pro_nhhb", "gly_posphi", "occ_dvol_loss", "ss_broken"],
          "N-cap": ["ncap_pref"], "helix dipole": ["helix_dipole"], "reference energies (Δref)": ["REF"]}


def stage_final():
    """D19: B6g selection numbers (Megascale-val + grouped CV) incl. the attenuation control without site_scsc, and
    a drop-one-group 'what contributes' table. Selection uses val/CV only; the test column is reported afterwards
    and labelled as such (fitted on train, scored on Megascale-test natural, never used for a choice)."""
    tr, va, _ = train_tables(EVAL_CAP)
    sel = load_sel()
    te = add_extra(add_phys(L.bench_table("megascale"), "megascale", EVAL_CAP), "megascale")
    te = te[te.domain_origin == "natural"]
    rows = []

    def score(feats, extra, stage, o, label, atten=None, constrained=True):
        trd, vad, ted = derive(tr, stage, o), derive(va, stage, o), derive(te, stage, o)
        cfg = make_cfg(feats, sel, extra, constrained)
        if atten:
            m = fit_attenuated(trd, cfg, atten)
            oof = cv_attenuated(trd, cfg, atten, k=5)
            pv, pt = predict_attenuated(m, vad), predict_attenuated(m, ted)
        else:
            m = FT.fit(trd, cfg)
            oof = FT.cross_validate(trd, cfg, k=5)
            pv, pt = FT.predict(m, vad), FT.predict(m, ted)
        v = vad.assign(ddg=vad.y, pred=pv)
        vn = v[(v.domain_origin == "natural") & v.ddg.notna()]
        r = {"label": label, "stage": stage, "n_feat": len(feats),
             "val_natural_prot": E.evaluate(vn, "pred")["spearman_prot_median"],
             "cv_natural_prot": FT.cv_metrics(oof[oof.domain_origin == "natural"], by_fold=False)["spearman_prot_median"],
             "test_natural_prot (reported, not used)": E.evaluate(ted.assign(pred=pt).dropna(subset=["pred"]), "pred")["spearman_prot_median"]}
        if atten:
            r["atten_weight"] = m["weights"][f"scale_{atten}"]
        rows.append(r)
        print({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in r.items()}, flush=True)
        return r

    for stage in (1, 2):
        o = dict(sel[f"opts_s{stage}"], cxxc=False)
        base, extra = feature_set("full", o)
        g, _ = feature_set("b6g", o)
        x, _ = feature_set("b6", o)
        score(base, extra, stage, o, f"B{stage + 3} (base)")
        score(g, extra, stage, o, f"B6g{'s1' if stage == 1 else ''}")
        score(x, extra, stage, o, f"B6x{'s1' if stage == 1 else ''} (+site_scsc)")
        score(base + ["site_hbsc_bur", "ncap_pref", "helix_dipole"], extra, stage, o, "B6 minus site_scsc (4-term set − scsc)")
        if stage == 2:
            score(g, extra, stage, o, "B6g + pLDDT attenuation (= B6gaf)", atten="plddt")
            score(g, extra, stage, o, "B6g + burial attenuation (control, sign-constrained)", atten="burial")
            # sign-free burial control
            trd = derive(tr, 2, o)
            b0 = FT.fit(trd, make_cfg(g, sel, extra))
            d = trd.assign(scale_burial=FT.predict(b0, trd) * _atten_col(trd, "burial"))
            mu = FT.fit(d, make_cfg(g + ["scale_burial"], sel, extra, constrained=False))
            rows.append({"label": "burial control, sign-free weight", "stage": 2, "atten_weight": mu["weights"]["scale_burial"]})
            # what contributes: drop one group from B6g
            for gname, cols in GROUPS.items():
                cols = energy_cols(o) if cols is None else cols
                feats = [f for f in g if f not in cols]
                ex = {**extra, "ref": False} if "REF" in cols else extra
                score(feats, ex, 2, o, f"B6g − {gname}")
    t = pd.DataFrame(rows)
    full = t[t.label == "B6g"].iloc[0]
    for k in ("val_natural_prot", "cv_natural_prot", "test_natural_prot (reported, not used)"):
        t[f"d_{k.split(' ')[0]}"] = t[k] - full[k]
    t.to_csv(OUT / "final_selection_ablation.csv", index=False)
    print(t.round(4).to_string(index=False))


# ----------------------------------------------------------------------------- attenuation (AF mode) fits
def _atten_col(d: pd.DataFrame, kind: str) -> np.ndarray:
    """Attenuation strength f in [0,1]. 'plddt': 1 - pLDDT/100 (AF models only; crystal/design -> 0).
    'burial': 1 - RSA, the burial-matched control the AF study asks for (notes/af_mode.md section 4)."""
    if kind == "plddt":
        return d["flag_lowplddt"].to_numpy(float)
    return (1.0 - d["x_rsa"].clip(0, 1)).to_numpy(float)


def fit_attenuated(train: pd.DataFrame, cfg: FT.FitConfig, kind: str, n_boot: int = 0, name: str = "B6af",
                   provenance: dict | None = None) -> dict:
    """Two-stage fit: core model, then core x f added as one extra (negatively constrained) weight, so the
    prediction is core*(1 - a*f) + ... . The core model is stored in the JSON as `scale_base` so that inference
    can recompute it (schema note for E1 in notes/engine_mutate.md)."""
    base = FT.fit(train, cfg)
    col = f"scale_{kind}"
    d = train.assign(**{col: FT.predict(base, train) * _atten_col(train, kind)})
    cfg2 = FT.FitConfig(**{**{k: v for k, v in cfg.__dict__.items()}, "features": list(cfg.features) + [col],
                           "signs": {**cfg.signs, col: -1}})
    m = FT.fit(d, cfg2, n_boot=n_boot, n_jobs=32, name=name, provenance=provenance)
    m["scale_base"], m["scale_kind"] = base, kind
    return m


def predict_attenuated(model: dict, d: pd.DataFrame) -> np.ndarray:
    col = f"scale_{model['scale_kind']}"
    d = d.assign(**{col: FT.predict(model["scale_base"], d) * _atten_col(d, model["scale_kind"])})
    return FT.predict(model, d)


def cv_attenuated(train: pd.DataFrame, cfg: FT.FitConfig, kind: str, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Grouped-CV OOF predictions for the two-stage model (the core is refitted per fold: no leakage)."""
    folds = FT.group_folds(train["cluster"], k, seed)
    out = train.copy()
    out["oof"], out["fold"] = np.nan, folds
    for j in range(k):
        m = fit_attenuated(train[folds != j], cfg, kind)
        out.loc[folds == j, "oof"] = predict_attenuated(m, train[folds == j])
    return out


# ----------------------------------------------------------------------------- stage: cap selection (train/val only)
def stage_caps():
    """Select E1's D9 repulsion cap (capA c0=1 | capB c0=3 = original | capC c0=1,c1=2 burial-dependent) and the
    repulsion form (hard cap + signed-log transform vs E1's log-softened `lj_rep_soft`) by Megascale-val +
    grouped CV on train. Never by test, and never by Ssym alone (Ssym duplicates FireProt-trainval)."""
    sel = load_sel()
    rows = []
    for cap in ("capA", "capB", "capC"):
        tr, va, _ = train_tables(cap)
        for form in ("hard", "soft"):
            o = dict(sel["opts_s1"], cap=cap, rep=form, cxxc=False)
            if form == "soft" and f"s1_dE_lj_rep_soft_{cap}" not in tr:
                continue
            r = _run_config(f"B4[{cap},{form}]", 1, o, "full", sel, tr, va)
            r["cap"], r["rep_form"] = cap, form
            # heavy-tail and antisymmetry diagnostics of the repulsion column actually used
            col = f"s1_dE_lj_rep_soft_{cap}" if form == "soft" else f"s1_dE_lj_rep_{cap}"
            r["rep_p99"], r["rep_max"] = float(np.nanpercentile(tr[col], 99)), float(np.nanmax(tr[col]))
            rows.append({k: v for k, v in r.items() if k != "cv_fold_prot"})
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cap_selection.csv", index=False)
    best = t.loc[(t.val_natural_prot + t.cv_natural_prot).idxmax()]
    print("BEST", best[["label", "cap", "rep_form", "val_natural_prot", "cv_natural_prot"]].to_dict())
    sel["cap"], sel["rep_form"] = best["cap"], best["rep_form"]
    (OUT / "selection_cap.json").write_text(json.dumps({k: sel[k] for k in ("cap", "rep_form")}, indent=1))
    # Ssym antisymmetry of the raw repulsion column per variant (diagnostic, not a selection criterion)
    ss = add_phys(L.bench_table("ssym"), "ssym", "")
    diag = []
    for cap in ("capA", "capB", "capC"):
        s2 = add_phys(L.bench_table("ssym"), "ssym", cap)
        for col in (f"s1_dE_lj_rep_{cap}", f"s1_dE_lj_rep_soft_{cap}", f"s1_dF_{cap}"):
            if col not in s2:
                continue
            w = s2.pivot_table(index="pair_id", columns="direction", values=col, aggfunc="first").dropna()
            diag.append({"cap": cap, "column": col, "mean_dir_plus_rev": float((w["direct"] + w["reverse"]).mean()),
                         "r_dir_vs_minus_rev": float(np.corrcoef(w["direct"], -w["reverse"])[0, 1])})
    pd.DataFrame(diag).to_csv(OUT / "cap_antisymmetry.csv", index=False)
    print(pd.DataFrame(diag).round(3).to_string(index=False))


# ----------------------------------------------------------------------------- stage: G2 stratum check (no test)
def stage_g2check():
    """B5 - B4 on the buried small->large stratum using (a) grouped-CV out-of-fold predictions on
    Megascale-train and (b) Megascale-val. The test-set version of this comparison is a test look (audit item 9)."""
    tr, va, _ = train_tables()
    sel = load_sel()
    rows = []
    oof = {}
    for name, stage in (("B4", 1), ("B5", 2)):
        o = dict(sel[f"opts_s{stage}"], cxxc=False)
        feats, extra = feature_set("full", o)
        cfg = make_cfg(feats, sel, extra)
        d = derive(tr, stage, o)
        oof[name] = FT.cross_validate(d, cfg, k=5)
        m = FT.fit(d, cfg)
        v = derive(va, stage, o)
        va[f"pred_{name}"] = FT.predict(m, v)
    for src, frames in (("train_cv_oof", {k: v.assign(pred=v.oof, ddg=v.y) for k, v in oof.items()}),
                        ("val", {k: va.assign(pred=va[f"pred_{k}"], ddg=va.y) for k in ("B4", "B5")})):
        for sname in ("all", "buried(rsa<0.25)", "small->large(dV<-30)", "buried small->large"):
            out = {"source": src, "stratum": sname}
            for k, f in frames.items():
                g = f[strata(f)[sname] & f.ddg.notna() & f.pred.notna()]
                r = E.evaluate(g, "pred")
                out[f"{k}_n"], out[f"{k}_spearman"], out[f"{k}_prot"] = r["n"], r["spearman"], r["spearman_prot_median"]
            out["d_spearman"] = out["B5_spearman"] - out["B4_spearman"]
            out["d_prot"] = out["B5_prot"] - out["B4_prot"]
            if src == "train_cv_oof":                      # per-fold sign test on the stratum
                per = []
                for fold in sorted(oof["B4"].fold.unique()):
                    gs = {k: f[(f.fold == fold) & strata(f)[sname] & f.ddg.notna()] for k, f in frames.items()}
                    if min(len(g) for g in gs.values()) >= 30:
                        per.append(E.evaluate(gs["B5"], "pred")["spearman"] - E.evaluate(gs["B4"], "pred")["spearman"])
                out["folds_B5_better"] = f"{sum(x > 0 for x in per)}/{len(per)}"
                out["fold_deltas"] = json.dumps([round(x, 4) for x in per])
            rows.append(out)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "g2_stratum_check.csv", index=False)
    print(t.round(4).to_string(index=False))


# ----------------------------------------------------------------------------- stage: report
def stage_report():
    """D17 reporting protocol: identical rows, strict-only S461/S669, Ssym as a diagnostic table only,
    FoldX raw and clipped, Megascale on natural / natural-strict / all, CIs on every headline cell."""
    met = pd.read_csv(OUT / "metrics.csv")
    cov = pd.read_csv(OUT / "prediction_coverage.csv").set_index("bench")
    com = met[met.rows == "common"] if "rows" in met else met
    own = met[met.rows == "own"] if "rows" in met else met
    site = pd.read_csv(OUT / "site_decomposition.csv")
    cols = [("megascale", "natural", "spearman_prot_median", "MS nat ρ_prot", True),
            ("megascale", "natural_strict", "spearman_prot_median", "MS nat-strict ρ_prot", True),
            ("megascale", "all", "spearman_prot_median", "MS all ρ_prot", True),
            ("megascale", "all", "pearson", "MS all r", False), ("megascale", "all", "rmse", "MS RMSE", False),
            ("fireprot", "all", "pearson", "FP r", True), ("fireprot", "all", "spearman", "FP ρ", False),
            ("s669", "strict", "pearson", "S669-strict r", True), ("s461", "strict", "pearson", "S461-strict r", True),
            ("domainome", "all", "spearman_prot_median", "Dom ρ_dom", True),
            ("abundance_dms", "all", "spearman_prot_median", "Abund ρ", False)]
    order = ["B3+", "B4", "B5", "B6gs1", "B6g", "B6gaf", "B6x",
             "B4-capC", "B4-capC-soft", "B4-noenv", "B4-phys", "B4-dF", "B4-odd", "B4j", "B4j+C",
             "B5-noenv", "B5-phys",
             "B5-dF", "B5-odd", "B5j", "B5j+C", "FoldX", "FoldX-clip", "tmol-refit", "KORPM", "DDGun3D"]
    rows = []
    for mod in order:
        r = {"model": mod}
        for b, sub, k, lab, with_ci in cols:
            x = com[(com.bench == b) & (com.subset == sub) & (com.model == mod)]
            c = cov.at[b, mod] if (b in cov.index and mod in cov.columns) else np.nan
            if not len(x) or not np.isfinite(x[k].iloc[0]):
                r[lab] = "–" if not (np.isfinite(c) and c < 0.95) else "(subset)"
                continue
            v = f"{x[k].iloc[0]:.3f}"
            if with_ci and f"{k}_lo" in x and np.isfinite(x[f"{k}_lo"].iloc[0]):
                v += f" [{x[f'{k}_lo'].iloc[0]:.2f},{x[f'{k}_hi'].iloc[0]:.2f}]"
            r[lab] = v
        x = site[(site.bench == "megascale") & (site.subset == "natural") & (site.model == mod)
                 & (site.get("rows", "own") == "common")]
        r["MS nat site/within"] = f"{x.site_spearman.iloc[0]:.2f}/{x.within_site_mean_spearman.iloc[0]:.2f}" if len(x) else "–"
        rows.append(r)
    t = pd.DataFrame(rows)
    md = ["## Main table (D17: identical rows; strict-only S461/S669; Ssym excluded — diagnostic table below)", ""]
    md.append("| " + " | ".join(t.columns) + " |")
    md.append("|" + "---|" * len(t.columns))
    md += ["| " + " | ".join(map(str, v)) + " |" for v in t.to_numpy()]
    # n per cell
    ns = {lab: int(com[(com.bench == b) & (com.subset == sub) & (com.model == "B4")].n.iloc[0])
          for b, sub, k, lab, _ in cols if len(com[(com.bench == b) & (com.subset == sub) & (com.model == "B4")])}
    md += ["", "Rows per column (identical across models): " + ", ".join(f"{k} {v:,}" for k, v in ns.items()), ""]
    # FireProt on ALL rows: the identical-row intersection excludes the two 2TRX disulfide rows (tmol and FoldX
    # fail on them), which is exactly where the test-informed CXXC exception acts. Shown explicitly (audit item 1).
    fp = own[(own.bench == "fireprot") & (own.subset == "all")]
    md += ["## FireProt-test on all 347 rows (not identical rows: FoldX covers 341, tmol-refit 344)", "",
           "The 2 rows that differ between the `j` and `j+C` models (2TRX C32A/C35A, catalytic CGPC disulfide,",
           "measured ddg ≈ −4) are missing from the identical-row set, so the column above cannot show the effect",
           "of the test-informed CXXC exception. Here it is:", "",
           "| model | n | Pearson | Spearman | ρ_prot (9 proteins ≥10 variants) | RMSE |", "|---|---|---|---|---|---|"]
    for mod in ["B4", "B5", "B6gs1", "B6g", "B6gaf", "B6x", "B4j", "B4j+C", "B5j", "B5j+C", "B3+", "FoldX", "FoldX-clip",
                "tmol-refit", "KORPM", "DDGun3D"]:
        x = fp[fp.model == mod]
        if len(x):
            x = x.iloc[0]
            md.append(f"| {mod} | {int(x.n)} | {x.pearson:.3f} | {x.spearman:.3f} | {x.spearman_prot_median:.3f} | {x.rmse:.2f} |")
    md += ["", "B4j 0.389 vs B4j+C 0.479 and B5j 0.396 vs B5j+C 0.484: the whole joint-model advantage on FireProt-test",
           "Pearson came from those two rows. The headline joint models (no `+C`) are the honest ones.", ""]
    # Ssym diagnostic
    anti = pd.read_csv(OUT / "antisymmetry.csv") if (OUT / "antisymmetry.csv").exists() else None
    if anti is not None:
        md += ["## Ssym — antisymmetry diagnostic only (85.7% of direct rows duplicate FireProt-trainval;",
               "never a test for any model fitted on FireProt: B4j/B5j/tmol-refit/KORPM/DDGun3D)", ""]
        a = anti[anti.model.isin(["raw_s1_dF", "raw_s2_dF", "B4", "B5", "B4-odd", "B5-odd", "B3+", "KORPM", "DDGun3D"])]
        md.append("| model | r(dir,−rev) | bias | Pearson direct | Pearson reverse | Pearson direct, 2-structure sym |")
        md.append("|---|---|---|---|---|---|")
        for _, x in a.iterrows():
            md.append(f"| {x.model} | {x.r_pearson:.2f} | {x.bias:+.2f} | {x.pearson_direct:.3f} | "
                      f"{x.pearson_reverse:.3f} | {x.pearson_direct_symmetrised:.3f} |")
        md += ["", "The last column is ½[pred(direct on the WT structure) − pred(reverse on the mutant structure)]:",
               "it needs BOTH experimental structures and is antisymmetric by construction, so it is not available",
               "at inference on a single structure. The other columns are single-structure scores.", ""]
        md.append("")
    (OUT / "main_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    models = load_models()
    w = []
    for name in ("B4", "B5", "B4-phys", "B5-phys", "B4j", "B5j"):
        m = models[name]
        se = m.get("se", {}).get("weights", {})
        for f, v in m["weights"].items():
            w.append({"model": name, "feature": f, "weight": v, "se": se.get(f, np.nan)})
        w.append({"model": name, "feature": "intercept", "weight": m["intercept"], "se": m.get("se", {}).get("intercept", np.nan)})
    pd.DataFrame(w).to_csv(OUT / "weights.csv", index=False)
    print(pd.DataFrame(w).pivot_table(index="feature", columns="model", values="weight", sort=False).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["select", "ablate", "fit", "eval", "antisym", "transfer", "g2check", "caps", "b6", "final", "report", "all"])
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    st = a.stage
    if st in ("select", "all"):
        stage_select()
    if st in ("ablate", "all"):
        stage_ablate()
    if st in ("fit", "all"):
        stage_fit(a.n_boot)
    if st in ("eval", "all"):
        stage_eval()
    if st in ("antisym", "all"):
        stage_antisym()
    if st in ("transfer", "all"):
        stage_transfer(a.n_boot)
    if st in ("g2check", "all"):
        stage_g2check()
    if st in ("caps",):
        stage_caps()
    if st in ("b6",):
        stage_b6()
    if st in ("final",):
        stage_final()
    if st in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    main()
