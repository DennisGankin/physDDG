#!/usr/bin/env python
"""AF-mode Q2 (model version / construct) and the G5 global-repacking ablation.

  PYTHONPATH=. .venv/bin/python scripts/85_af_models_crops.py

  dom_v4_v6.csv     AF v4 vs v6 construct crops: feature/prediction agreement and accuracy on identical rows
  dom_crop_full.csv construct crop vs full-length AF model: burial change and accuracy, by construct pLDDT
  repack_ablation.csv  paired xtal/AF agreement and accuracy with native side chains vs after a global repack
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.af_common as C  # noqa: E402
from physddg import fit as FT  # noqa: E402

AFD = C.FEAT / "af"


def p1():
    return FT.load_model(C.OUT / "models" / "P1.json")


def agree(a, b, g=None) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    out = dict(n=int(ok.sum()), r=pearsonr(a[ok], b[ok])[0] if ok.sum() > 10 else np.nan,
               rho=spearmanr(a[ok], b[ok])[0] if ok.sum() > 10 else np.nan,
               mad=float(np.mean(np.abs(a[ok] - b[ok]))))
    if g is not None:
        per = [pearsonr(x.a, x.b)[0] for _, x in pd.DataFrame({"a": a, "b": b, "g": g}).dropna().groupby("g")
               if len(x) >= 10 and x.a.nunique() > 1 and x.b.nunique() > 1]
        out["r_prot_median"] = float(np.median(per)) if per else np.nan
        out["frac_prot_ge_09"] = float(np.mean(np.array(per) >= 0.9)) if per else np.nan
    return out


def main():
    out = C.ensure_out()
    m = p1()
    base = C.domainome(with_feats=False)

    # ---------------- v4 vs v6
    rows = []
    f4 = C.phys("domainome", 1)
    pv6 = AFD / "phys_s1_domainome_v6_off.parquet"
    if pv6.exists():
        f6 = C.load_feat(pv6).drop(columns=["structure"])
        b4 = C.attach(base, f4)
        b6 = C.attach(base[C.KEYS], f6, on_structure=False)
        d = base.copy()
        d["v4"] = FT.predict(m, b4)
        d["v6"] = FT.predict(m, b6)
        d["dF4"], d["dF6"] = b4.s1_dF.to_numpy(), b6.s1_dF.to_numpy()
        d["rsa4"], d["rsa6"] = b4.ctx_rsa.to_numpy(), b6.ctx_rsa.to_numpy()
        d = d[d.v6.notna() & d.v4.notna()]
        band = pd.cut(d.construct_plddt_mean, [0, 70, 80, 90, 101], labels=["<70", "70-80", "80-90", ">90"],
                      right=False)
        for key, s in [("all", d)] + [(f"construct pLDDT {b}", d[band == b]) for b in ["<70", "70-80", "80-90", ">90"]]:
            if len(s) < 100:
                continue
            for q, (a_, b_) in {"P1": ("v4", "v6"), "s1_dF": ("dF4", "dF6"), "rsa": ("rsa4", "rsa6")}.items():
                r = agree(s[a_], s[b_], s.protein_id)
                r.update(subset=key, quantity=q,
                         rho_v4=C.median_rho(s, a_, "mscore"), rho_v6=C.median_rho(s, b_, "mscore"))
                rows.append(r)
        pd.DataFrame(rows).to_csv(out / "dom_v4_v6.csv", index=False)
        print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---------------- construct crop vs full-length model
    pfull = AFD / "phys_s1_domainome_full_off.parquet"
    if pfull.exists():
        ff = C.load_feat(pfull)
        d = base.copy()
        d["pdb_resnum_full"] = C.norm_resnum(d.af_full_resnum)
        key = d[["protein_id", "chain", "pdb_resnum_full", "wt", "mut"]].rename(
            columns={"pdb_resnum_full": "pdb_resnum"})
        bf = C.attach(key, ff.drop(columns=["structure"]), on_structure=False)
        bc = C.attach(base, f4)
        d["P1_crop"], d["P1_full"] = FT.predict(m, bc), FT.predict(m, bf)
        d["rsa_crop"], d["rsa_full"] = bc.ctx_rsa.to_numpy(), bf.ctx_rsa.to_numpy()
        d["cn_crop"], d["cn_full"] = bc.ctx_cn.to_numpy(), bf.ctx_cn.to_numpy()
        d["dF_crop"], d["dF_full"] = bc.s1_dF.to_numpy(), bf.s1_dF.to_numpy()
        d = d[d.P1_full.notna() & d.P1_crop.notna()]
        rows = []
        pb = C.plddt_band(d.plddt)
        for key_, s in [("all", d)] + [(f"site pLDDT {b}", d[pb == b]) for b in C.PLDDT_LABELS]:
            if len(s) < 200:
                continue
            for q, (a_, b_) in {"P1": ("P1_crop", "P1_full"), "s1_dF": ("dF_crop", "dF_full"),
                                "rsa": ("rsa_crop", "rsa_full"), "cn": ("cn_crop", "cn_full")}.items():
                r = agree(s[a_], s[b_], s.protein_id)
                r.update(subset=key_, quantity=q, mean_crop=float(s[a_].mean()), mean_full=float(s[b_].mean()),
                         rho_crop=C.median_rho(s, a_, "mscore"), rho_full=C.median_rho(s, b_, "mscore"))
                rows.append(r)
        pd.DataFrame(rows).to_csv(out / "dom_crop_full.csv", index=False)
        print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---------------- global repack ablation on the paired set
    pr = {s: AFD / f"phys_s1_{s}_repack_all.parquet" for s in ("paired_af", "paired_af_xtal")}
    if all(p.exists() for p in pr.values()):
        p = pd.read_csv(C.PROC / "paired_af.csv", low_memory=False)
        p = p[p.usable.astype(bool) & ~p.af_is_mutant_state.fillna(False).astype(bool)].reset_index(drop=True)
        p["pdb_resnum"] = C.norm_resnum(p.pdb_resnum)
        p["xtal_resnum"] = C.norm_resnum(p.xtal_resnum)
        sides = {"af": p[C.KEYS].copy(),
                 "xtal": p.assign(chain=p.xtal_chain, pdb_resnum=p.xtal_resnum)[C.KEYS]}
        for side, rows_ in sides.items():
            for tag, path in (("native", None), ("repack", pr[f"paired_af{'' if side == 'af' else '_xtal'}"])):
                f = C.phys(f"paired_af{'' if side == 'af' else '_xtal'}", 1) if path is None \
                    else C.load_feat(path)
                t = C.attach(rows_, f.drop(columns=["structure"]), on_structure=False)
                p[f"{side}_{tag}_P1"] = FT.predict(m, t)
                p[f"{side}_{tag}_dF"] = t.s1_dF.to_numpy()
        cols = [f"{side}_{tag}_{q}" for side in ("af", "xtal") for tag in ("native", "repack")
                for q in ("P1", "dF")]
        p = p[p[cols].notna().all(axis=1)].reset_index(drop=True)   # identical rows for both protocols
        rows = []
        for q in ("P1", "dF"):
            for tag in ("native", "repack"):
                r = agree(p[f"af_{tag}_{q}"], p[f"xtal_{tag}_{q}"], p.protein_id)
                d = p[[f"af_{tag}_{q}", f"xtal_{tag}_{q}", "ddg"]].dropna()
                r.update(quantity=q, protocol=tag,
                         rho_af_vs_ddg=spearmanr(d[f"af_{tag}_{q}"], d.ddg)[0],
                         rho_xtal_vs_ddg=spearmanr(d[f"xtal_{tag}_{q}"], d.ddg)[0])
                rows.append(r)
        pd.DataFrame(rows).to_csv(out / "repack_ablation.csv", index=False)
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
