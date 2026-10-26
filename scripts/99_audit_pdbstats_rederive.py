"""Audit (read-only w.r.t. the project): re-derive rotamer library + PDB statistics from the PISCES 'train' chains
MINUS the chains homologous (>=30% id, >=50% of shorter seq) to any test/external benchmark protein
(exclude list from scripts/99_audit_pdbstats_leak.py), and estimate the effect on the statistical features and on
B4/B5 predictions. Never overwrites physddg/data/*.npz; writes OUTDIR/{rotlib_bbdep,pdbstats}_clean.npz.

Effect estimate: d_env_cn_ss_bb and d_rama are recomputed for Megascale-test / FireProt-test / Domainome rows from the
stored context (ctx_ss3, ctx_cn, ctx_phi, ctx_psi) with the original and the clean tables; the prediction shift is
w_env*delta(d_env_cn_ss_bb) + w_rama*delta(d_rama) with the fitted B4/B5 weights (weights NOT refitted; engine energies
that depend on the rotamer library are NOT recomputed - the library change is summarised by table differences).

Usage: PYTHONPATH=. .venv/bin/python scripts/99_audit_pdbstats_rederive.py OUTDIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from physddg import rotlib as RL  # noqa: E402
from physddg import topology as T  # noqa: E402

REC = ROOT / "data/cache/pdbstats_records.parquet"
KT = 0.593


def build(out: Path):
    excl = set((out / "exclude_chains.txt").read_text().split())
    df = pd.read_parquet(REC)
    tr = df[df.split == "train"]
    uid = tr.pdb.str.upper() + "_" + tr.chain
    tr_c = tr[~uid.isin(excl)]
    print(f"train chains {uid.nunique()} -> clean {tr_c[['pdb', 'chain']].drop_duplicates().shape[0]}; "
          f"residues {len(tr)} -> {len(tr_c)}")
    qb = lambda d: (~d.altloc) & (d.b_bb < 40) & (d.occ_min >= 0.99) & d.phi.notna() & d.psi.notna() & (~d.modified)
    qs = lambda d: qb(d) & (~d.missing_sc) & (d.b_sc < 40)
    lib = RL.derive_rotlib(tr_c[qs(tr_c)])
    np.savez_compressed(out / "rotlib_bbdep_clean.npz", **lib)
    st = RL.derive_pdbstats(tr_c[qb(tr_c)], tr_c[qs(tr_c)], lib)
    np.savez_compressed(out / "pdbstats_clean.npz", **st)


def compare_tables(out: Path):
    a, b = np.load(ROOT / "physddg/data/pdbstats.npz"), np.load(out / "pdbstats_clean.npz")
    for k in ("env_cn_ss_bb_energy", "env_cn_ss_energy", "rama_energy", "sc_entropy_rotameric", "cn_edges"):
        x, y = a[k].astype(float), b[k].astype(float)
        f = np.isfinite(x) & np.isfinite(y)
        print(f"{k:24s} max|d| {np.abs(x - y)[f].max():.4f}  mean|d| {np.abs(x - y)[f].mean():.5f} (kT / nats)")
    la, lb = np.load(ROOT / "physddg/data/rotlib_bbdep.npz"), np.load(out / "rotlib_bbdep_clean.npz")
    worst = []
    for aa in T.AA1:
        k = f"{aa}_logp"
        if k in la.files and k in lb.files and la[k].shape == lb[k].shape:
            pa, pb = np.exp(la[k].astype(float)), np.exp(lb[k].astype(float))
            worst.append((aa, float(np.abs(pa - pb).max()), float(np.abs(pa - pb).mean())))
        elif k in la.files:
            worst.append((aa, np.nan, np.nan))
    print("rotlib max|dP(rot|phi,psi)| per aa:", {a: round(m, 4) for a, m, _ in worst})
    print("rotlib mean|dP| over all:", round(float(np.nanmean([w[2] for w in worst])), 5))


def feats(stats: RL.PDBStats, d: pd.DataFrame):
    E = stats.env_energy(d.ctx_ss3.to_numpy(), cn=d.ctx_cn.to_numpy(), phi=d.ctx_phi.to_numpy(),
                         psi=d.ctx_psi.to_numpy(), variant="cn_ss_bb")
    iw, im = d.wt.map(T.AA_INDEX).to_numpy(), d.mut.map(T.AA_INDEX).to_numpy()
    r = np.arange(len(d))
    env = KT * (E[r, im] - E[r, iw])                 # stored features are kcal/mol (mutate.stat_features)
    R = np.full((len(d), 20), np.nan)
    ok = np.isfinite(d.ctx_phi.to_numpy()) & np.isfinite(d.ctx_psi.to_numpy())
    if ok.any():
        R[ok] = stats.rama_energy_all_aa(d.ctx_phi.to_numpy()[ok], d.ctx_psi.to_numpy()[ok])
    return env, KT * (R[r, im] - R[r, iw])


def effect(out: Path):
    s0, s1 = RL.PDBStats(), RL.PDBStats(out / "pdbstats_clean.npz")
    W = {m: json.loads(Path(ROOT / f"results/ladder_physics/models/{m}.json").read_text())["weights"] for m in ("B4", "B5")}
    for bench, sub in (("megascale", "megascale"), ("fireprot", "fireprot"), ("domainome", "domainome")):
        f = pd.read_parquet(ROOT / f"data/features/phys_s1_{sub}.parquet")
        p = pd.read_parquet(ROOT / f"results/ladder_physics/preds_{bench}.parquet")
        f["pdb_resnum"], p["pdb_resnum"] = f.pdb_resnum.astype(str), p.pdb_resnum.astype(str)
        keys = ["protein_id", "chain", "pdb_resnum", "wt", "mut", "structure"]
        d = p.merge(f[keys + ["ctx_ss3", "ctx_cn", "ctx_phi", "ctx_psi", "d_env_cn_ss_bb", "d_rama"]].drop_duplicates(keys),
                    on=keys, how="left")
        e0, r0 = feats(s0, d)
        e1, r1 = feats(s1, d)
        chk = np.nanmax(np.abs(e0 - d.d_env_cn_ss_bb.to_numpy()))  # must be ~0
        y = d.ddg if d.ddg.notna().any() else -d.score
        msg = [f"{bench}: reproduce stored d_env max|d| {chk:.2e};",
               f"|d env| mean {np.nanmean(np.abs(e1 - e0)):.4f} max {np.nanmax(np.abs(e1 - e0)):.3f} kcal/mol;",
               f"|d rama| mean {np.nanmean(np.abs(r1 - r0)):.4f} max {np.nanmax(np.abs(r1 - r0)):.3f} kcal/mol (rama w/o pre-Pro map)"]
        print(" ".join(msg))
        for m in ("B4", "B5"):
            shift = W[m]["d_env_cn_ss_bb"] * (e1 - e0) + W[m]["d_rama"] * np.nan_to_num(r1 - r0)
            new = d[m].to_numpy() + shift
            sub_ = (d.domain_origin == "natural").to_numpy() if "domain_origin" in d else np.ones(len(d), bool)

            def pm(col):
                g = pd.DataFrame({"p": col[sub_], "y": y.to_numpy()[sub_], "g": d.protein_id.to_numpy()[sub_]})
                return np.median([spearmanr(x.p, x.y)[0] for _, x in g.groupby("g") if len(x) >= 10])
            print(f"   {m}: mean|shift| {np.mean(np.abs(shift)):.4f} kcal/mol; per-protein median rho "
                  f"{'(natural) ' if 'domain_origin' in d else ''}{pm(d[m].to_numpy()):.4f} -> {pm(new):.4f}")


if __name__ == "__main__":
    o = Path(sys.argv[1])
    if not (o / "pdbstats_clean.npz").exists():
        build(o)
    compare_tables(o)
    effect(o)
