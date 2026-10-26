#!/usr/bin/env python
"""Build the external low-throughput benchmark tables: s669, s461, ssym (direct + reverse), s2648.

Sources (downloaded by scripts/10_download_ht.py into data/raw/):
  korpm/S669.txt        S669 (Pancotti et al. 2022, Brief Bioinf), as redistributed by the S461 authors (KORPM repo)
  korpm/S461.txt        S461 (Martin-Hernandez et al. 2023, Bioinformatics; curated subset of S669)
  korpm/KORPM_Ssym.txt  Ssym (Pucci et al. 2018): WT pdb, mutant pdb, direct mut, reverse mut, exp ddG(direct), ...
  thermonet/s_sym.txt   Ssym as distributed with ThermoNet (cross-check of values/sign only)
  prostata/S2648.csv    S2648 (Dehouck et al. 2009, PoPMuSiC training set) with chain, pH, T

Sign: all four files use ddG < 0 = destabilising (Fariselli/Pancotti/KORPM/PROSTATA convention; mean ~ -1 and
buried hydrophobic losses negative, checked below) -> flipped: ddg = -ddG_orig. For Ssym reverse: ddg_rev = -ddg_dir.

Structures: local wwPDB mirror (/rsyncPDB, obsolete dir as fallback), first model, the mutated chain only, waters/
ligands/H removed, first altloc, MSE->MET etc. -> data/structures/xtal/pdb/<PDB>_<chain>.pdb (numbering + icodes kept).
Mapping (PI decision, 10-26): author residue number + WT letter must match exactly (100% WT identity); mismatches are
dropped and counted, never rescued by global offsets. (Optional --allow-offset re-enables a documented per-protein
offset / sequential-index search; e.g. it would rescue the Kabat-numbered antibody 1LVE in S2648. Not used.)
Chain/assembly: deposited asymmetric unit, model 1, mutated chain only; near_hetatm / near_other_chain = any heavy
atom of the site within 5 A of a non-water HETATM / another protein chain in the ORIGINAL entry.
"""
import sys
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ht_common import (ROOT, XTAL_DIR, cb_burial, chain_residues, clean_chain, mirror_path, read_structure,  # noqa: E402
                       sign_report, site_context)

warnings.filterwarnings("ignore")
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
SDIR = XTAL_DIR / "pdb"
MUT_RE = r"^([A-Z])([A-Za-z0-9])(-?\d+)([A-Z])$"
ALLOW_OFFSET = "--allow-offset" in sys.argv  # PI decision: default = exact author numbering only, mismatches dropped


def log(*a):
    print(*a, flush=True)


def parse_mut(s):
    m = pd.Series(s).str.extract(MUT_RE)
    assert m.notna().all().all(), s[m.isna().any(axis=1)].head()
    return m[0].values, m[1].values, m[2].values, m[3].values


def load_s669(path, name):
    d = pd.read_csv(path, sep=r"\s+", header=None)
    d = d.iloc[:, :3]
    d.columns = ["pdb", "mutstr", "ddg_orig"]
    d["wt"], d["chain"], d["resnum"], d["mut"] = parse_mut(d.mutstr.values)
    d["dataset"] = name
    d["direction"] = "direct"
    d["pair_id"] = [f"{name}_{i}" for i in range(len(d))]
    return d


def load_ssym():
    k = pd.read_csv(RAW / "korpm/KORPM_Ssym.txt", sep=r"\s+", header=None)
    k = k.iloc[:, [0, 1, 2, 3, 4]]
    k.columns = ["pdb_wt", "pdb_mut", "dir", "rev", "ddg_orig"]
    k["pair_id"] = [f"ssym_{i:03d}" for i in range(len(k))]
    # Values: KORPM's copy differs from ThermoNet's copy for 24/342 pairs. A third copy (ACDC-NN / PROSTATA
    # Ssym direct files, Fariselli group) agrees with ThermoNet on all of them -> ThermoNet/ACDC values are used
    # (majority, and they match the published Ssym); KORPM's value is kept as ddg_korpm_orig.
    t = pd.read_csv(RAW / "thermonet/s_sym.txt", sep=r"\s+", header=None, names=["wt", "mut", "pos", "a", "b", "ddg"])
    assert (t.wt.str[:4].str.upper().values == k.pdb_wt.values).all() and \
        (t.mut.str[:4].str.upper().values == k.pdb_mut.values).all() and \
        ((t.a + t.pos.astype(str) + t.b).values == (k.dir.str[0] + k.dir.str[2:-1] + k.dir.str[-1]).values).all()
    acdc = pd.concat([pd.read_csv(p, sep=r"\s+", header=None, names=["pc", "m", "ddg"])
                      for p in sorted((RAW / "prostata").glob("ssym_TS_dir_*.mut"))]).drop_duplicates()
    acdc["key"] = acdc.pc.str[:4].str.upper() + "|" + acdc.m
    k["key"] = k.pdb_wt + "|" + k.dir.str[0] + k.dir.str[2:]
    k["ddg_acdc"] = k.key.map(acdc.drop_duplicates("key").set_index("key").ddg)
    k["ddg_korpm_orig"] = k.ddg_orig
    k["ddg_orig"] = -t.ddg.values  # ThermoNet is positive=destabilising; store in the common negative=destab form
    dis = (k.ddg_korpm_orig - k.ddg_orig).abs() > 0.01
    log(f"Ssym: KORPM vs ThermoNet disagree on {int(dis.sum())}/{len(k)} pairs; ACDC copy found for "
        f"{k.ddg_acdc.notna().sum()} pairs, agrees with ThermoNet on {int(((k.ddg_acdc - k.ddg_orig).abs() < 0.01).sum())} "
        f"(of disputed: {int(((k.ddg_acdc - k.ddg_orig).abs() < 0.01)[dis].sum())}/{int(dis.sum())}), with KORPM on disputed: "
        f"{int(((k.ddg_acdc - k.ddg_korpm_orig).abs() < 0.01)[dis].sum())}")
    rows = []
    for r in k.itertuples():
        for direction, pdb, ms, sgn in (("direct", r.pdb_wt, r.dir, 1.0), ("reverse", r.pdb_mut, r.rev, -1.0)):
            rows.append(dict(pdb=pdb, mutstr=ms, ddg_orig=sgn * r.ddg_orig, direction=direction, pair_id=r.pair_id,
                             wt_pdb=r.pdb_wt + "_" + r.dir[1], ddg_korpm=-sgn * r.ddg_korpm_orig))
    d = pd.DataFrame(rows)
    d["wt"], d["chain"], d["resnum"], d["mut"] = parse_mut(d.mutstr.values)
    d["dataset"] = "ssym"
    return d


def load_s2648():
    d = pd.read_csv(RAW / "prostata/S2648.csv")
    d = d.rename(columns={"WILD_RES": "wt", "MUTANT_RES": "mut", "PH": "pH", "TEMPERATURE": "temp", "EXP_DDG": "ddg_orig"})
    d["pdb"], d["chain"] = d.PDB_CHAIN.str[:4].str.upper(), d.PDB_CHAIN.str[4]
    d["resnum"] = d.POSITION.astype(str)
    d["mutstr"] = d.wt + d.chain + d.resnum + d.mut
    d["dataset"] = "s2648"
    d["direction"] = "direct"
    d["pair_id"] = [f"s2648_{i}" for i in range(len(d))]
    return d


def prep_structure(args):
    pdb, chain = args
    src = mirror_path(pdb)
    if src is None:
        return (pdb, chain), {"error": "not in mirror"}
    try:
        st = read_structure(src)
        ch = chain
        if st[0].find_chain(ch) is None:
            prot = sorted({c.name for c in st[0] if chain_residues(st, c.name)})
            if len(prot) == 1:
                ch = prot[0]
            else:
                return (pdb, chain), {"error": f"chain {chain} absent ({prot})", "src": str(src)}
        out = SDIR / f"{pdb}_{ch}.pdb"
        info = clean_chain(st, ch, out)
        cst = read_structure(out)
        res = chain_residues(cst, ch)
        info.update(src=str(src), chain_used=ch, structure=str(out.relative_to(ROOT)), residues=res,
                    burial=cb_burial(cst, ch), exp_method=dict(st.info).get("_exptl.method", ""),
                    resolution=st.resolution, obsolete="obsolete" in str(src), n_res=len(res),
                    sequence="".join(a for _, a in res), context=site_context(st, ch),
                    other_protein_chains=";".join(sorted({c.name for c in st[0] if c.name != ch and chain_residues(st, c.name)})))
        return (pdb, chain), info
    except Exception as e:  # noqa: BLE001
        return (pdb, chain), {"error": repr(e), "src": str(src)}


def map_rows(d, S):
    """Add structure, chain_used, pdb_resnum, cb_nb, map_method. Offset search per (dataset, pdb, chain)."""
    out = []
    for (ds, pdb, ch), g in d.groupby(["dataset", "pdb", "chain"], sort=False):
        info = S[(pdb, ch)]
        g = g.copy()
        if "error" in info:
            g["map_method"] = "fail:" + info["error"]
            out.append(g)
            continue
        resmap = dict(info["residues"])
        nums = [int("".join(c for c in k if c.isdigit() or c == "-")) for k, _ in info["residues"]]
        num2key = {}
        for (k, _), n in zip(info["residues"], nums):
            num2key.setdefault(n, k)  # icode-free number -> first residue with that number

        keys = [k for k, _ in info["residues"]]

        def hits(off, mode="num"):
            out_ = []
            for r, w in zip(g.resnum, g.wt):
                if mode == "num":
                    key = num2key.get(int(r) + off)
                else:  # sequential index (1-based) over observed residues, e.g. Kabat-numbered antibodies
                    i = int(r) + off - 1
                    key = keys[i] if 0 <= i < len(keys) else None
                out_.append(key if key is not None and resmap.get(key) == w else None)
            return out_
        cnt = lambda h: sum(x is not None for x in h)  # noqa: E731
        h0 = hits(0)
        n0 = cnt(h0)
        best, bh = "exact", h0
        if n0 < len(g) and ALLOW_OFFSET:
            for mode in ("num", "seqidx"):
                for off in range(-150, 151):
                    h = hits(off, mode)
                    if cnt(h) > cnt(bh):
                        best, bh = (f"offset{off:+d}" if mode == "num" else f"seqidx{off:+d}"), h
            if not (cnt(bh) >= 0.9 * len(g) and cnt(bh) > n0):
                best, bh = "exact", h0
        g["pdb_resnum"] = bh
        g["map_method"] = [best if x is not None else "fail:wt_mismatch" for x in bh]
        g.loc[g.pdb_resnum.isna(), "map_method"] = "fail:wt_mismatch"
        g["structure"] = info["structure"]
        g["chain_used"] = info["chain_used"]
        g["cb_nb"] = [info["burial"].get(x, np.nan) if x else np.nan for x in bh]
        g["het_removed"] = info["het_removed"]
        g["exp_method"] = info["exp_method"]
        g["resolution"] = info["resolution"]
        g["obsolete_entry"] = info["obsolete"]
        cx = [info["context"].get(x, (False, "", False)) if x else (False, "", False) for x in bh]
        g["near_hetatm"] = [c[0] for c in cx]
        g["hetatm_near"] = [c[1] for c in cx]
        g["near_other_chain"] = [c[2] for c in cx]
        g["other_protein_chains"] = info["other_protein_chains"]
        out.append(g)
    return pd.concat(out)


def finalize(d, name):
    d = d.copy()
    d["ddg"] = -d.ddg_orig
    ok = ~d.map_method.str.startswith("fail")
    log(f"[{name}] rows {len(d)}, proteins {d.pdb.nunique()}; mapping: " +
        ", ".join(f"{k}={v}" for k, v in Counter(np.where(ok, np.where(d.map_method == 'exact', 'exact', 'offset'),
                                                           d.map_method)).items()))
    if (~ok).any():
        log(d.loc[~ok, ["pdb", "chain", "mutstr", "map_method"]].head(15).to_string())
    d = d[ok]
    sign_report(d[d.direction == "direct"], log=log, label=name)
    # duplicates (same structure, residue, mutation, direction): average
    keys = ["pdb", "chain_used", "pdb_resnum", "wt", "mut", "direction"]
    n = len(d)
    extra = [c for c in ["pH", "temp"] if c in d]
    agg = {"ddg": ("ddg", "mean"), "ddg_orig": ("ddg_orig", "mean"), "n_meas": ("ddg", "size"),
           "ddg_range": ("ddg", lambda x: x.max() - x.min()), "pair_id": ("pair_id", "first")}
    for c in ["structure", "cb_nb", "map_method", "het_removed", "exp_method", "resolution", "obsolete_entry", "chain",
              "near_hetatm", "hetatm_near", "near_other_chain", "other_protein_chains"]:
        agg[c] = (c, "first")
    for c in extra:
        agg[c] = (c, "mean")
    if "wt_pdb" in d:
        agg["wt_pdb"] = ("wt_pdb", "first")
        agg["ddg_korpm"] = ("ddg_korpm", "first")
    g = d.groupby(keys, sort=False).agg(**agg).reset_index()
    if len(g) < n:
        log(f"[{name}] duplicates averaged: {n} -> {len(g)} rows; max within-group range {g.ddg_range.max():.2f}")
    out = pd.DataFrame({
        "dataset": name,
        "protein_id": g.wt_pdb if "wt_pdb" in g else g.pdb + "_" + g.chain,  # Ssym reverse -> WT protein id
        "structure": g.structure, "chain": g.chain_used, "pdb_resnum": g.pdb_resnum.astype(str),
        "wt": g.wt, "mut": g.mut, "ddg": g.ddg, "split": "external", "cluster": "",
        "direction": g.direction, "pair_id": g.pair_id, "pdb_id": g.pdb,
        "ddg_orig": g.ddg_orig, "n_meas": g.n_meas, "map_method": g.map_method, "cb_nb": g.cb_nb,
        "struct_source": "xtal", "exp_method": g.exp_method, "resolution": g.resolution,
        "het_removed": g.het_removed, "obsolete_entry": g.obsolete_entry,
        "near_hetatm": g.near_hetatm, "hetatm_near": g.hetatm_near, "near_other_chain": g.near_other_chain,
        "other_protein_chains": g.other_protein_chains, "assembly": "asym_unit_model1_chain_only",
    })
    for c in extra:
        out[c] = g[c]
    if "ddg_korpm" in g:
        out["ddg_korpm"] = g.ddg_korpm  # alternative value from the KORPM copy (our sign convention)
    out.to_csv(OUT / f"{name}.csv", index=False)
    s = out.groupby("direction").agg(rows=("ddg", "size"), proteins=("protein_id", "nunique"),
                                     structures=("pdb_id", "nunique"), ddg_mean=("ddg", "mean"), ddg_sd=("ddg", "std"))
    log(f"[{name}] written {len(out)} rows\n" + s.round(3).to_string())
    return out


def main():
    sets = {"s669": load_s669(RAW / "korpm/S669.txt", "s669"), "s461": load_s669(RAW / "korpm/S461.txt", "s461"),
            "ssym": load_ssym(), "s2648": load_s2648()}
    for n, d in sets.items():
        log(f"{n}: {len(d)} rows, mean ddG_orig {d[d.direction == 'direct'].ddg_orig.mean():+.2f}")
    allpc = sorted({(p, c) for d in sets.values() for p, c in zip(d.pdb, d.chain)})
    log(f"structures to prepare: {len(allpc)}")
    with Pool(48) as pool:
        S = dict(pool.map(prep_structure, allpc, chunksize=2))
    errs = {k: v["error"] for k, v in S.items() if "error" in v}
    log(f"structure failures: {errs}")
    log("chain substitutions: " + str({k: v["chain_used"] for k, v in S.items() if "error" not in v and v["chain_used"] != k[1]}))
    log("obsolete entries used: " + str([k for k, v in S.items() if v.get("obsolete")]))
    pd.DataFrame([{"pdb": k[0], "chain": k[1], **{x: y for x, y in v.items() if x not in ("residues", "burial", "context")}}
                  for k, v in S.items()]).to_csv(OUT / "external_structures.csv", index=False)
    res = {n: finalize(map_rows(d, S), n) for n, d in sets.items()}
    # S461 subset of S669?
    k = lambda x: set(x.pdb_id + x.chain + x.pdb_resnum + x.wt + x.mut)  # noqa: E731
    log(f"s461 rows found in s669: {len(k(res['s461']) & k(res['s669']))}/{len(res['s461'])}")
    for n, x in res.items():
        log(f"[{n}] near_hetatm rows {x.near_hetatm.mean():.1%}, near_other_chain rows {x.near_other_chain.mean():.1%}")
    ss = res["ssym"]
    pr = ss.pivot_table(index="pair_id", columns="direction", values="ddg")
    log(f"ssym complete pairs: {pr.dropna().shape[0]}/{len(pr)}; corr(direct, -reverse) = "
        f"{np.corrcoef(pr.dropna().direct, -pr.dropna().reverse)[0, 1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
