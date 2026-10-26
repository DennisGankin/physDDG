#!/usr/bin/env python
"""Build data/processed/fireprot.csv (ddG rows) and fireprot_dtm.csv (dTm-only rows) from the on-disk
ThermoMPNN-curated FireProtDB table.

Input (read-only): /workspace/data/runtime_tables/4_fireprotDB_bestpH.csv (FireProtDB export curated by the
ThermoMPNN authors: pdb_id_corrected, pdb_position = 0-based index into pdb_sequence, 'best pH' per mutation),
/workspace/data/fireprot_pdbs/<pdb>.pdb, data/raw/thermompnn/fireprot_splits.pkl.

Sign: FireProtDB defines stabilising as ddG < -0.5 kcal/mol / dTm > +1 C (FireProtDB help page), i.e. ddG positive =
destabilising = our convention -> NO flip. Data checks (printed): corr(ddG, dTm) < 0; buried hydrophobic losses > 0.

Mapping: pdb_sequence (with 'X' for unresolved residues) is globally aligned to the observed residues of the
cleaned chain; pdb_position -> structure residue; WT letter must match, else row dropped (counted).
If the annotated chain is absent and the file has a single protein chain, that chain is used (recorded).
Chain/assembly choice: the deposited asymmetric unit, model 1, the annotated chain only (other chains are removed
from the cleaned file and listed in other_protein_chains); near_hetatm = any heavy atom of the site within 5 A of a
non-water HETATM (ligand/ion/cofactor; modified residues inside the chain excluded) in the ORIGINAL wwPDB entry
(local mirror; the on-disk FireProt PDBs were pre-cleaned to one chain without HETATM, so they cannot be used);
near_other_chain = same vs another protein chain of the asymmetric unit (possible interface).
Duplicates (same protein/residue/mutation): ddg = mean; n_meas, ddg_sd, ddg_range kept; flag_inconsistent if
range > 2 kcal/mol or opposite signs with both |ddg| > 1.
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ht_common import (ROOT, XTAL_DIR, align_map, cb_burial, chain_residues, clean_chain, read_structure,  # noqa: E402
                       mirror_context, sign_report)

warnings.filterwarnings("ignore")
SRC = "/workspace/data/runtime_tables/4_fireprotDB_bestpH.csv"
PDBDIR = Path("/workspace/data/fireprot_pdbs")
SPLITS = ROOT / "data/raw/thermompnn/fireprot_splits.pkl"
FOLDX_VARIANTS = ["/foldx_native_clean/results/variants_fireprot.csv",
                  "/foldx_native_clean/results/variants_fireprot_trainval.csv"]
OUT = ROOT / "data/processed"
SDIR = XTAL_DIR / "fireprot"


def log(*a):
    print(*a, flush=True)


def map_protein(pdb, sub):
    """Clean the chain, align pdb_sequence to it, return per-row (resnum, cb_nb, chain_used, ok) and info."""
    st = read_structure(PDBDIR / f"{pdb}.pdb")
    chain = sub.chain.iloc[0]
    prot_chains = [c.name for c in st[0] if chain_residues(st, c.name)]
    note = ""
    if st[0].find_chain(chain) is None or not chain_residues(st, chain):
        if len(set(prot_chains)) == 1:
            note = f"chain {chain}->{prot_chains[0]}"
            chain = prot_chains[0]
        else:
            return None, {"pdb": pdb, "error": f"chain {chain} absent; chains {prot_chains}"}
    out = SDIR / f"{pdb}_{chain}.pdb"
    info = clean_chain(st, chain, out)
    cst = read_structure(out)
    cr = chain_residues(cst, chain)
    sseq = "".join(a for _, a in cr)
    pseq = sub.pdb_sequence.iloc[0]
    amap, ident = align_map(pseq, sseq)
    bur = cb_burial(cst, chain)
    # the on-disk files are pre-cleaned (single chain, no HETATM): context comes from the original mirror entry
    ctx, minfo = mirror_context(pdb, cst, chain)
    res = []
    for p, w in zip(sub.pdb_position, sub.wild_type):
        j = amap.get(int(p))
        ok = j is not None and sseq[j] == w
        rn = cr[j][0] if ok else None
        cx = ctx.get(rn, (False, "", False)) if ok else (False, "", False)
        res.append((rn, bur.get(rn, np.nan) if ok else np.nan, ok, cx[0], cx[1], cx[2]))
    info.update(pdb=pdb, chain=chain, chain_note=note, n_models=len(st), struct_len=len(sseq),
                assembly="asymmetric unit (deposited file), model 1, annotated chain only", **minfo,
                fireprot_seq_len=len(pseq), aln_identity=round(ident, 3), structure=str(out.relative_to(ROOT)),
                n_rows=len(sub), n_ok=sum(r[2] for r in res))
    return res, info


def main():
    f = pd.read_csv(SRC).rename(columns={"Unnamed: 0": "src_row"})
    log(f"rows {len(f)}, proteins {f.pdb_id_corrected.nunique()}, with ddG {f.ddG.notna().sum()}, "
        f"dTm-only {(f.ddG.isna() & f.dTm.notna()).sum()}")
    assert all(s[p] == w for s, p, w in zip(f.pdb_sequence, f.pdb_position, f.wild_type))
    log("pdb_sequence[pdb_position] == wild_type for 100% of rows (pdb_position is 0-based)")
    both = f[["ddG", "dTm"]].dropna()
    log(f"sign evidence: Pearson(ddG, dTm) = {both.corr().iloc[0, 1]:.2f} on {len(both)} rows with both")

    # ---- structure mapping (all rows, incl. dTm-only) ----
    infos, parts = [], []
    for pdb, sub in f.groupby("pdb_id_corrected"):
        res, info = map_protein(pdb, sub)
        infos.append(info)
        if res is None:
            log("  FAIL", info)
            continue
        sub = sub.copy()
        sub["pdb_resnum"], sub["cb_nb"], sub["map_ok"], sub["near_hetatm"], sub["hetatm_near"], sub["near_other_chain"] = zip(*res)
        sub["other_protein_chains"] = info["other_protein_chains"]
        sub["chain_used"], sub["structure"] = info["chain"], info["structure"]
        parts.append(sub)
    info = pd.DataFrame(infos)
    info.to_csv(OUT / "fireprot_structures.csv", index=False)
    m = pd.concat(parts)
    bad = info[info.n_ok < info.n_rows]
    log(f"mapping: {m.map_ok.sum()}/{len(f)} rows mapped with WT match ({1 - m.map_ok.mean():.3%} fail)")
    log("chain substitutions: " + "; ".join(f"{r.pdb}:{r.chain_note}" for r in info.itertuples() if r.chain_note))
    log("proteins with failures:\n" + bad[["pdb", "aln_identity", "n_rows", "n_ok", "struct_len",
                                           "fireprot_seq_len"]].to_string())
    log("dropped rows (ddG non-null):", int((~m.map_ok & m.ddG.notna()).sum()))
    m = m[m.map_ok]

    # ---- cross-check against previous FoldX run's residue mapping (keyed by 1-based pdb_sequence index) ----
    fx = pd.concat([pd.read_csv(p) for p in FOLDX_VARIANTS])
    fx["pdb_resnum"] = fx.pdb_resnum.astype(str).str.replace(r"\.0$", "", regex=True)
    q = m[m.ddG.notna()].assign(position_1based=lambda x: x.pdb_position + 1)
    q = q.merge(fx.drop_duplicates(["entity_id", "wt_aa", "mut_aa", "position_1based"]),
                left_on=["pdb_id_corrected", "wild_type", "mutation", "position_1based"],
                right_on=["entity_id", "wt_aa", "mut_aa", "position_1based"], how="inner", suffixes=("", "_fx"))
    same = q.pdb_resnum.astype(str) == q.pdb_resnum_fx
    log(f"cross-check vs /foldx_native_clean mapping: {len(q)} rows; agree on runnable rows "
        f"{same[q.runnable].mean():.2%}; rows they marked non-runnable (wt_letter_mismatch/out_of_range) that we map: "
        f"{int((~q.runnable).sum())} (their mapping ignored gaps in pdb_sequence)")

    # ---- splits ----
    sp = pickle.load(open(SPLITS, "rb"))
    split = {p: k for k in ("train", "val", "test") for p in sp[k]}
    m["split"] = m.pdb_id_corrected.map(split).fillna("none")
    m["homologue_free_thermompnn"] = m.pdb_id_corrected.isin(set(sp["homologue-free"]))

    # ---- ddG table: average duplicates ----
    d = m[m.ddG.notna()].copy()
    d["ddg"] = d.ddG  # no flip (see docstring)
    d["wt"], d["mut"] = d.wild_type, d.mutation
    sign_report(d, log=log, label="fireprot")
    keys = ["pdb_id_corrected", "chain_used", "pdb_resnum", "wt", "mut"]
    n_before = len(d)
    agg = d.groupby(keys, sort=False).agg(
        ddg=("ddg", "mean"), n_meas=("ddg", "size"), ddg_sd=("ddg", "std"),
        ddg_min=("ddg", "min"), ddg_max=("ddg", "max"), dTm=("dTm", "mean"), pH=("pH", "mean"),
        temp=("tm", "first"), method=("method", lambda x: ";".join(sorted(set(map(str, x))))),
        source=("publication_doi", lambda x: ";".join(sorted(set(map(str, x.dropna()))))),
        experiment_id=("experiment_id", lambda x: ";".join(x)),
        seq_pos=("pdb_position", "first"), uniprot=("uniprot_id", "first"), protein_name=("protein_name", "first"),
        structure=("structure", "first"), split=("split", "first"), cb_nb=("cb_nb", "first"),
        near_hetatm=("near_hetatm", "first"), hetatm_near=("hetatm_near", "first"),
        near_other_chain=("near_other_chain", "first"), other_protein_chains=("other_protein_chains", "first"),
        homologue_free_thermompnn=("homologue_free_thermompnn", "first"), oligomeric_state=("oligomeric_state", "first"),
        structure_method=("structure_method", "first"), resolution=("resolution", "first")).reset_index()
    agg["ddg_range"] = agg.ddg_max - agg.ddg_min
    agg["flag_inconsistent"] = (agg.ddg_range > 2) | ((agg.ddg_min < -1) & (agg.ddg_max > 1))
    # Nisthal 2019 GB1 (1PGA): unfolded variants reported at the 4.0 kcal/mol assay ceiling -> censored lower bound
    vc = d.groupby("pdb_id_corrected").ddg.agg(lambda x: (x == x.max()).sum())
    cens = set(vc[vc >= 10].index)
    agg["flag_censored"] = agg.pdb_id_corrected.isin(cens) & (agg.ddg == agg.pdb_id_corrected.map(d.groupby("pdb_id_corrected").ddg.max()))
    log(f"duplicates: {n_before} ddG rows -> {len(agg)} unique mutations; {int((agg.n_meas > 1).sum())} averaged; "
        f"{int(agg.flag_inconsistent.sum())} flagged inconsistent; censored-at-max proteins {sorted(cens)} "
        f"({int(agg.flag_censored.sum())} rows)")
    dup = agg[agg.n_meas > 1]
    log(f"  duplicate groups: median sd {dup.ddg_sd.median():.2f}, median range {dup.ddg_range.median():.2f}")

    out = pd.DataFrame({
        "dataset": "fireprot", "protein_id": agg.pdb_id_corrected, "structure": agg.structure,
        "chain": agg.chain_used, "pdb_resnum": agg.pdb_resnum.astype(str), "wt": agg.wt, "mut": agg.mut,
        "ddg": agg.ddg, "split": agg.split, "cluster": "",
        "seq_pos": agg.seq_pos + 1, "uniprot": agg.uniprot, "pH": agg.pH, "temp": agg.temp, "method": agg.method,
        "source": agg.source, "struct_source": "xtal", "n_meas": agg.n_meas, "ddg_sd": agg.ddg_sd,
        "ddg_range": agg.ddg_range, "flag_inconsistent": agg.flag_inconsistent, "flag_censored": agg.flag_censored,
        "dTm": agg.dTm, "cb_nb": agg.cb_nb, "homologue_free_thermompnn": agg.homologue_free_thermompnn,
        "near_hetatm": agg.near_hetatm, "hetatm_near": agg.hetatm_near, "near_other_chain": agg.near_other_chain,
        "other_protein_chains": agg.other_protein_chains,
        "assembly": "asym_unit_model1_chain_only",
        "protein_name": agg.protein_name, "oligomeric_state": agg.oligomeric_state,
        "structure_method": agg.structure_method, "resolution": agg.resolution, "experiment_id": agg.experiment_id,
    })
    out["structure_method"] = out.structure_method.fillna("")
    out = out.sort_values(["split", "protein_id", "seq_pos", "mut"]).reset_index(drop=True)
    out.to_csv(OUT / "fireprot.csv", index=False)

    # ---- dTm-only rows (proxy; not ddG) ----
    t = m[m.ddG.isna() & m.dTm.notna()]
    tt = pd.DataFrame({"dataset": "fireprot_dtm", "protein_id": t.pdb_id_corrected, "structure": t.structure,
                       "chain": t.chain_used, "pdb_resnum": t.pdb_resnum.astype(str), "wt": t.wild_type,
                       "mut": t.mutation, "ddg": np.nan, "split": t.split, "cluster": "", "seq_pos": t.pdb_position + 1,
                       "score": t.dTm, "pH": t.pH, "method": t.method, "uniprot": t.uniprot_id})
    tt.to_csv(OUT / "fireprot_dtm.csv", index=False)
    log(f"fireprot_dtm.csv: {len(tt)} rows, {tt.protein_id.nunique()} proteins (score = dTm, positive = stabilising)")

    s = out.groupby("split").agg(rows=("ddg", "size"), proteins=("protein_id", "nunique"), ddg_mean=("ddg", "mean"),
                                 ddg_sd=("ddg", "std"), frac_gt1=("ddg", lambda x: (x > 1).mean()))
    log(s.round(3).to_string())
    log(f"total {len(out)} rows, {out.protein_id.nunique()} proteins; near_hetatm rows {out.near_hetatm.mean():.1%} "
        f"({out[out.near_hetatm].protein_id.nunique()} proteins); near_other_chain rows {out.near_other_chain.mean():.1%}; "
        f"proteins whose file has other protein chains: {out[out.other_protein_chains.fillna('') != ''].protein_id.nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
