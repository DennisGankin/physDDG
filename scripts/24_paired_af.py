#!/usr/bin/env python
"""Paired crystal-vs-AlphaFold variant sets for AF-mode development (data/processed/paired_af.csv).

Sources (each used if present; rerun after the data-HT agent writes its tables):
  fireprot_raw : (fallback, only if data/processed/fireprot.csv is absent) /workspace/data/runtime_tables/4_fireprotDB_bestpH.csv (xtal = /workspace/data/fireprot_pdbs/<pdb>.pdb;
                 `position` = UniProt position (verified: sequence[position-1]==wild_type for 100% rows);
                 `pdb_position` = 0-based index into pdb_sequence (verified 100%) -> mapped to the PDB file residue)
  fireprot/s669/ssym : data/processed/<name>.csv canonical tables (structure, chain, pdb_resnum, wt, mut, ddg[, uniprot]).
                 Missing UniProt accessions are taken from SIFTS pdb_chain_uniprot.tsv (downloaded to data/raw/sifts/).

For each (xtal structure, chain, UniProt): AF model chosen as human v4 (/af_data/af2, as for Domainome), else v6
(Swiss-Prot / proteome tars), else EBI download (latest). Xtal chain sequence (observed residues) is aligned to the AF sequence (BLOSUM62 local alignment);
every variant residue is mapped xtal resnum -> AF resnum (= UniProt numbering for single-fragment models).
WT identity is checked on both sides. Also reported: aligned CA RMSD (after superposition) and the CA deviation at
the mutated site, plus pLDDT.
Structures written: data/structures/paired_af/af_full/<AF file>.pdb (copy) and
data/structures/paired_af/af_xtalspan/<uniprot>__<pdb>_<chain>.pdb (AF model cropped to the span aligned to the xtal
chain, UniProt numbering, chain A, B=pLDDT).
"""
import os, sys, subprocess, gzip
import numpy as np
import pandas as pd
import gemmi
from Bio import Align
from Bio.Align import substitution_matrices
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afdb_io

ROOT = afdb_io.ROOT
SDIR = os.path.join(ROOT, "data/structures/paired_af")
SIFTS = os.path.join(ROOT, "data/raw/sifts/pdb_chain_uniprot.tsv.gz")
SIFTS_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz"
FIREPROT_RAW = "/workspace/data/runtime_tables/4_fireprotDB_bestpH.csv"
# human: v4 (af2 tree; same release as the Domainome structures), everything else: v6, then EBI
AF_ORDER = ("af2_tree", "v6_swissprot", "v6_human", "v6_yeast", "v6_ecoli", "v6_mouse", "v4_human_tar")

_al = Align.PairwiseAligner()
_al.mode = "local"
_al.substitution_matrix = substitution_matrices.load("BLOSUM62")
_al.open_gap_score = -10
_al.extend_gap_score = -0.5


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def chain_residues(path, chain):
    path = resolve(path)
    try:
        st = gemmi.read_structure(path)
    except RuntimeError:  # e.g. misplaced ANISOU records in some raw FireProt PDBs
        txt = "".join(l for l in open(path) if not l.startswith("ANISOU"))
        st = gemmi.read_pdb_string(txt)
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    model = st[0]
    ch = model.find_chain(chain) if hasattr(model, "find_chain") else model[chain]
    if ch is None:
        return None
    res = []
    for r in ch:
        if r.het_flag == "H" and r.name not in ("MSE",):
            continue
        info = gemmi.find_tabulated_residue(r.name)
        if info is None or not info.is_amino_acid():
            continue
        code = info.one_letter_code.upper()
        if r.name == "MSE":
            code = "M"
        ca = r.find_atom("CA", "*")
        res.append((f"{r.seqid.num}{r.seqid.icode.strip()}", code, np.array(ca.pos.tolist()) if ca else None))
    return res


MODELS = {}  # uniprot -> list of candidate AF models (all entries/fragments of the first source that has any)


def uniprot_merged_into(acc):
    """Secondary/merged accessions (e.g. Q9H782 -> O75400): ask UniProt REST which accession replaced it."""
    import json as _json
    txt = afdb_io._curl(f"https://rest.uniprot.org/uniprotkb/{acc}?format=json", ua=False)
    try:
        j = _json.loads(txt)
    except Exception:
        return None
    if j.get("primaryAccession") and j.get("primaryAccession") != acc:
        return j["primaryAccession"]
    tgt = (j.get("inactiveReason") or {}).get("mergeDemergeTo") or []
    return tgt[0] if tgt else None


def af_models(acc):
    out = _af_models(acc)
    if not out:
        new = uniprot_merged_into(acc)
        if new:
            out = _af_models(new)
            for c in out:
                c["af_accession"] = new
    return out


def _af_models(acc):
    out, src = [], None
    for c in afdb_io.candidates(acc, sources=AF_ORDER, ebi=True):
        if src is not None and c["source"] != src:
            break
        src = c["source"]
        c["seqinfo"] = afdb_io.structure_seq(c["text"])
        out.append(c)
    return out


def pair_entity(args):
    """args: (structure, chain, uniprot) -> dict with mapping info and per-residue map."""
    xs, xc, acc = args
    out = dict(xtal_structure=xs, xtal_chain=xc, uniprot=acc)
    xr = chain_residues(xs, xc)
    if not xr:
        out["pair_status"] = "xtal_chain_missing"
        return out, None
    cands = MODELS.get(acc, []) if isinstance(acc, str) else []
    if not cands:
        out["pair_status"] = "no_af_model"
        return out, {rn: (aa, None) for rn, aa, _ in xr}
    xseq = "".join(r[1] for r in xr)
    best = None
    for c in cands:  # several fragments / segment models: keep the one with most identical aligned residues
        aseq, anums, apl = c["seqinfo"]
        aln = _al.align(xseq, aseq)[0]
        mapping = {}
        for (xs0, xe0), (as0, ae0) in zip(*aln.aligned):
            for k in range(xe0 - xs0):
                mapping[xs0 + k] = as0 + k
        ident = sum(xseq[i] == aseq[j] for i, j in mapping.items())
        if best is None or ident > best[0]:
            best = (ident, c, mapping)
    ident, c, mapping = best
    aseq, anums, apl = c["seqinfo"]
    off = c["uniprot_offset"]
    st = gemmi.read_pdb_string(c["text"])
    afca = {}
    for r in st[0][0]:
        a = r.find_atom("CA", "*")
        afca[r.seqid.num] = np.array(a.pos.tolist())
    # superpose aligned CAs (Kabsch)
    P, Q = [], []
    for i, j in mapping.items():
        if xr[i][2] is not None:
            P.append(xr[i][2]); Q.append(afca[anums[j]])
    P, Q = np.array(P), np.array(Q)
    dev = None
    if len(P) >= 3:
        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        U, S, Vt = np.linalg.svd(Pc.T @ Qc)
        dd = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1, 1, dd]) @ Vt
        diff = Pc @ R - Qc
        dev = np.sqrt((diff ** 2).sum(1))
        rmsd = float(np.sqrt((dev ** 2).mean()))
    else:
        rmsd = np.nan
    fname = os.path.basename(c["src_path"].split("::")[-1]).replace(".pdb.gz", ".pdb")
    fpath = os.path.join(SDIR, "af_full", fname)
    if not os.path.exists(fpath):
        open(fpath, "w").write(c["text"])
    # crop AF to aligned span
    lo = min(mapping.values()); hi = max(mapping.values())
    new = gemmi.Structure(); nm = gemmi.Model("1"); nc = gemmi.Chain("A")
    for r in st[0][0]:
        if anums[lo] <= r.seqid.num <= anums[hi]:
            r2 = r.clone(); r2.seqid = gemmi.SeqId(r.seqid.num + off, " "); nc.add_residue(r2)
    nm.add_chain(nc); new.add_model(nm); new.setup_entities()
    tag = os.path.splitext(os.path.basename(xs))[0]
    cpath = os.path.join(SDIR, "af_xtalspan", f"{acc}__{tag}_{xc}.pdb")
    hdr = (f"REMARK 250 {fname} ({c['src_path']}) cropped to residues aligned to {xs} chain {xc}\n"
           f"REMARK 250 UniProt numbering; B-factor = pLDDT\n")
    open(cpath, "w").write(hdr + new.make_pdb_string(gemmi.PdbWriteOptions(minimal=True)))
    devmap, locmap = {}, {}
    if dev is not None:
        idx = [i for i, j in mapping.items() if xr[i][2] is not None]
        for k, i in enumerate(idx):
            devmap[i] = float(dev[k])
        # local RMSD: CA atoms within 10 A of the site (in the xtal), superposed locally
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        for k, i in enumerate(idx):
            nb = np.where(D[k] < 10.0)[0]
            if len(nb) >= 4:
                A, B = P[nb] - P[nb].mean(0), Q[nb] - Q[nb].mean(0)
                U, S_, Vt = np.linalg.svd(A.T @ B)
                R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
                locmap[i] = float(np.sqrt((((A @ R) - B) ** 2).sum(1).mean()))
    resmap = {}
    for i, (rn, aa, _) in enumerate(xr):
        j = mapping.get(i)
        resmap[rn] = (aa, (str(anums[j] + off), aseq[j], apl[j], devmap.get(i, np.nan), locmap.get(i, np.nan))
                      if j is not None else None)
    out.update(pair_status="ok", af_source=c["source"], af_version=c["version"], af_src=c["src_path"], af_uniprot_offset=off,
               af_accession=c.get("af_accession", acc),
               af_full_structure=os.path.relpath(fpath, ROOT), af_structure=os.path.relpath(cpath, ROOT),
               aln_identity=ident / max(1, len(mapping)), aln_cov_xtal=len(mapping) / len(xseq),
               n_xtal_res=len(xseq), af_len=len(aseq), ca_rmsd_aligned=rmsd,
               af_span_plddt_mean=float(np.mean([apl[j] for j in range(lo, hi + 1)])))
    return out, resmap


def load_sifts():
    if not os.path.exists(SIFTS):
        os.makedirs(os.path.dirname(SIFTS), exist_ok=True)
        subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", SIFTS, SIFTS_URL], check=True)
    s = pd.read_csv(SIFTS, sep="\t", comment="#", dtype=str)
    s.columns = [c.lower() for c in s.columns]
    return s


def fireprot_raw():
    f = pd.read_csv(FIREPROT_RAW)
    rows = []
    cache = {}
    bad = 0
    for r in f.itertuples():
        pdb = r.pdb_id_corrected
        path = f"/workspace/data/fireprot_pdbs/{pdb}.pdb"
        key = (path, r.chain)
        if key not in cache:
            cache[key] = chain_residues(path, r.chain)
        xr = cache[key]
        xseq = "".join(x[1] for x in xr) if xr else ""
        # pdb_sequence is the observed chain sequence; pdb_position its 0-based index
        if xr and xseq == r.pdb_sequence:
            rn = xr[int(r.pdb_position)][0]
        elif xr and r.pdb_sequence in xseq:
            rn = xr[xseq.find(r.pdb_sequence) + int(r.pdb_position)][0]
        else:
            rn = None; bad += 1
        rows.append(dict(dataset="fireprot_raw", protein_id=f"{pdb}_{r.chain}", xtal_structure=path, xtal_chain=r.chain,
                         xtal_resnum=rn, wt=r.wild_type, mut=r.mutation, ddg=r.ddG, uniprot=r.uniprot_id,
                         seq_pos_source=int(r.position), source_id=r.experiment_id, pH=r.pH))
    print(f"fireprot_raw: {len(rows)} rows; pdb_sequence not matching xtal chain: {bad}")
    return pd.DataFrame(rows)


def canonical(name):
    p = os.path.join(ROOT, "data/processed", f"{name}.csv")
    if not os.path.exists(p):
        print(f"{name}: {p} not present yet - skipped (rerun this script later)")
        return None
    t = pd.read_csv(p, dtype={"pdb_resnum": str, "chain": str})
    if "struct_source" in t.columns:
        t = t[t.struct_source.fillna("xtal") != "af"]
    t = t.reset_index(drop=True)
    d = pd.DataFrame(dict(dataset=t.dataset if "dataset" in t else name, protein_id=t.protein_id,
                          xtal_structure=t.structure, xtal_chain=t.chain, xtal_resnum=t.pdb_resnum,
                          wt=t.wt, mut=t.mut, ddg=t.ddg,
                          uniprot=t.uniprot if "uniprot" in t.columns else np.nan,
                          seq_pos_source=np.nan,
                          source_id=t.index))
    for c in ("split", "cluster", "direction", "pair_id", "experiment_id"):
        if c in t.columns:
            d[c] = t[c].values
    if name == "fireprot" and "experiment_id" in t.columns:
        # canonical seq_pos is the index in the PDB sequence; take the UniProt position from the raw FireProt table
        raw = pd.read_csv(FIREPROT_RAW, usecols=["experiment_id", "position"]).drop_duplicates("experiment_id")
        d["seq_pos_source"] = d.experiment_id.map(raw.set_index("experiment_id").position).values
    if d.uniprot.isna().any():
        s = load_sifts()
        s["key"] = s.pdb.str.upper() + "_" + s.chain
        # one accession per PDB chain: the one covering most residues
        s["n"] = pd.to_numeric(s.sp_end, errors="coerce") - pd.to_numeric(s.sp_beg, errors="coerce")
        best = s.sort_values("n", ascending=False).drop_duplicates("key").set_index("key").sp_primary
        pdbid = d.xtal_structure.astype(str).map(lambda x: os.path.basename(x)[:4].upper())
        k = pdbid + "_" + d.xtal_chain.astype(str)
        fill = k.map(best)
        d["uniprot_from_sifts"] = d.uniprot.isna() & fill.notna()
        d["uniprot"] = d.uniprot.astype(object).where(d.uniprot.notna(), fill)
    print(f"{name}: {len(d)} rows, uniprot missing after SIFTS: {d.uniprot.isna().sum()}")
    return d


def main():
    for sub in ("af_full", "af_xtalspan"):
        os.makedirs(os.path.join(SDIR, sub), exist_ok=True)
    for lab, _ in afdb_io.TARS:
        afdb_io.tar_index(lab)
    afdb_io._af2()
    # the canonical fireprot table (data-HT agent, cleaned xtal chains) supersedes the raw runtime table
    fp = canonical("fireprot")
    parts = [fp if fp is not None else fireprot_raw()] + [canonical(n) for n in ("s669", "ssym")]
    v = pd.concat([p for p in parts if p is not None], ignore_index=True)
    ents = v[["xtal_structure", "xtal_chain", "uniprot"]].drop_duplicates()
    ents = ents[ents.uniprot.notna()]
    # resolve AF models serially (EBI rate-limits bursts with HTTP 429), then align in parallel
    for acc in sorted(ents.uniprot.unique()):
        MODELS[acc] = af_models(acc)
    print("AF models per source:", pd.Series([m[0]["source"] if m else "none" for m in MODELS.values()]).value_counts().to_dict())
    print("no AF model:", sorted(a for a, m in MODELS.items() if not m))
    with Pool(32) as p:
        res = p.map(pair_entity, [tuple(x) for x in ents.itertuples(index=False)], chunksize=1)
    E = pd.DataFrame([r for r, _ in res])
    maps = {(r["xtal_structure"], r["xtal_chain"], r["uniprot"]): m for r, m in res}
    v = v.merge(E, on=["xtal_structure", "xtal_chain", "uniprot"], how="left")
    cols = dict(xtal_aa=[], af_resnum=[], af_aa=[], plddt=[], ca_dev_site=[], local_rmsd_10A=[])
    for r in v.itertuples():
        m = maps.get((r.xtal_structure, r.xtal_chain, r.uniprot))
        e = m.get(str(r.xtal_resnum)) if m else None
        xa = e[0] if e else None
        a = e[1] if e and e[1] else (None, None, np.nan, np.nan, np.nan)
        cols["xtal_aa"].append(xa); cols["af_resnum"].append(a[0]); cols["af_aa"].append(a[1])
        cols["plddt"].append(a[2]); cols["ca_dev_site"].append(a[3]); cols["local_rmsd_10A"].append(a[4])
    for k, x in cols.items():
        v[k] = x
    v["wt_match_xtal"] = v.xtal_aa == v.wt
    # Ssym/S669 'reverse' rows use the mutant structure: the AF (wild-type) model carries the reverse's `mut` there
    rev = v.direction.eq("reverse") if "direction" in v.columns else pd.Series(False, index=v.index)
    v["af_expected_aa"] = np.where(rev, v.mut, v.wt)
    v["wt_match_af"] = v.af_aa == v.af_expected_aa
    v["seqpos_agree"] = np.where(v.seq_pos_source.notna() & v.af_resnum.notna(),
                                 v.af_resnum.astype(str) == v.seq_pos_source.astype("Int64").astype(str), pd.NA)
    v["structure"] = v.af_structure; v["chain"] = np.where(v.af_resnum.notna(), "A", None)
    v["pdb_resnum"] = v.af_resnum; v["struct_source"] = "af"
    v["seq_pos"] = pd.to_numeric(v.af_resnum, errors="coerce")
    v["usable"] = v.wt_match_xtal & v.wt_match_af
    v["af_is_mutant_state"] = rev  # True: AF model = the reverse mutation's *target* state, not its starting structure
    out = os.path.join(ROOT, "data/processed/paired_af.csv")
    v.to_csv(out, index=False)
    E.to_csv(os.path.join(ROOT, "data/processed/paired_af_entities.csv"), index=False)
    print(E.pair_status.value_counts().to_string())
    for ds, g in v.groupby("dataset"):
        print(f"[{ds}] rows {len(g)} proteins {g.uniprot.nunique()} | AF mapped {g.af_resnum.notna().mean():.4f} | "
              f"wt==xtal {g.wt_match_xtal.mean():.4f} | wt==AF {g.wt_match_af.mean():.4f} | usable {g.usable.sum()} | "
              f"seqpos agree {pd.to_numeric(g.seqpos_agree, errors='coerce').mean():.4f} | pLDDT median {g.plddt.median():.1f} "
              f"frac<70 {(g.plddt<70).mean():.3f} | site CA dev median {g.ca_dev_site.median():.2f} | local RMSD10 median "
              f"{g.local_rmsd_10A.median():.2f} p90 {g.local_rmsd_10A.quantile(.9):.2f}")
    print(E[["aln_identity", "aln_cov_xtal", "ca_rmsd_aligned", "af_span_plddt_mean"]].describe().round(3).to_string())
    print("no AF model entities:", E[E.pair_status != "ok"].groupby("uniprot").size().to_dict())
    na = v[(v.pair_status == "ok") & v.af_resnum.isna()]
    print("rows in paired entities but site not aligned to AF:", na.groupby(["dataset", "protein_id"]).size().to_dict())


if __name__ == "__main__":
    main()
