#!/usr/bin/env python
"""AlphaFold structures for the Domainome constructs.

For each Domainome domain (natural proteins; the 3 'rockdoms' de novo designs have no AF DB entry):
  1. find the AF DB model of its UniProt entry. Priority: human proteome v4 (/af_data/af2 tree,
     = AF DB release 4, the release current when Domainome and the FoldX runs were done), then v6
     proteome/Swiss-Prot tars, then EBI AFDB download.
     For proteins >2700 aa (AF DB fragments F1..Fn, 1400 aa every 200 aa) the fragment in which
     the construct is most central is used; fragment residue i <-> UniProt i + 200*(k-1).
  2. verify the construct aa_seq against the AF sequence at the UniProt offset given by the
     domain_ID suffix (= UniProt position of the construct's first residue).
  3. write
       data/structures/domainome/af_full/AF-<acc>-F<k>-model_v<n>.pdb   (unchanged copy, AF numbering)
       data/structures/domainome/af_construct/<domain_ID>.pdb            (crop to the assayed construct,
            chain A, residues renumbered to UniProt numbering = Domainome `position` = FoldX pdb_resnum;
            B-factor = pLDDT kept)
  4. data/processed/domainome_structures.csv: one row per domain with provenance, match stats and
     construct pLDDT summary; data/processed/domainome_residue_plddt.csv: per construct residue pLDDT.
"""
import os, sys
import numpy as np
import pandas as pd
import gemmi
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afdb_io

ROOT = afdb_io.ROOT
DOM = "/workspace/Domainome/Extended_data_Table_2_fitness_scores_normalized_domainranks.txt"
SDIR = os.path.join(ROOT, "data/structures/domainome")
FULL = os.path.join(SDIR, "af_full")
CROP = os.path.join(SDIR, "af_construct")


def domains():
    d = pd.read_csv(DOM, sep="\t", usecols=["domain_ID", "uniprot_ID", "aa_seq", "wt_aa"])
    w = d[d.wt_aa.isna()][["domain_ID", "uniprot_ID", "aa_seq"]].drop_duplicates("domain_ID")
    w["start"] = w.domain_ID.str.rsplit("_", n=1).str[-1].astype(int)
    w["L"] = w.aa_seq.str.len()
    w["is_design"] = w.domain_ID.str.contains("rockdoms")
    return w.reset_index(drop=True)


from collections import namedtuple
Row = namedtuple("Row", ["domain_ID", "uniprot_ID", "aa_seq", "start", "L", "is_design"])


def n_mismatch(a, b):
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def process(t):
    row = Row(*t)
    dom, acc, cseq, start, L = row.domain_ID, row.uniprot_ID, row.aa_seq, int(row.start), int(row.L)
    end = start + L - 1
    rec = dict(domain_ID=dom, uniprot=acc, construct_start=start, construct_end=end, L=L)
    if row.is_design:
        rec.update(status="design_no_afdb")
        return rec, None
    best = None
    tried = []
    for c in afdb_io.candidates(acc):
        off = c["uniprot_offset"]
        seq, nums, pl = afdb_io.structure_seq(c["text"])
        # AF numbering is 1..len within the model
        lo, hi = start - off, end - off
        tried.append(f'{c["source"]}:F{c["frag"]}')
        if lo < 1 or hi > len(seq):
            continue
        sub = seq[lo - 1:hi]
        mm = n_mismatch(sub, cseq)
        centrality = min(lo - 1, len(seq) - hi)
        alt = seq.find(cseq)
        key = (mm, -centrality)
        if best is None or key < best[0]:
            best = (key, c, off, seq, nums, pl, mm, alt)
        if mm == 0 and (c["n_frags"] == 1 or centrality >= 200):
            break
        # keep scanning other sources only if mismatched
    if best is None:
        rec.update(status="no_af_model", tried=";".join(tried))
        return rec, None
    _, c, off, seq, nums, pl, mm, alt = best
    # If the construct does not match at the domain_ID offset but occurs exactly elsewhere in the AF/UniProt
    # sequence, the Domainome numbering is shifted: use the exact match and record the shift
    # (UniProt position = Domainome position + seqpos_shift).
    shift = 0
    if mm > 0 and alt >= 0:
        shift = (alt + 1 + off) - start
    rec.update(seqpos_shift=shift)
    rec.update(af_source=c["source"], af_version=c["version"], af_frag=c["frag"], af_n_frags=c["n_frags"],
               af_frag_offset=off, af_len=len(seq), af_src=c["src_path"], construct_mismatches=mm,
               construct_found_at=(alt + 1 + off) if alt >= 0 else -1)
    # write full copy
    fname = f"AF-{acc}-F{c['frag']}-model_v{c['version']}.pdb"
    fpath = os.path.join(FULL, fname)
    if not os.path.exists(fpath):
        open(fpath, "w").write(c["text"])
    # crop
    st = gemmi.read_pdb_string(c["text"])
    st.setup_entities()
    model = st[0]
    ch = model[0]
    new = gemmi.Structure()
    new.cell = st.cell
    new.spacegroup_hm = st.spacegroup_hm
    nm = gemmi.Model("1")
    nc = gemmi.Chain("A")
    res_pl = []
    for r in ch:
        u = r.seqid.num + off
        if start + shift <= u <= end + shift:
            r2 = r.clone()
            r2.seqid = gemmi.SeqId(u, " ")
            nc.add_residue(r2)
            ca = r.find_atom("CA", "*")
            k = u - shift - start
            res_pl.append((dom, u, u - shift, gemmi.find_tabulated_residue(r.name).one_letter_code.upper(),
                           cseq[k] if 0 <= k < L else "", ca.b_iso))
    nm.add_chain(nc)
    new.add_model(nm)
    new.setup_entities()
    cpath = os.path.join(CROP, f"{dom}.pdb")
    hdr = (f"REMARK 250 Domainome construct {dom} crop of {fname} ({c['src_path']})\n"
           f"REMARK 250 residues {start+shift}-{end+shift} in UniProt {acc} numbering (Domainome position + {shift}); AF residue = UniProt - {off}\n"
           f"REMARK 250 B-factor column = AlphaFold pLDDT\n")
    open(cpath, "w").write(hdr + new.make_pdb_string(gemmi.PdbWriteOptions(minimal=True)))
    p = np.array([x[5] for x in res_pl])
    n_res_mm = sum(x[3] != x[4] for x in res_pl)
    rec.update(status="ok" if n_res_mm == 0 and mm == 0 else ("ok_shifted" if n_res_mm == 0 else "construct_mismatch"),
               n_res_mismatch=n_res_mm, af_full_path=os.path.relpath(fpath, ROOT),
               construct_path=os.path.relpath(cpath, ROOT), n_res_crop=len(res_pl),
               plddt_mean=float(p.mean()), plddt_min=float(p.min()),
               frac_plddt_lt70=float((p < 70).mean()), frac_plddt_lt50=float((p < 50).mean()),
               af_mean_plddt_full=float(np.nanmean(pl)))
    return rec, res_pl


def main():
    os.makedirs(FULL, exist_ok=True)
    os.makedirs(CROP, exist_ok=True)
    w = domains()
    # warm tar indices in parent so forked workers share them
    for lab, _ in afdb_io.TARS:
        afdb_io.tar_index(lab)
    afdb_io._af2()
    with Pool(32) as p:
        res = p.map(process, list(w.itertuples(name=None, index=False)), chunksize=4)
    recs = pd.DataFrame([r for r, _ in res])
    per = pd.DataFrame([x for _, rp in res if rp for x in rp],
                       columns=["domain_ID", "pdb_resnum", "domainome_position", "af_aa", "construct_aa", "plddt"])
    out = os.path.join(ROOT, "data/processed")
    recs.to_csv(os.path.join(out, "domainome_structures.csv"), index=False)
    per.to_csv(os.path.join(out, "domainome_residue_plddt.csv"), index=False)
    print(recs.status.value_counts())
    print(recs.af_source.value_counts(dropna=False))
    print(recs[recs.status != "ok"][["domain_ID", "status", "af_source", "construct_mismatches", "construct_found_at", "seqpos_shift"]].to_string())
    print("residue aa mismatches (AF vs construct):", (per.af_aa != per.construct_aa).sum(), "of", len(per))


if __name__ == "__main__":
    main()
