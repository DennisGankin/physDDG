#!/usr/bin/env python
"""Compare AF DB v4 and v6 models over each Domainome construct (human domains whose primary model is v4).

v6 re-predicted the human proteome, so the two releases are independent AF models of the same sequence: their
disagreement over a construct is a direct measure of AF model uncertainty there. Writes
  data/structures/domainome/af_construct_v6/<domain_ID>.pdb  (v6 crop, same conventions as af_construct/)
  data/processed/domainome_af_v4_v6.csv: CA RMSD (superposed over the construct), all-heavy-atom RMSD of identical
  atoms after the CA superposition, pLDDT means of both, and their per-construct difference.
"""
import os, sys
import numpy as np
import pandas as pd
import gemmi
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afdb_io

ROOT = afdb_io.ROOT
OUTD = os.path.join(ROOT, "data/structures/domainome/af_construct_v6")


def atoms(st, lo, hi, off):
    out = {}
    for r in st[0][0]:
        u = r.seqid.num + off
        if lo <= u <= hi:
            for a in r:
                out[(u, a.name)] = (np.array(a.pos.tolist()), a.b_iso)
    return out


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, P.mean(0), Q.mean(0)


def run(t):
    dom, acc, lo, hi, v4path, off4 = t
    c6 = next((c for c in afdb_io.candidates(acc, sources=("v6_human",), ebi=False)
               if lo - c["uniprot_offset"] >= 1 and hi - c["uniprot_offset"] <= len(afdb_io.structure_seq(c["text"])[0])), None)
    if c6 is None:
        return dict(domain_ID=dom, v6_status="no_v6_model")
    s4 = gemmi.read_structure(os.path.join(ROOT, v4path))
    s6 = gemmi.read_pdb_string(c6["text"])
    a4 = atoms(s4, lo, hi, 0)  # construct crop is already in UniProt numbering
    a6 = atoms(s6, lo, hi, c6["uniprot_offset"])
    r4 = {r.seqid.num: r.name for r in s4[0][0]}
    r6 = {r.seqid.num + c6["uniprot_offset"]: r.name for r in s6[0][0] if lo <= r.seqid.num + c6["uniprot_offset"] <= hi}
    n_seq_diff = sum(r4.get(u) != r6.get(u) for u in range(lo, hi + 1))
    if n_seq_diff:  # v6 sequence (UniProt update) or fragmentation differs over the construct
        return dict(domain_ID=dom, v6_status="v6_seq_differs", n_seq_diff=n_seq_diff)
    ca = sorted(k for k in a4 if k[1] == "CA" and k in a6)
    P = np.array([a4[k][0] for k in ca]); Q = np.array([a6[k][0] for k in ca])
    R, pm, qm = kabsch(P, Q)
    ca_rmsd = float(np.sqrt((((P - pm) @ R - (Q - qm)) ** 2).sum(1).mean()))
    common = sorted(k for k in a4 if k in a6)
    P2 = np.array([a4[k][0] for k in common]); Q2 = np.array([a6[k][0] for k in common])
    ha_rmsd = float(np.sqrt((((P2 - pm) @ R - (Q2 - qm)) ** 2).sum(1).mean()))
    pl4 = np.array([a4[k][1] for k in ca]); pl6 = np.array([a6[k][1] for k in ca])
    # write v6 crop
    new = gemmi.Structure(); nm = gemmi.Model("1"); nc = gemmi.Chain("A")
    for r in s6[0][0]:
        u = r.seqid.num + c6["uniprot_offset"]
        if lo <= u <= hi:
            r2 = r.clone(); r2.seqid = gemmi.SeqId(u, " "); nc.add_residue(r2)
    nm.add_chain(nc); new.add_model(nm); new.setup_entities()
    open(os.path.join(OUTD, f"{dom}.pdb"), "w").write(
        f"REMARK 250 Domainome construct {dom}, AF DB v6 crop ({c6['src_path']}), UniProt numbering, B = pLDDT\n"
        + new.make_pdb_string(gemmi.PdbWriteOptions(minimal=True)))
    return dict(domain_ID=dom, v6_status="ok", n_seq_diff=0, n_ca=len(ca), ca_rmsd_v4_v6=ca_rmsd, heavy_rmsd_v4_v6=ha_rmsd,
                plddt_v4=float(pl4.mean()), plddt_v6=float(pl6.mean()),
                frac_lt70_v4=float((pl4 < 70).mean()), frac_lt70_v6=float((pl6 < 70).mean()),
                v6_construct=os.path.relpath(os.path.join(OUTD, f"{dom}.pdb"), ROOT))


def main():
    os.makedirs(OUTD, exist_ok=True)
    s = pd.read_csv(os.path.join(ROOT, "data/processed/domainome_structures.csv"))
    s = s[(s.af_version == 4) & s.construct_path.notna()]
    s["seqpos_shift"] = s.seqpos_shift.fillna(0).astype(int)
    jobs = [(r.domain_ID, r.uniprot, r.construct_start + r.seqpos_shift, r.construct_end + r.seqpos_shift,
             r.construct_path, r.af_frag_offset) for r in s.itertuples()]
    afdb_io.tar_index("v6_human")
    with Pool(32) as p:
        res = p.map(run, jobs, chunksize=2)
    out = pd.DataFrame(res)
    out.to_csv(os.path.join(ROOT, "data/processed/domainome_af_v4_v6.csv"), index=False)
    ok = out[out.v6_status == "ok"]
    print(out.v6_status.value_counts().to_dict())
    print(ok[["ca_rmsd_v4_v6", "heavy_rmsd_v4_v6", "plddt_v4", "plddt_v6"]].describe(percentiles=[.5, .75, .9, .95]).round(2).to_string())
    print("constructs with CA RMSD > 1 / 2 / 4 A:", (ok.ca_rmsd_v4_v6 > 1).sum(), (ok.ca_rmsd_v4_v6 > 2).sum(), (ok.ca_rmsd_v4_v6 > 4).sum())
    print("spearman(ca_rmsd, plddt_v4):", round(ok.ca_rmsd_v4_v6.corr(ok.plddt_v4, method="spearman"), 3))


if __name__ == "__main__":
    main()
