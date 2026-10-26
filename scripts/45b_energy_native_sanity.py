"""45b: sanity of the energy terms on native crystal structures.

(i)  per-residue energies (decomposition: one-body + half pair) by residue type and burial class
     (rSA < 0.2 buried, > 0.5 exposed; freesasa via structure.annotate); solvation reported relative to the
     EEF1 reference (desolvation on folding = per-residue solvation - sum dG_ref of the residue's atoms)
(ii) backbone H-bonds (our DSSP-model term, amide H on bisector, E < -0.5) vs structure.dssp() (Kabsch-Sander
     re-implementation by E1, DSSP-style H): precision / recall over (acceptor i, donor j) pairs
(iii) EEF1 per-atom solvation vs freesasa atom SASA: Spearman of dG_i/dG_ref_i with relative atom SASA
Proteins: the first 30 X-ray structures (60-450 residues) of /workspace/data/fireprot_pdbs.
Outputs: results/engine_energy/45b_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physddg import energy as E  # noqa: E402
from physddg.params import EnergyParams  # noqa: E402
from physddg.structure import annotate, backbone_h, load_protein  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
p = EnergyParams()
OUT = ROOT / "results/engine_energy"
OUT.mkdir(parents=True, exist_ok=True)


def xray_list(n=30):
    out = []
    for f in sorted(Path("/workspace/data/fireprot_pdbs").glob("*.pdb")):
        txt = f.read_text(errors="ignore")
        if "X-RAY" not in txt[:5000]:
            continue
        nca = sum(1 for ln in txt.splitlines() if ln.startswith("ATOM") and ln[12:16] == " CA ")
        if 60 <= nca <= 450:
            out.append(f)
        if len(out) >= n:
            break
    return out


def dssp_map(prot):
    """Kabsch-Sander reference map E[i, j] (C=O of i <- N-H of j), DSSP-style H (structure.backbone_h),
    CA-CA < 9 Å, |i-j| >= 2, as in structure.dssp()."""
    n, ca, c, o = (prot.xyz14[:, k].astype(np.float64) for k in range(4))
    h = backbone_h(prot, dssp_style=True)
    r = lambda x, y: np.linalg.norm(x[:, None] - y[None], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Em = 0.084 * 332.0 * (1 / r(o, n) + 1 / r(c, h) - 1 / r(o, h) - 1 / r(c, n))
    idx = np.arange(len(prot))
    ok = np.isfinite(Em) & (r(ca, ca) < 9.0) & (np.abs(idx[:, None] - idx[None]) >= 2)
    return np.where(ok, Em, 0.0)


rows, hb_rows, atom_rows = [], [], []
for f in xray_list():
    prot = annotate(load_protein(f))
    L = len(prot)
    if not (60 <= L <= 450):
        continue
    b = E.blocks_from_protein(prot, dev, p)
    with torch.no_grad():
        d = E.decompose(b, p)
        fe = E.features(b, p)
        per = d.per_residue.cpu().numpy()
        ref = (fe["dg_ref"] * fe["m"].float())
        ref_ap = (ref * (1 - fe["polar"])).sum(-1).cpu().numpy()
        ref_po = (ref * fe["polar"]).sum(-1).cpu().numpy()
        # (ii) directional backbone H-bond map: E[i, j] = C=O(i) <- N-H(j)
        fd = {k: v[None] for k, v in fe.items()}
        fa = {k: v[:, None] for k, v in fe.items()}
        nonadj = (fe["pos"][None, :] - fe["pos"][:, None]).abs() >= 2
        Emine = E._dssp_pair(fd, fa, nonadj, p).cpu().numpy()
        solv = E.atom_solvation(b, p).cpu().numpy()
    Eref = dssp_map(prot)
    mine, refm = Emine < -0.5, Eref < -0.5
    hb_rows.append(dict(pdb=f.stem, L=L, n_dssp=int(refm.sum()), n_mine=int(mine.sum()), tp=int((mine & refm).sum())))
    for i in range(L):
        r = dict(pdb=f.stem, i=i, aa=prot.seq[i], rsa=float(prot.ann["rsa"][i]), ss=prot.ann["ss3"][i])
        r.update({n: float(per[i, k]) for k, n in enumerate(d.names)})
        r["dsolv_apolar"] = r["solv_apolar"] - ref_ap[i]
        r["dsolv_polar"] = r["solv_polar"] - ref_po[i]
        rows.append(r)
    # (iii) per-atom
    sas = prot.ann["sasa_atom14"]
    dgr = fe["dg_ref"].cpu().numpy()
    pol = fe["polar"].cpu().numpy()
    m = prot.mask14
    for i in range(L):
        for a in range(14):
            if m[i, a] and abs(dgr[i, a]) > 0.3:
                atom_rows.append((f.stem, prot.seq[i], a, pol[i, a], solv[i, a] / dgr[i, a], sas[i, a]))
    print(f.stem, L, "dssp", int(refm.sum()), "mine", int(mine.sum()), "tp", int((mine & refm).sum()), flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "45b_per_residue.csv.gz", index=False)
hb = pd.DataFrame(hb_rows)
hb.to_csv(OUT / "45b_hbond_vs_dssp.csv", index=False)
prec, rec = hb.tp.sum() / hb.n_mine.sum(), hb.tp.sum() / hb.n_dssp.sum()
print(f"\nbackbone H-bonds vs DSSP: precision {prec:.3f} recall {rec:.3f} (n_dssp {hb.n_dssp.sum()}, {len(hb)} proteins)")

at = pd.DataFrame(atom_rows, columns=["pdb", "aa", "slot", "polar", "solv_frac", "sasa"])
for nm, sub in (("polar", at[at.polar == 1]), ("apolar", at[at.polar == 0])):
    rho = spearmanr(sub.solv_frac, sub.sasa)[0]
    print(f"EEF1 dG_i/dG_ref_i vs atom SASA ({nm} atoms, n={len(sub)}): Spearman {rho:.3f}")
# residue level: side-chain-free, total desolvation vs rSA
for c in ("dsolv_apolar", "dsolv_polar"):
    print(f"per-residue EEF1 {c} (rel. to ref, decomposition) vs rSA: Spearman {spearmanr(res[c], res.rsa)[0]:.3f}")

res["burial"] = np.where(res.rsa < 0.2, "buried", np.where(res.rsa > 0.5, "exposed", "mid"))
res["lj"] = res.lj_atr + res.lj_rep
res["hb"] = res.hb_sc_bb + res.hb_sc_sc + res.hb_bb_bb
cols = ["lj", "lj_atr", "lj_rep", "dsolv_apolar", "dsolv_polar", "hb", "elec_dh", "elec_dd"]
tab = res[res.burial != "mid"].groupby(["aa", "burial"])[cols].mean().round(2)
tab["n"] = res[res.burial != "mid"].groupby(["aa", "burial"]).size()
tab.to_csv(OUT / "45b_by_aa_burial.csv")
print(tab.to_string())
grp = {"hydrophobic": set("AILMFVWC"), "charged": set("DEKR"), "polar": set("STNQHY"), "GP": set("GP")}
res["cls"] = res.aa.map(lambda a: next(k for k, v in grp.items() if a in v))
print(res[res.burial != "mid"].groupby(["cls", "burial"])[cols].mean().round(2).to_string())
