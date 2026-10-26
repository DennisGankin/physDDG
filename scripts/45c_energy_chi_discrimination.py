"""45c: native vs perturbed side-chain chis — does each energy term prefer the native conformation?

For every well-ordered residue with >= 1 chi (no Pro/Gly/Ala; complete side chain; all side-chain B < 40 and
occupancy 1) in 30 FireProt X-ray structures, we generate perturbed side chains by rigidly rotating the atoms
downstream of each chi bond (bond geometry stays native):
  large: chi1 (and chi2 if present) rotated by +-120° (rotamer jumps), 4 variants
  small: every chi jittered by U(+-15..40°), 4 variants
Each variant is scored with energy.local_energy (moving residue vs all other residues fixed; one-body incl.
intra-residue) and compared with the native. For each term: fraction of (residue, variant) pairs where the
native is better (|dE| < 1e-4 kcal/mol = tie, excluded), mean dE; plus unit-weight totals.
Outputs: results/engine_energy/45c_chi_discrimination.csv, 45c_pairs.csv.gz
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physddg import energy as E  # noqa: E402
from physddg import topology as topo  # noqa: E402
from physddg.params import EnergyParams  # noqa: E402
from physddg.structure import load_protein  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "results/engine_energy"
rng = np.random.default_rng(0)

# chi axes and downstream slot masks per aa
CHI = {}
for aa in topo.AA1:
    s = {nm: i for i, nm in enumerate(E.slot_names(aa)) if nm}
    CHI[aa] = [(s[q[1]], s[q[2]], sorted(E._downstream(aa, q[1], q[2]))) for q in topo.CHI_ATOMS[aa]]


def rotate(xyz: torch.Tensor, aa: str, deltas: torch.Tensor) -> torch.Tensor:
    """xyz [B, A, 3]; deltas [B, nchi] degrees; rotate downstream atoms about each chi bond (in order)."""
    xyz = xyz.clone()
    for k, (ib, ic, down) in enumerate(CHI[aa]):
        th = torch.deg2rad(deltas[:, k])[:, None]
        o = xyz[:, ic][:, None]
        ax = E._unit(xyz[:, ic] - xyz[:, ib])[:, None]
        v = xyz[:, down] - o
        c, s_ = torch.cos(th)[..., None], torch.sin(th)[..., None]
        v = v * c + torch.cross(ax.expand_as(v), v, dim=-1) * s_ + ax * (v * ax).sum(-1, keepdim=True) * (1 - c)
        xyz[:, down] = o + v
    return xyz


def perturbations(nchi: int):
    large = []
    for s1 in (120, -120):
        large.append([s1] + [0] * (nchi - 1))
    if nchi >= 2:
        large += [[0, 120] + [0] * (nchi - 2), [0, -120] + [0] * (nchi - 2)]
    else:
        large += [[60] , [-60]]
    small = [list(rng.choice([-1, 1], nchi) * rng.uniform(15, 40, nchi)) for _ in range(4)]
    return np.array(large, float), np.array(small, float)


def run(params: EnergyParams, files, tag=""):
    rows = []
    for f in files:
        prot = load_protein(f)
        b = E.blocks_from_protein(prot, dev, params)
        for i, aa in enumerate(prot.seq):
            nchi = topo.N_CHI[aa]
            na = len(topo.TEMPLATES[aa].atoms)
            if aa in "PGA" or nchi == 0 or prot.missing_sc[i]:
                continue
            if (prot.b14[i, 4:na] > 40).any() or (prot.occ14[i, 4:na] < 1).any():
                continue
            large, small = perturbations(nchi)
            dl = torch.tensor(np.concatenate([np.zeros((1, nchi)), large, small]), dtype=torch.float32, device=dev)
            idx = torch.full((len(dl),), i, device=dev)
            mov = b[idx]
            mov.xyz = rotate(mov.xyz, aa, dl)
            env = E.make_env(b, [i], params)
            with torch.no_grad():
                one, _ = E.local_energy(mov[:, None], env, params)
            e = one[:, 0].cpu().numpy()
            for v in range(1, len(dl)):
                rows.append([f.stem, i, aa, "large" if v <= len(large) else "small"] + list(e[v] - e[0]))
    names = E.term_names(params)
    return pd.DataFrame(rows, columns=["pdb", "i", "aa", "kind"] + [f"d_{n}" for n in names])


def summarise(df: pd.DataFrame, names, tag):
    df = df.copy()
    df["d_lj"] = df.d_lj_atr + df.d_lj_rep
    df["d_solv"] = df.d_solv_apolar + df.d_solv_polar
    df["d_hb"] = df.d_hb_sc_bb + df.d_hb_sc_sc + df.d_hb_bb_bb
    df["d_total_unit"] = df.d_lj + df.d_solv + df.d_hb + df.d_elec_dh + df.d_disulf
    df["d_total_norep"] = df.d_total_unit - df.d_lj_rep
    out = []
    terms = [f"d_{n}" for n in names] + ["d_lj", "d_solv", "d_hb", "d_total_unit", "d_total_norep"]
    # clash-free perturbations: does attraction/solvation/H-bonding alone recognise the native?
    nc = df[df.d_lj_rep < 0.5].copy()
    nc["kind"] = nc["kind"] + "_noclash"
    for kind, sub in pd.concat([df, nc]).groupby("kind"):
        for t in terms:
            x = sub[t].to_numpy()
            nz = np.abs(x) > 1e-4
            out.append(dict(setting=tag, kind=kind, term=t[2:], frac_native_better=(x[nz] > 0).mean() if nz.any() else np.nan,
                            frac_nontie=nz.mean(), mean_dE=x.mean(), median_dE=np.median(x), n=len(x)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    files = []
    for f in sorted(Path("/workspace/data/fireprot_pdbs").glob("*.pdb")):
        txt = f.read_text(errors="ignore")
        if "X-RAY" in txt[:5000] and 60 <= sum(1 for ln in txt.splitlines() if ln.startswith("ATOM") and ln[12:16] == " CA ") <= 450:
            files.append(f)
        if len(files) >= 30:
            break
    settings = {"default": EnergyParams(), "w14=0": EnergyParams(w14=0.0), "w14=1": EnergyParams(w14=1.0),
                "cap=1": EnergyParams(rep_cap=1.0), "cap=10": EnergyParams(rep_cap=10.0),
                "rep_scale=1.0": EnergyParams(rep_scale=1.0), "rep_scale=0.8": EnergyParams(rep_scale=0.8)}
    summ = []
    for tag, prm in settings.items():
        df = run(prm, files)
        if tag == "default":
            df.to_csv(OUT / "45c_pairs.csv.gz", index=False)
            print(f"{df.groupby(['pdb', 'i']).ngroups} residues, {len(df)} perturbations")
        summ.append(summarise(df, E.term_names(prm), tag))
        print(tag, "done", flush=True)
    s = pd.concat(summ)
    s.to_csv(OUT / "45c_chi_discrimination.csv", index=False)
    piv = s.pivot_table(index="term", columns=["setting", "kind"], values="frac_native_better").round(3)
    print(piv.to_string())
    print(s[s.setting == "default"].pivot_table(index="term", columns="kind", values=["mean_dE", "frac_nontie"]).round(3).to_string())
