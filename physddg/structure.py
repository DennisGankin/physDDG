"""Structure loading and per-residue geometry descriptors.

`load_protein(path)` -> `Protein` (numpy arrays). Handles PDB / mmCIF (gz ok) via gemmi, first model only,
waters and hydrogens dropped, highest-occupancy altloc kept per residue (one consistent conformer),
common modified residues mapped to their parent amino acid (MSE->MET with SE->SD, phospho/oxidised/
methylated variants: extra atoms dropped), other HETATM groups (metals, ligands) optionally kept separately.

Residue numbering: `resnum` = auth seq id + insertion code as string ("52A"). Chains: auth chain id.
Dense layout: `xyz14[L,14,3]` / `mask14[L,14]` in topology template order (N, CA, C, O, side chain);
`oxt[L,3]` (NaN where absent). Flat per-atom arrays (`atom_*`) list present heavy atoms incl. OXT.

`annotate(protein)` adds phi/psi/omega, cis flags, chain breaks, disulfides, DSSP secondary structure
(Kabsch & Sander 1983 re-implementation: H G I E B T S, 3-state H/E/L), smooth CB contact number, rSASA (freesasa, ProtOr radii,
Tien et al. 2013 theoretical max ASA), CB / centroid neighbour counts (8/10/12 A) and HSE-beta (13 A).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np

from .topology import AA1, MAX_ATOMS, TEMPLATES, THREE_TO_ONE, TYPE_INDEX, atom_type

MODIFIED = {  # modified residue -> parent (atoms renamed per RENAME, non-template atoms dropped)
    "MSE": "MET", "SEP": "SER", "TPO": "THR", "PTR": "TYR", "CSO": "CYS", "CSD": "CYS", "CME": "CYS",
    "CSS": "CYS", "OCS": "CYS", "CSX": "CYS", "CAS": "CYS", "SMC": "CYS", "CYX": "CYS", "CYM": "CYS",
    "HYP": "PRO", "MLY": "LYS", "M3L": "LYS", "MLZ": "LYS", "KCX": "LYS", "ALY": "LYS", "LYN": "LYS",
    "HIC": "HIS", "HIP": "HIS", "HID": "HIS", "HIE": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "NEP": "HIS", "ASH": "ASP", "GLH": "GLU", "PCA": "GLN", "MEN": "ASN", "SAC": "SER", "DAH": "PHE",
    "TYS": "TYR", "AGM": "ARG", "CGU": "GLU", "LLP": "LYS", "TRQ": "TRP", "FME": "MET", "MHO": "MET",
}
RENAME = {("MSE", "SE"): "SD", ("PCA", "OE"): "OE1"}
MAX_ASA_TIEN = {  # Tien et al. 2013 PLoS One, theoretical
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "E": 223.0, "Q": 225.0, "G": 104.0,
    "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0, "S": 155.0,
    "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}
WATER = {"HOH", "WAT", "DOD", "H2O"}


@dataclass
class Protein:
    path: str
    chain: np.ndarray            # [L] str
    resnum: np.ndarray           # [L] str, auth seq id + icode
    seq: str                     # 1-letter, length L
    aa: np.ndarray               # [L] int index into AA1
    resname: np.ndarray          # [L] original 3-letter name
    modified: np.ndarray         # [L] bool (mapped modified residue)
    xyz14: np.ndarray            # [L,14,3] float32, NaN where missing
    mask14: np.ndarray           # [L,14] bool
    b14: np.ndarray              # [L,14] B-factor (pLDDT for AF models)
    occ14: np.ndarray            # [L,14] occupancy
    oxt: np.ndarray              # [L,3] NaN unless C-terminal OXT present
    missing_sc: np.ndarray       # [L] bool: some side-chain heavy atoms absent (rebuild)
    missing_o: np.ndarray        # [L] bool
    altloc: np.ndarray           # [L] bool: residue had alternative locations (highest-occupancy one kept)
    het_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    het_name: np.ndarray = field(default_factory=lambda: np.zeros(0, "U4"))
    het_resname: np.ndarray = field(default_factory=lambda: np.zeros(0, "U3"))
    het_element: np.ndarray = field(default_factory=lambda: np.zeros(0, "U2"))
    ann: dict = field(default_factory=dict)   # filled by annotate()

    def __len__(self) -> int:
        return len(self.seq)

    @property
    def bfactor(self) -> np.ndarray:
        """Per-residue CA B-factor (= pLDDT for AlphaFold models)."""
        return self.b14[:, 1]

    @property
    def keys(self) -> list[tuple[str, str]]:
        return list(zip(self.chain.tolist(), self.resnum.tolist()))

    def index(self, chain: str, resnum: str) -> int:
        hit = np.flatnonzero((self.chain == chain) & (self.resnum == str(resnum)))
        if len(hit) != 1:
            raise KeyError(f"{chain}:{resnum} not found (or ambiguous) in {self.path}")
        return int(hit[0])

    # flat per-atom view (present atoms only, template order, OXT appended after its residue)
    def atoms(self) -> dict[str, np.ndarray]:
        res, names, xyz, types, b = [], [], [], [], []
        for i, a in enumerate(self.seq):
            t = TEMPLATES[a]
            for k, nm in enumerate(t.atoms):
                if self.mask14[i, k]:
                    res.append(i); names.append(nm); xyz.append(self.xyz14[i, k])
                    types.append(t.type_ids[k]); b.append(self.b14[i, k])
            if not np.isnan(self.oxt[i, 0]):
                res.append(i); names.append("OXT"); xyz.append(self.oxt[i])
                types.append(TYPE_INDEX[atom_type(a, "OXT")]); b.append(self.b14[i, 2])
        return {"res_index": np.array(res, np.int64), "name": np.array(names), "xyz": np.array(xyz, np.float32),
                "type_id": np.array(types, np.int64), "bfactor": np.array(b, np.float32)}


# ----------------------------------------------------------------------------- loading
def _pick_altloc(res: gemmi.Residue) -> str:
    """Altloc with the highest summed occupancy in this residue ('' if none)."""
    occ: dict[str, float] = {}
    for at in res:
        if at.altloc != "\0":
            occ[at.altloc] = occ.get(at.altloc, 0.0) + at.occ
    return max(occ, key=occ.get) if occ else ""


def load_protein(path: str | Path, chains: str | list[str] | None = None, keep_het: bool = True,
                 min_occ: float = 0.0) -> Protein:
    """Load the first model of a PDB/mmCIF file. `chains` restricts to these auth chain ids."""
    st = gemmi.read_structure(str(path), merge_chain_parts=True)
    st.setup_entities()
    model = st[0]
    want = None if chains is None else set(chains)
    rows = []
    het = []
    for ch in model:
        if want is not None and ch.name not in want:
            continue
        for res in ch:
            rn = res.name.strip()
            if rn in WATER:
                continue
            parent = rn if rn in THREE_TO_ONE else MODIFIED.get(rn)
            alt = _pick_altloc(res)
            atoms = [a for a in res if a.element.name not in ("H", "D") and (a.altloc == "\0" or a.altloc == alt)
                     and a.occ >= min_occ]
            if parent is None or not any(a.name == "CA" for a in atoms):
                if keep_het and (res.het_flag == "H" or parent is None):
                    het += [(a.pos.x, a.pos.y, a.pos.z, a.name, rn, a.element.name) for a in atoms]
                continue
            rows.append((ch.name, f"{res.seqid.num}{res.seqid.icode.strip()}", rn, parent, atoms, alt != ""))

    L = len(rows)
    xyz = np.full((L, MAX_ATOMS, 3), np.nan, np.float32)
    mask = np.zeros((L, MAX_ATOMS), bool)
    bf = np.zeros((L, MAX_ATOMS), np.float32)
    oc = np.zeros((L, MAX_ATOMS), np.float32)
    oxt = np.full((L, 3), np.nan, np.float32)
    seq, keep = [], np.ones(L, bool)
    for i, (_, _, rn, parent, atoms, _) in enumerate(rows):
        a1 = THREE_TO_ONE[parent]
        seq.append(a1)
        idx = {n: k for k, n in enumerate(TEMPLATES[a1].atoms)}
        for at in atoms:
            nm = RENAME.get((rn, at.name), at.name)
            if nm == "OXT":
                oxt[i] = (at.pos.x, at.pos.y, at.pos.z)
            elif nm in idx:
                k = idx[nm]
                xyz[i, k] = (at.pos.x, at.pos.y, at.pos.z)
                mask[i, k] = True
                bf[i, k] = at.b_iso
                oc[i, k] = at.occ
        keep[i] = mask[i, :3].all()          # need N, CA, C to model a residue
    sel = np.flatnonzero(keep)
    seq_s = "".join(seq[i] for i in sel)
    nat = np.array([len(TEMPLATES[a].atoms) for a in seq_s], np.int64)
    m = mask[sel]
    missing_sc = np.array([not m[j, 4:nat[j]].all() for j in range(len(sel))], bool)
    het_arr = np.array([h[:3] for h in het], np.float32).reshape(-1, 3)
    return Protein(
        path=str(path),
        chain=np.array([rows[i][0] for i in sel]), resnum=np.array([rows[i][1] for i in sel]),
        seq=seq_s, aa=np.array([AA1.index(a) for a in seq_s], np.int64),
        resname=np.array([rows[i][2] for i in sel]),
        modified=np.array([rows[i][2] != rows[i][3] for i in sel], bool),
        xyz14=xyz[sel], mask14=m, b14=bf[sel], occ14=oc[sel], oxt=oxt[sel],
        missing_sc=missing_sc, missing_o=~m[:, 3], altloc=np.array([rows[i][5] for i in sel], bool),
        het_xyz=het_arr, het_name=np.array([h[3] for h in het], "U4"),
        het_resname=np.array([h[4] for h in het], "U3"), het_element=np.array([h[5] for h in het], "U2"),
    )


# ----------------------------------------------------------------------------- geometry helpers
def dihedral(p0, p1, p2, p3) -> np.ndarray:
    """Dihedral angle in degrees for arrays [...,3] (NaN propagates)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, -1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, -1, keepdims=True) * b1n
    x = np.sum(v * w, -1)
    y = np.sum(np.cross(b1n, v) * w, -1)
    return np.degrees(np.arctan2(y, x))


def ideal_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ideal CB from backbone N, CA, C ([...,3]); same internal coordinates as the side-chain builder."""
    from .topology import CB_IC
    return place_atom(c, n, ca, CB_IC[0], CB_IC[1], CB_IC[2])


def place_atom(a, b, c, bond: float, angle_deg, torsion_deg) -> np.ndarray:
    """NeRF (numpy): D with |CD|=bond, angle BCD, dihedral ABCD."""
    th, ph = np.radians(angle_deg), np.radians(torsion_deg)
    bc = c - b
    bc = bc / np.linalg.norm(bc, axis=-1, keepdims=True)
    n = np.cross(b - a, bc)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    m = np.cross(n, bc)
    d = np.stack([-bond * np.cos(th) * np.ones_like(ph), bond * np.sin(th) * np.cos(ph),
                  bond * np.sin(th) * np.sin(ph)], -1)
    return c + d[..., :1] * bc + d[..., 1:2] * m + d[..., 2:3] * n


def backbone_h(p: Protein, dssp_style: bool = False) -> np.ndarray:
    """Amide H positions [L,3] (NaN for Pro, first residue and after chain breaks).

    Default: H on the external bisector of C(i-1)-N(i) and CA(i)-N(i), 1.01 A.
    dssp_style: N + unit(C(i-1) - O(i-1)) (the Kabsch-Sander convention)."""
    n, ca, c, o = (p.xyz14[:, k] for k in range(4))
    h = np.full_like(n, np.nan)
    ok = np.zeros(len(p), bool)
    ok[1:] = ~breaks(p)[1:]
    ok &= p.aa != AA1.index("P")
    cp, op = np.roll(c, 1, 0), np.roll(o, 1, 0)
    if dssp_style:
        v = cp - op
        h_all = n + v / np.linalg.norm(v, axis=-1, keepdims=True)
    else:
        u1 = n - cp
        u2 = n - ca
        v = u1 / np.linalg.norm(u1, axis=-1, keepdims=True) + u2 / np.linalg.norm(u2, axis=-1, keepdims=True)
        h_all = n + 1.01 * v / np.linalg.norm(v, axis=-1, keepdims=True)
    h[ok] = h_all[ok]
    return h


def breaks(p: Protein, max_cn: float = 2.0) -> np.ndarray:
    """[L] True if residue i is not peptide-bonded to i-1 (new chain or C(i-1)-N(i) > max_cn)."""
    out = np.ones(len(p), bool)
    if len(p) > 1:
        d = np.linalg.norm(p.xyz14[1:, 0] - p.xyz14[:-1, 2], axis=-1)
        out[1:] = (p.chain[1:] != p.chain[:-1]) | ~(d <= max_cn)
    return out


def backbone_dihedrals(p: Protein) -> dict[str, np.ndarray]:
    """phi, psi, omega (omega of the peptide bond preceding residue i) in degrees; cis flag (|omega|<30)."""
    n, ca, c = p.xyz14[:, 0], p.xyz14[:, 1], p.xyz14[:, 2]
    L = len(p)
    br = breaks(p)
    phi = np.full(L, np.nan, np.float32)
    psi = np.full(L, np.nan, np.float32)
    omg = np.full(L, np.nan, np.float32)
    if L > 1:
        ok = ~br[1:]
        phi[1:] = np.where(ok, dihedral(c[:-1], n[1:], ca[1:], c[1:]), np.nan)
        psi[:-1] = np.where(ok, dihedral(n[:-1], ca[:-1], c[:-1], n[1:]), np.nan)
        omg[1:] = np.where(ok, dihedral(ca[:-1], c[:-1], n[1:], ca[1:]), np.nan)
    return {"phi": phi, "psi": psi, "omega": omg, "cis": np.abs(omg) < 30.0, "chain_break": br}


def side_chain_chis(p: Protein) -> np.ndarray:
    """[L,4] native chi angles in degrees (NaN where undefined or atoms missing)."""
    from .topology import CHI_ATOMS
    out = np.full((len(p), 4), np.nan, np.float32)
    for a in set(p.seq):
        rows = np.flatnonzero(p.aa == AA1.index(a))
        t = TEMPLATES[a]
        for k, quad in enumerate(CHI_ATOMS[a]):
            pts = [p.xyz14[rows, t.index(nm)] for nm in quad]
            out[rows, k] = dihedral(*pts)
    return out


def disulfides(p: Protein, max_d: float = 2.5) -> list[tuple[int, int]]:
    cys = np.flatnonzero((p.aa == AA1.index("C")) & p.mask14[:, 5])
    if len(cys) < 2:
        return []
    sg = p.xyz14[cys, 5]
    d = np.linalg.norm(sg[:, None] - sg[None], axis=-1)
    i, j = np.nonzero(np.triu(d < max_d, 1))
    return [(int(cys[a]), int(cys[b])) for a, b in zip(i, j)]


# ----------------------------------------------------------------------------- DSSP (Kabsch & Sander 1983)
def dssp(p: Protein) -> dict[str, np.ndarray]:
    """8-state (H G I E B T S -) and 3-state (H E L) secondary structure (sparse, O(L) memory).

    Classic DSSP priority H > E/B > G > I > T > S; beta bulges are not merged into ladders (minor differences)."""
    from scipy.spatial import cKDTree
    L = len(p)
    n, ca, c, o = (p.xyz14[:, k].astype(np.float64) for k in range(4))
    h = backbone_h(p, dssp_style=True)
    seg = np.cumsum(breaks(p))                       # contiguous segment id
    pairs = cKDTree(ca).query_pairs(9.0, output_type="ndarray")
    pairs = np.concatenate([pairs, pairs[:, ::-1]]) if len(pairs) else np.zeros((0, 2), np.int64)
    i, j = pairs[:, 0], pairs[:, 1]                  # C=O of i accepts from N-H of j
    keep = np.abs(i - j) >= 2
    i, j = i[keep], j[keep]
    d = lambda x, y: np.linalg.norm(x[i] - y[j], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        E = 0.084 * 332.0 * (1 / d(o, n) + 1 / d(c, h) - 1 / d(o, h) - 1 / d(c, n))
    ok = np.isfinite(E) & (E < -0.5)
    hb_codes = np.unique(i[ok] * L + j[ok])

    def HB(a_, b_):
        a_, b_ = np.broadcast_arrays(np.asarray(a_), np.asarray(b_))
        valid = (a_ >= 0) & (a_ < L) & (b_ >= 0) & (b_ < L)
        return valid & np.isin(np.where(valid, a_ * L + b_, -1), hb_codes)

    idx = np.arange(L)
    turn = {}
    for k in (3, 4, 5):
        t = np.zeros(L, bool)
        ii = idx[: max(L - k, 0)]
        t[ii] = HB(ii, ii + k) & (seg[ii] == seg[ii + k])
        turn[k] = t
    helix = {}
    for k in (4, 3, 5):
        hx = np.zeros(L, bool)
        for s0 in np.flatnonzero(turn[k][1:] & turn[k][:-1]) + 1:
            hx[s0:s0 + k] = True
        helix[k] = hx
    # bridges over candidate pairs (|i-j| > 2, CA < 9 A)
    bi, bj = pairs[:, 0], pairs[:, 1]
    far = np.abs(bi - bj) > 2
    bi, bj = bi[far], bj[far]
    par = (HB(bi - 1, bj) & HB(bj, bi + 1)) | (HB(bj - 1, bi) & HB(bi, bj + 1))
    anti = (HB(bi, bj) & HB(bj, bi)) | (HB(bi - 1, bj + 1) & HB(bj - 1, bi + 1))
    bridge = np.zeros(L, bool)
    bridge[bi[par | anti]] = True
    pc = set((bi[par] * L + bj[par]).tolist())
    ac = set((bi[anti] * L + bj[anti]).tolist())
    ladder = np.zeros(L, bool)
    for x, y in zip(bi[par], bj[par]):
        if ((x - 1) * L + (y - 1)) in pc or ((x + 1) * L + (y + 1)) in pc:
            ladder[x] = True
    for x, y in zip(bi[anti], bj[anti]):
        if ((x - 1) * L + (y + 1)) in ac or ((x + 1) * L + (y - 1)) in ac:
            ladder[x] = True
    bend = np.zeros(L, bool)
    if L > 4:
        v1 = ca[2:-2] - ca[:-4]
        v2 = ca[4:] - ca[2:-2]
        cosang = np.sum(v1 * v2, -1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
        bend[2:-2] = (np.degrees(np.arccos(np.clip(cosang, -1, 1))) > 70) & (seg[:-4] == seg[4:])
    tt = np.zeros(L, bool)
    for k in (3, 4, 5):
        for s0 in np.flatnonzero(turn[k]):
            tt[s0 + 1:s0 + k] = True
    ss = np.full(L, "-", "U1")
    for m_, ch in ((bend, "S"), (tt, "T"), (helix[5], "I"), (helix[3], "G"), (bridge, "B"),
                   (ladder, "E"), (helix[4], "H")):
        ss[m_] = ch
    ss3 = np.where(np.isin(ss, ["H", "G", "I"]), "H", np.where(np.isin(ss, ["E", "B"]), "E", "L"))
    return {"ss8": ss, "ss3": ss3}


# ----------------------------------------------------------------------------- burial descriptors
_CLASSIFIER = None


def _radius(resname: str, atom: str) -> float:
    global _CLASSIFIER
    import freesasa
    if _CLASSIFIER is None:
        _CLASSIFIER = freesasa.Classifier()
    r = _CLASSIFIER.radius(resname, atom)
    if r <= 0:
        r = {"C": 1.88, "N": 1.64, "O": 1.42, "S": 1.77}.get(atom[0], 1.8)
    return r


def sasa(p: Protein, probe: float = 1.4) -> dict[str, np.ndarray]:
    """Per-atom SASA [L,14] (Lee-Richards, freesasa, ProtOr radii) and per-residue absolute / relative SASA."""
    import freesasa
    from .topology import ONE_TO_THREE
    at = p.atoms()
    res3 = [ONE_TO_THREE[p.seq[i]] for i in at["res_index"]]
    radii = [_radius(r, nm) for r, nm in zip(res3, at["name"])]
    par = freesasa.Parameters({"algorithm": freesasa.LeeRichards, "probe-radius": probe, "n-slices": 20})
    res = freesasa.calcCoord(at["xyz"].astype(np.float64).ravel(), radii, par)
    a = np.array([res.atomArea(k) for k in range(len(radii))], np.float32)
    per_res = np.bincount(at["res_index"], weights=a, minlength=len(p)).astype(np.float32)
    atom14 = np.zeros((len(p), MAX_ATOMS), np.float32)
    names = at["name"]
    for k, (i, nm) in enumerate(zip(at["res_index"], names)):
        if nm != "OXT":
            atom14[i, TEMPLATES[p.seq[i]].index(nm)] = a[k]
    rel = per_res / np.array([MAX_ASA_TIEN[x] for x in p.seq], np.float32)
    sc = (atom14 * p.mask14 * np.array([[k >= 4 for k in range(MAX_ATOMS)]])).sum(1)
    return {"sasa_atom14": atom14, "sasa": per_res, "sasa_sc": sc, "rsa": rel}


def cb_coords(p: Protein) -> np.ndarray:
    """CB coordinates [L,3]; ideal CB for Gly and residues with missing CB."""
    cb = p.xyz14[:, 4].copy()
    need = ~p.mask14[:, 4]
    if need.any():
        cb[need] = ideal_cb(p.xyz14[need, 0], p.xyz14[need, 1], p.xyz14[need, 2])
    return cb


def smooth_cn(x, r0: float = 10.0, width: float = 1.0, exclude_self: bool = True):
    """Smooth contact number: sum_j sigmoid((r0 - d_ij) / width) over points x [N,3].

    torch tensor in -> differentiable dense torch computation; numpy in -> KD-tree (cutoff r0 + 12 width)."""
    if type(x).__module__.startswith("torch"):            # avoid importing torch for numpy callers
        import torch
        d = torch.cdist(x, x)
        w = torch.sigmoid((r0 - d) / width)
        return w.sum(-1) - (torch.sigmoid(torch.tensor(r0 / width, dtype=x.dtype, device=x.device))
                            if exclude_self else 0.0)
    from scipy.spatial import cKDTree
    x = np.asarray(x, np.float64)
    pr = cKDTree(x).query_pairs(r0 + 12 * width, output_type="ndarray")
    out = np.zeros(len(x))
    if len(pr):
        w = 1.0 / (1.0 + np.exp(-(r0 - np.linalg.norm(x[pr[:, 0]] - x[pr[:, 1]], axis=-1)) / width))
        np.add.at(out, pr[:, 0], w)
        np.add.at(out, pr[:, 1], w)
    if not exclude_self:
        out += 1.0 / (1.0 + np.exp(-r0 / width))
    return out.astype(np.float32)


def burial(p: Protein, radii=(8.0, 10.0, 12.0), hse_r: float = 13.0) -> dict[str, np.ndarray]:
    """CB and side-chain-centroid neighbour counts within radii, smooth CB contact number (10 A, width 1 A),
    and half-sphere exposure (HSE-beta: CA neighbours within 13 A in the CA->CB half-sphere = up)."""
    from scipy.spatial import cKDTree
    cb = cb_coords(p)
    ca = p.xyz14[:, 1]
    sc = np.where(p.mask14[:, 4:, None], p.xyz14[:, 4:], 0.0)
    cnt = p.mask14[:, 4:].sum(1, keepdims=True)
    cen = np.where(cnt > 0, sc.sum(1) / np.maximum(cnt, 1), ca)
    out = {}
    for name, x in (("cb", cb), ("cen", cen)):
        tree = cKDTree(x)
        for r in radii:
            out[f"n{name}{int(r)}"] = (np.array([len(v) for v in tree.query_ball_point(x, r)]) - 1).astype(np.int32)
    out["cn"] = smooth_cn(cb)
    u = cb - ca
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    pr = cKDTree(ca).query_pairs(hse_r, output_type="ndarray")
    up = np.zeros(len(p), np.int32)
    down = np.zeros(len(p), np.int32)
    if len(pr):
        a_, b_ = pr[:, 0], pr[:, 1]
        v = ca[b_] - ca[a_]
        ua = np.sum(v * u[a_], -1) > 0           # b in a's upper half-sphere
        ub = np.sum(-v * u[b_], -1) > 0
        np.add.at(up, a_, ua); np.add.at(down, a_, ~ua)
        np.add.at(up, b_, ub); np.add.at(down, b_, ~ub)
    out["hse_up"], out["hse_down"] = up, down
    return out


def annotate(p: Protein, do_sasa: bool = True) -> Protein:
    """Fill p.ann with backbone dihedrals, chis, SS, disulfides and burial descriptors."""
    p.ann.update(backbone_dihedrals(p))
    p.ann["chi"] = side_chain_chis(p)
    p.ann.update(dssp(p))
    p.ann["disulfides"] = disulfides(p)
    p.ann.update(burial(p))
    if do_sasa:
        p.ann.update(sasa(p))
    return p
