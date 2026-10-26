"""Shared helpers for the data-HT scripts (1x_*.py): structure reading/cleaning and sequence->structure mapping."""
import gzip
import io
from pathlib import Path

import gemmi
import numpy as np
from Bio import Align

ROOT = Path(__file__).resolve().parents[1]
PDB_MIRROR = Path("/rsyncPDB/data/structures/divided")
XTAL_DIR = ROOT / "data" / "structures" / "xtal"

# modified residues mapped to their parent when cleaning (anything else non-standard is dropped and reported)
MODIFIED_PARENT = {"MSE": "MET", "SEP": "SER", "TPO": "THR", "PTR": "TYR", "CSO": "CYS", "CME": "CYS",
                   "MLY": "LYS", "HYP": "PRO", "KCX": "LYS", "CSD": "CYS", "OCS": "CYS", "LLP": "LYS",
                   "SAC": "SER", "CSS": "CYS", "NEP": "HIS", "HIC": "HIS", "MLE": "LEU", "CGU": "GLU"}
STD = set(gemmi.find_tabulated_residue(r).name for r in
          ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
           "PRO", "SER", "THR", "TRP", "TYR", "VAL"])


def _read_text(path: Path) -> str:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


def read_structure(path) -> gemmi.Structure:
    """Read PDB/mmCIF (optionally gzipped). Strips ANISOU records if gemmi rejects them."""
    path = Path(path)
    name = path.name.replace(".gz", "")
    if name.endswith((".cif", ".mmcif")):
        st = gemmi.read_structure(str(path))
    else:
        try:
            st = gemmi.read_structure(str(path))
        except RuntimeError:
            txt = "\n".join(l for l in _read_text(path).splitlines() if not l.startswith("ANISOU"))
            st = gemmi.read_pdb_string(txt)
    st.setup_entities()
    return st


def mirror_path(pdb_id: str):
    """Local wwPDB mirror path (PDB format preferred, mmCIF fallback); None if absent."""
    pid = pdb_id.lower()
    p = PDB_MIRROR / "pdb" / pid[1:3] / f"pdb{pid}.ent.gz"
    if p.exists():
        return p
    c = PDB_MIRROR / "mmCIF" / pid[1:3] / f"{pid}.cif.gz"
    if c.exists():
        return c
    o = PDB_MIRROR.parent / "obsolete" / "pdb" / pid[1:3] / f"pdb{pid}.ent.gz"  # obsolete entries (e.g. 2A01)
    return o if o.exists() else None


def resid(res: gemmi.Residue) -> str:
    return f"{res.seqid.num}{res.seqid.icode.strip()}"


def one(resname: str) -> str:
    resname = MODIFIED_PARENT.get(resname, resname)
    return gemmi.find_tabulated_residue(resname).one_letter_code.upper() if gemmi.find_tabulated_residue(resname) else "X"


def chain_residues(st: gemmi.Structure, chain: str, model: int = 0):
    """[(resnum_str, one_letter)] for polymer amino-acid residues of a chain (first model)."""
    ch = st[model].find_chain(chain)
    if ch is None:
        return None
    out = []
    for res in ch:
        if res.het_flag == "H" and res.name not in MODIFIED_PARENT:
            continue
        info = gemmi.find_tabulated_residue(MODIFIED_PARENT.get(res.name, res.name))
        if info is None or not info.is_amino_acid() or res.find_atom("CA", "*") is None:
            continue
        out.append((resid(res), one(res.name)))
    return out


def clean_chain(st: gemmi.Structure, chain: str, out_path: Path):
    """Write a minimally cleaned single-chain PDB: model 1, one chain, no waters/H/ligands, first altloc,
    modified residues -> parent (MSE->MET, ...). Residue numbers and insertion codes are kept. Returns info dict."""
    st = st.clone()
    while len(st) > 1:
        del st[1]
    model = st[0]
    for name in [c.name for c in model]:
        if name != chain:
            model.remove_chain(name)
    if len(model) == 0:
        raise KeyError(f"chain {chain} not found")
    # gemmi may split one author chain into polymer + ligand/water parts with the same name -> merge view
    het, modified = set(), set()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    for ch in model:
        for i in range(len(ch) - 1, -1, -1):
            res = ch[i]
            if res.is_water():
                del ch[i]
                continue
            if res.name in MODIFIED_PARENT:
                modified.add(res.name)
                res.name = MODIFIED_PARENT[res.name]
                res.het_flag = "A"
                if res.name == "MET":
                    for a in res:
                        if a.name == "SE":
                            a.name, a.element = "SD", gemmi.Element("S")
                continue
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                het.add(res.name)
                del ch[i]
    st.remove_empty_chains()
    st.setup_entities()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(out_path), gemmi.PdbWriteOptions(minimal=True))
    return {"het_removed": ";".join(sorted(het)), "modified_converted": ";".join(sorted(modified))}


def aligner():
    al = Align.PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score = 2, -1
    al.open_gap_score, al.extend_gap_score = -5, -0.5
    al.end_insertion_score = al.end_deletion_score = 0
    return al


def align_map(seq_a: str, seq_b: str):
    """Global alignment; returns (dict index_a->index_b for aligned columns, identity over aligned non-X columns)."""
    aln = aligner().align(seq_a, seq_b)[0]
    m, same, tot = {}, 0, 0
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        for i, j in zip(range(a0, a1), range(b0, b1)):
            m[i] = j
            if seq_a[i] != "X" and seq_b[j] != "X":
                tot += 1
                same += seq_a[i] == seq_b[j]
    return m, (same / tot if tot else 0.0)


def cb_burial(st: gemmi.Structure, chain: str):
    """{resnum_str: number of CB (CA for Gly) atoms within 10 A}, for sign sanity checks."""
    ch = st[0].find_chain(chain)
    xyz, keys = [], []
    for res in ch:
        a = res.find_atom("CB", "*") or res.find_atom("CA", "*")
        if a is not None and gemmi.find_tabulated_residue(res.name) is not None:
            xyz.append(a.pos.tolist())
            keys.append(resid(res))
    xyz = np.array(xyz)
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return dict(zip(keys, (d < 10).sum(1) - 1))


def sign_report(df, burial_col="cb_nb", log=print, label=""):
    """Print the standard sign-sanity table (ddg positive = destabilising expected for buried hydrophobic loss)."""
    hyd = df.wt.isin(list("ILVFMWY")) & df.mut.isin(list("AGPDKSTNQE"))
    rows = [("buried(>=16) hydrophobic->small/polar", hyd & (df[burial_col] >= 16)),
            ("exposed(<=8) hydrophobic->small/polar", hyd & (df[burial_col] <= 8)),
            ("any -> P", df.mut == "P"), ("all", df.ddg.notna())]
    for lab, m in rows:
        x = df.loc[m, "ddg"].dropna()
        log(f"  sign {label} {lab:40s} n={len(x):6d} mean={x.mean():+.2f} frac>0={(x > 0).mean():.2f}")


def site_context(st: gemmi.Structure, chain: str, cutoff: float = 5.0):
    """For each residue of `chain` in the ORIGINAL structure (model 1): whether any heavy atom lies within `cutoff` A
    of (a) a non-water HETATM ligand/ion (modified amino acids inside a polymer chain are not counted) and (b) a heavy
    atom of another protein chain of the asymmetric unit. Returns {resnum: (near_het, het_names, near_other_chain)}."""
    model = st[0]
    lig, oth = [], []
    for ch in model:
        for res in ch:
            if res.is_water():
                continue
            info = gemmi.find_tabulated_residue(MODIFIED_PARENT.get(res.name, res.name))
            is_aa = info is not None and info.is_amino_acid() and res.find_atom("CA", "*") is not None
            for a in res:
                if a.element.name == "H":
                    continue
                if not is_aa and (res.het_flag == "H" or info is None or not info.is_amino_acid()):
                    lig.append((a.pos, res.name))
                elif is_aa and ch.name != chain:
                    oth.append(a.pos)
    from scipy.spatial import cKDTree
    out = {}
    target = model.find_chain(chain)
    lig_tree = cKDTree(np.array([[p.x, p.y, p.z] for p, _ in lig])) if lig else None
    oth_tree = cKDTree(np.array([[p.x, p.y, p.z] for p in oth])) if oth else None
    for res in target:
        if gemmi.find_tabulated_residue(MODIFIED_PARENT.get(res.name, res.name)) is None:
            continue
        xyz = np.array([a.pos.tolist() for a in res if a.element.name != "H"])
        if len(xyz) == 0:
            continue
        names = set()
        if lig_tree is not None:
            for hits in lig_tree.query_ball_point(xyz, cutoff):
                names.update(lig[i][1] for i in hits)
        near_o = oth_tree is not None and any(len(h) for h in oth_tree.query_ball_point(xyz, cutoff))
        k = resid(res)
        if k not in out:  # keep first occurrence of a residue number
            out[k] = (bool(names), ";".join(sorted(names)), bool(near_o))
    return out


def mirror_context(pdb_id: str, local_st: gemmi.Structure, local_chain: str, cutoff: float = 5.0):
    """Annotate a pre-cleaned local chain (e.g. ThermoMPNN's FireProt PDBs, which have no HETATM / other chains)
    with site context from the ORIGINAL wwPDB entry in the local mirror. The mirror chain is the one whose
    sequence best matches; residues are matched by alignment and must have the same amino acid.
    Returns ({local_resnum: (near_het, het_names, near_other_chain)}, info)."""
    src = mirror_path(pdb_id)
    if src is None:
        return {}, {"mirror_error": "not in mirror"}
    st = read_structure(src)
    loc = chain_residues(local_st, local_chain)
    lseq = "".join(a for _, a in loc)
    best = None
    for ch in st[0]:
        cr = chain_residues(st, ch.name)
        if not cr:
            continue
        m, ident = align_map(lseq, "".join(a for _, a in cr))
        score = ident * len(m) / max(len(lseq), 1)
        if best is None or score > best[0]:
            best = (score, ch.name, cr, m, ident)
    if best is None:
        return {}, {"mirror_error": "no protein chain"}
    _, mch, cr, m, ident = best
    ctx = site_context(st, mch, cutoff)
    # coordinate check: CA displacement between local and mirror residues (no superposition)
    lch, mchain = local_st[0].find_chain(local_chain), st[0].find_chain(mch)
    lca = {resid(r): r.find_atom("CA", "*") for r in lch}
    mca = {resid(r): r.find_atom("CA", "*") for r in mchain}
    out, disp, same_num = {}, [], 0
    for i, j in m.items():
        if loc[i][1] != cr[j][1]:
            continue
        out[loc[i][0]] = ctx.get(cr[j][0], (False, "", False))
        same_num += loc[i][0] == cr[j][0]
        a, b = lca.get(loc[i][0]), mca.get(cr[j][0])
        if a is not None and b is not None:
            disp.append(a.pos.dist(b.pos))
    prot_chains = sorted({c.name for c in st[0] if chain_residues(st, c.name)})
    info = {"mirror_src": str(src), "mirror_chain": mch, "mirror_identity": round(ident, 3),
            "mirror_n_protein_chains": len(prot_chains), "other_protein_chains": ";".join(c for c in prot_chains if c != mch),
            "mirror_same_numbering_frac": round(same_num / max(len(out), 1), 3),
            "mirror_ca_disp_median": round(float(np.median(disp)), 3) if disp else np.nan,
            "mirror_hetero_in_entry": ";".join(sorted({r.name for c in st[0] for r in c
                                                        if r.het_flag == "H" and not r.is_water()
                                                        and r.name not in MODIFIED_PARENT}))}
    return out, info
