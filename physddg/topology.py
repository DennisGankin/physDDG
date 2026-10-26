"""Residue topology for PhysDDG: heavy-atom templates, atom types, charges, H-bond roles, chis, ideal geometry.

This module is the interface between geometry (structure.py, builder.py, rotlib.py) and energy (energy.py).
Energy parameters are keyed on the names in ATOM_TYPES; do not rename them.

Conventions
- Heavy atoms only, PDB (IUPAC) names and PDB order: N, CA, C, O, side chain.
- `charge`: formal charges on ionisable groups only, spread over the chemically equivalent atoms
  (Asp/Glu -1 over the two carboxyl O; Lys +1 on NZ; Arg +1 over NE/NH1/NH2; His neutral unless
  `his_charged`, then +1 over ND1/NE2; optional termini: N-term N +1, C-term -1 over O/OXT).
- H-bond roles (HBOND): per donor/acceptor heavy atom, its base atoms for angle terms. Base names refer to the
  same residue except "-C" = carbonyl C of residue i-1 (backbone amide N). `don_geom` tells how the polar H
  is placed: "bisector" (one H on the external bisector of D-base and D-base2: backbone NH, His/Trp ring NH,
  Arg NE), "sp2_2H" (two H in the base2-base-D plane: Asn ND2, Gln NE2, Arg NH1/NH2), "rot" (rotatable,
  H position not determined by heavy atoms: Ser/Thr/Tyr OH, Lys NZ). `acc_hyb` is sp2 / sp3 / ring.
- Ideal internal coordinates (IC): each side-chain atom D is built from three earlier atoms A,B,C with
  |CD| = bond, angle(B,C,D) = angle, dihedral(A,B,C,D) = chi_k + offset (chi_k in 1..4) or offset if chi=0.
  Bond lengths/angles are Engh & Huber (1991/2001) values cross-checked against medians in a culled
  high-resolution PDB set (scripts/41_ideal_geometry.py; notes/engine_geometry.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------- atom type vocabulary (E2 keys on these)
ATOM_TYPES = [
    "N_bb", "N_pro", "CA", "C_bb", "O_bb",
    "CH1", "CH2", "CH3", "C_aro", "C_sp2",
    "N_amide", "N_his", "N_trp", "N_lys", "N_arg",
    "O_amide", "O_carboxyl", "O_hydroxyl", "S_cys", "S_met",
]
TYPE_INDEX = {t: i for i, t in enumerate(ATOM_TYPES)}

AA1 = "ACDEFGHIKLMNPQRSTVWY"
AA3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
       "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
ONE_TO_THREE = dict(zip(AA1, AA3))
THREE_TO_ONE = dict(zip(AA3, AA1))
AA_INDEX = {a: i for i, a in enumerate(AA1)}

BACKBONE = ("N", "CA", "C", "O")

# ----------------------------------------------------------------------------- side-chain atoms and their types
_SC = {  # 1-letter -> [(atom, type)] side-chain heavy atoms in PDB order
    "A": [("CB", "CH3")],
    "C": [("CB", "CH2"), ("SG", "S_cys")],
    "D": [("CB", "CH2"), ("CG", "C_sp2"), ("OD1", "O_carboxyl"), ("OD2", "O_carboxyl")],
    "E": [("CB", "CH2"), ("CG", "CH2"), ("CD", "C_sp2"), ("OE1", "O_carboxyl"), ("OE2", "O_carboxyl")],
    "F": [("CB", "CH2"), ("CG", "C_aro"), ("CD1", "C_aro"), ("CD2", "C_aro"), ("CE1", "C_aro"),
          ("CE2", "C_aro"), ("CZ", "C_aro")],
    "G": [],
    "H": [("CB", "CH2"), ("CG", "C_aro"), ("ND1", "N_his"), ("CD2", "C_aro"), ("CE1", "C_aro"), ("NE2", "N_his")],
    "I": [("CB", "CH1"), ("CG1", "CH2"), ("CG2", "CH3"), ("CD1", "CH3")],
    "K": [("CB", "CH2"), ("CG", "CH2"), ("CD", "CH2"), ("CE", "CH2"), ("NZ", "N_lys")],
    "L": [("CB", "CH2"), ("CG", "CH1"), ("CD1", "CH3"), ("CD2", "CH3")],
    "M": [("CB", "CH2"), ("CG", "CH2"), ("SD", "S_met"), ("CE", "CH3")],
    "N": [("CB", "CH2"), ("CG", "C_sp2"), ("OD1", "O_amide"), ("ND2", "N_amide")],
    "P": [("CB", "CH2"), ("CG", "CH2"), ("CD", "CH2")],
    "Q": [("CB", "CH2"), ("CG", "CH2"), ("CD", "C_sp2"), ("OE1", "O_amide"), ("NE2", "N_amide")],
    "R": [("CB", "CH2"), ("CG", "CH2"), ("CD", "CH2"), ("NE", "N_arg"), ("CZ", "C_sp2"),
          ("NH1", "N_arg"), ("NH2", "N_arg")],
    "S": [("CB", "CH2"), ("OG", "O_hydroxyl")],
    "T": [("CB", "CH1"), ("OG1", "O_hydroxyl"), ("CG2", "CH3")],
    "V": [("CB", "CH1"), ("CG1", "CH3"), ("CG2", "CH3")],
    "W": [("CB", "CH2"), ("CG", "C_aro"), ("CD1", "C_aro"), ("CD2", "C_aro"), ("NE1", "N_trp"),
          ("CE2", "C_aro"), ("CE3", "C_aro"), ("CZ2", "C_aro"), ("CZ3", "C_aro"), ("CH2", "C_aro")],
    "Y": [("CB", "CH2"), ("CG", "C_aro"), ("CD1", "C_aro"), ("CD2", "C_aro"), ("CE1", "C_aro"),
          ("CE2", "C_aro"), ("CZ", "C_aro"), ("OH", "O_hydroxyl")],
}

# ----------------------------------------------------------------------------- chi angle definitions
CHI_ATOMS = {
    "A": [], "G": [],
    "C": [("N", "CA", "CB", "SG")],
    "D": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "E": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "F": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "H": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "I": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "K": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "L": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "M": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "N": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "P": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "Q": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "R": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
    "S": [("N", "CA", "CB", "OG")],
    "T": [("N", "CA", "CB", "OG1")],
    "V": [("N", "CA", "CB", "CG1")],
    "W": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "Y": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
}
N_CHI = {a: len(v) for a, v in CHI_ATOMS.items()}
# terminal chis with exact 2-fold symmetry (chi and chi+180 give the same heavy-atom set up to name swap)
CHI_SYMMETRIC = {("D", 2), ("E", 3), ("F", 2), ("Y", 2)}
# terminal chis that are only pseudo-symmetric (heavy atoms differ but are often mis-assigned in X-ray): flips
CHI_AMBIGUOUS = {("N", 2), ("Q", 3), ("H", 2)}
# atom-name swaps that map a residue onto itself (exact symmetry) or onto its common flip (ambiguous)
SYMMETRIC_SWAPS = {
    "D": [("OD1", "OD2")], "E": [("OE1", "OE2")], "R": [("NH1", "NH2")],
    "F": [("CD1", "CD2"), ("CE1", "CE2")], "Y": [("CD1", "CD2"), ("CE1", "CE2")],
}
AMBIGUOUS_SWAPS = {"N": [("OD1", "ND2")], "Q": [("OE1", "NE2")], "H": [("ND1", "CD2"), ("CE1", "NE2")]}
# terminal chis treated as non-rotameric (continuous distribution) in the rotamer library
NONROTAMERIC_CHI = {"D": 2, "E": 3, "F": 2, "H": 2, "N": 2, "Q": 3, "W": 2, "Y": 2}

# ----------------------------------------------------------------------------- H-bond donors / acceptors


@dataclass(frozen=True)
class HB:
    donor: int = 0          # number of polar H (0 = not a donor)
    acceptor: bool = False
    base: str = ""          # heavy atom bonded to the donor/acceptor ("-C" = C of residue i-1)
    base2: str = ""         # second reference atom (bisector partner or plane-defining atom)
    don_geom: str = ""      # bisector | sp2_2H | rot
    acc_hyb: str = ""       # sp2 | sp3 | ring


HB_BACKBONE = {
    "N": HB(donor=1, base="-C", base2="CA", don_geom="bisector"),   # not for Pro
    "O": HB(acceptor=True, base="C", base2="CA", acc_hyb="sp2"),
    "OXT": HB(acceptor=True, base="C", base2="CA", acc_hyb="sp2"),
}
HB_SIDECHAIN = {
    "S": {"OG": HB(1, True, "CB", "CA", "rot", "sp3")},
    "T": {"OG1": HB(1, True, "CB", "CA", "rot", "sp3")},
    "Y": {"OH": HB(1, True, "CZ", "CE1", "rot", "sp2")},
    "N": {"OD1": HB(0, True, "CG", "CB", "", "sp2"), "ND2": HB(2, False, "CG", "CB", "sp2_2H", "")},
    "Q": {"OE1": HB(0, True, "CD", "CG", "", "sp2"), "NE2": HB(2, False, "CD", "CG", "sp2_2H", "")},
    "D": {"OD1": HB(0, True, "CG", "CB", "", "sp2"), "OD2": HB(0, True, "CG", "CB", "", "sp2")},
    "E": {"OE1": HB(0, True, "CD", "CG", "", "sp2"), "OE2": HB(0, True, "CD", "CG", "", "sp2")},
    # His: both ring N flagged donor AND acceptor (tautomer not resolved; charged His: both donors)
    "H": {"ND1": HB(1, True, "CG", "CE1", "bisector", "ring"), "NE2": HB(1, True, "CD2", "CE1", "bisector", "ring")},
    "W": {"NE1": HB(1, False, "CD1", "CE2", "bisector", "")},
    "K": {"NZ": HB(3, False, "CE", "CD", "rot", "")},
    "R": {"NE": HB(1, False, "CD", "CZ", "bisector", ""),
          "NH1": HB(2, False, "CZ", "NE", "sp2_2H", ""), "NH2": HB(2, False, "CZ", "NE", "sp2_2H", "")},
}


def hbond_roles(aa: str) -> dict[str, HB]:
    """All H-bond roles of residue `aa` (1-letter), backbone included (Pro N excluded)."""
    roles = {k: v for k, v in HB_BACKBONE.items() if k != "OXT" and not (aa == "P" and k == "N")}
    roles.update(HB_SIDECHAIN.get(aa, {}))
    return roles


# ----------------------------------------------------------------------------- charges
def atom_charges(aa: str, his_charged: bool = False, nterm: bool = False, cterm: bool = False) -> dict[str, float]:
    """Formal charges (spread) per heavy atom of residue `aa`; atoms not listed are 0."""
    q: dict[str, float] = {}
    if aa == "D":
        q = {"OD1": -0.5, "OD2": -0.5}
    elif aa == "E":
        q = {"OE1": -0.5, "OE2": -0.5}
    elif aa == "K":
        q = {"NZ": 1.0}
    elif aa == "R":
        q = {"NE": 1 / 3, "NH1": 1 / 3, "NH2": 1 / 3}
    elif aa == "H" and his_charged:
        q = {"ND1": 0.5, "NE2": 0.5}
    if nterm:
        q["N"] = q.get("N", 0.0) + 1.0
    if cterm:
        q["O"] = q.get("O", 0.0) - 0.5
        q["OXT"] = q.get("OXT", 0.0) - 0.5
    return q


# ----------------------------------------------------------------------------- residue templates


@dataclass(frozen=True)
class ResidueTemplate:
    aa: str
    atoms: tuple[str, ...]           # heavy atoms, PDB order (backbone first)
    types: tuple[str, ...]           # ATOM_TYPES names
    type_ids: np.ndarray             # int indices into ATOM_TYPES
    charges: np.ndarray              # formal charges (His neutral, no termini)
    is_sidechain: np.ndarray         # bool (CB and beyond)
    chis: tuple[tuple[str, str, str, str], ...]

    def index(self, name: str) -> int:
        return self.atoms.index(name)


def _bb_types(aa: str) -> list[tuple[str, str]]:
    return [("N", "N_pro" if aa == "P" else "N_bb"), ("CA", "CA"), ("C", "C_bb"), ("O", "O_bb")]


def _make_template(aa: str) -> ResidueTemplate:
    at = _bb_types(aa) + _SC[aa]
    names = tuple(a for a, _ in at)
    types = tuple(t for _, t in at)
    q = atom_charges(aa)
    return ResidueTemplate(
        aa=aa, atoms=names, types=types,
        type_ids=np.array([TYPE_INDEX[t] for t in types], dtype=np.int64),
        charges=np.array([q.get(a, 0.0) for a in names], dtype=np.float32),
        is_sidechain=np.array([a not in BACKBONE for a in names]),
        chis=tuple(CHI_ATOMS[aa]),
    )


TEMPLATES = {aa: _make_template(aa) for aa in AA1}
OXT_TYPE = "O_carboxyl"
MAX_SC_ATOMS = max(len(v) for v in _SC.values())          # 10 (Trp)
MAX_ATOMS = 4 + MAX_SC_ATOMS                               # 14


def atom_type(aa: str, atom: str) -> str:
    """Atom type for (residue, atom name); OXT -> O_carboxyl. Raises KeyError for unknown atoms."""
    if atom == "OXT":
        return OXT_TYPE
    t = TEMPLATES[aa]
    return t.types[t.atoms.index(atom)]
