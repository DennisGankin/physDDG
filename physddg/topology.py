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


# ----------------------------------------------------------------------------- ideal internal coordinates
# IC_SPEC[aa] = [(atom, A, B, C, chi)]: D=atom is placed from A,B,C; torsion(A,B,C,D) = chi_k + offset (chi>0)
# or offset (chi=0). Numeric values (bond, angle, offset) are in IDEAL_IC below.
IC_SPEC = {
    "G": [],
    "A": [],
    "C": [("SG", "N", "CA", "CB", 1)],
    "D": [("CG", "N", "CA", "CB", 1), ("OD1", "CA", "CB", "CG", 2), ("OD2", "CA", "CB", "CG", 2)],
    "E": [("CG", "N", "CA", "CB", 1), ("CD", "CA", "CB", "CG", 2), ("OE1", "CB", "CG", "CD", 3),
          ("OE2", "CB", "CG", "CD", 3)],
    "F": [("CG", "N", "CA", "CB", 1), ("CD1", "CA", "CB", "CG", 2), ("CD2", "CA", "CB", "CG", 2),
          ("CE1", "CB", "CG", "CD1", 0), ("CE2", "CB", "CG", "CD2", 0), ("CZ", "CG", "CD1", "CE1", 0)],
    "H": [("CG", "N", "CA", "CB", 1), ("ND1", "CA", "CB", "CG", 2), ("CD2", "CA", "CB", "CG", 2),
          ("CE1", "CB", "CG", "ND1", 0), ("NE2", "CB", "CG", "CD2", 0)],
    "I": [("CG1", "N", "CA", "CB", 1), ("CG2", "N", "CA", "CB", 1), ("CD1", "CA", "CB", "CG1", 2)],
    "K": [("CG", "N", "CA", "CB", 1), ("CD", "CA", "CB", "CG", 2), ("CE", "CB", "CG", "CD", 3),
          ("NZ", "CG", "CD", "CE", 4)],
    "L": [("CG", "N", "CA", "CB", 1), ("CD1", "CA", "CB", "CG", 2), ("CD2", "CA", "CB", "CG", 2)],
    "M": [("CG", "N", "CA", "CB", 1), ("SD", "CA", "CB", "CG", 2), ("CE", "CB", "CG", "SD", 3)],
    "N": [("CG", "N", "CA", "CB", 1), ("OD1", "CA", "CB", "CG", 2), ("ND2", "CA", "CB", "CG", 2)],
    "P": [("CG", "N", "CA", "CB", 1), ("CD", "CA", "CB", "CG", 2)],
    "Q": [("CG", "N", "CA", "CB", 1), ("CD", "CA", "CB", "CG", 2), ("OE1", "CB", "CG", "CD", 3),
          ("NE2", "CB", "CG", "CD", 3)],
    "R": [("CG", "N", "CA", "CB", 1), ("CD", "CA", "CB", "CG", 2), ("NE", "CB", "CG", "CD", 3),
          ("CZ", "CG", "CD", "NE", 4), ("NH1", "CD", "NE", "CZ", 0), ("NH2", "CD", "NE", "CZ", 0)],
    "S": [("OG", "N", "CA", "CB", 1)],
    "T": [("OG1", "N", "CA", "CB", 1), ("CG2", "N", "CA", "CB", 1)],
    "V": [("CG1", "N", "CA", "CB", 1), ("CG2", "N", "CA", "CB", 1)],
    "W": [("CG", "N", "CA", "CB", 1), ("CD1", "CA", "CB", "CG", 2), ("CD2", "CA", "CB", "CG", 2),
          ("NE1", "CB", "CG", "CD1", 0), ("CE2", "CB", "CG", "CD2", 0), ("CE3", "CB", "CG", "CD2", 0),
          ("CZ2", "CG", "CD2", "CE2", 0), ("CZ3", "CG", "CD2", "CE3", 0), ("CH2", "CD2", "CE2", "CZ2", 0)],
    "Y": [("CG", "N", "CA", "CB", 1), ("CD1", "CA", "CB", "CG", 2), ("CD2", "CA", "CB", "CG", 2),
          ("CE1", "CB", "CG", "CD1", 0), ("CE2", "CB", "CG", "CD2", 0), ("CZ", "CG", "CD1", "CE1", 0),
          ("OH", "CD1", "CE1", "CZ", 0)],
}
# CB is placed from (C, N, CA) for every residue with a CB: torsion(C,N,CA,CB) fixed.
CB_SPEC = ("CB", "C", "N", "CA", 0)

# IDEAL_IC[aa][atom] = (bond A, angle deg, torsion offset deg). Bond/angle = medians over well-ordered residues
# of the PISCES pc30/1.8 A set (scripts/41_ideal_geometry.py); they agree with Engh & Huber (2001) to <=0.005 A /
# ~1 deg. Planar offsets snapped to 0/180; branch offsets (Ile CG2, Leu CD2, Thr CG2, Val CG2) and CB torsion measured.
IDEAL_IC = {
    "A": {"CB": (1.523, 110.3, -122.7)},
    "C": {"CB": (1.528, 110.5, -122.3), "SG": (1.807, 114.1, 0.0)},
    "D": {"CB": (1.529, 110.6, -122.7), "CG": (1.515, 113.4, 0.0), "OD1": (1.249, 119.3, 0.0),
          "OD2": (1.251, 117.6, 180.0)},
    "E": {"CB": (1.529, 110.4, -122.8), "CG": (1.519, 113.6, 0.0), "CD": (1.516, 113.9, 0.0),
          "OE1": (1.250, 119.2, 0.0), "OE2": (1.251, 117.9, 180.0)},
    "F": {"CB": (1.532, 110.6, -122.5), "CG": (1.505, 113.8, 0.0), "CD1": (1.386, 120.8, 0.0),
          "CD2": (1.385, 120.7, 180.0), "CE1": (1.386, 120.8, 180.0), "CE2": (1.385, 120.8, 180.0),
          "CZ": (1.377, 120.0, 0.0)},
    "G": {},
    "H": {"CB": (1.530, 110.4, -122.6), "CG": (1.492, 113.6, 0.0), "ND1": (1.378, 122.6, 0.0),
          "CD2": (1.354, 131.0, 180.0), "CE1": (1.322, 109.1, 180.0), "NE2": (1.373, 107.2, 180.0)},
    "I": {"CB": (1.540, 111.2, -123.3), "CG1": (1.532, 110.8, 0.0), "CG2": (1.525, 110.6, -124.2),
          "CD1": (1.514, 114.1, 0.0)},
    "K": {"CB": (1.529, 110.5, -122.7), "CG": (1.519, 113.9, 0.0), "CD": (1.519, 112.1, 0.0),
          "CE": (1.516, 111.9, 0.0), "NZ": (1.487, 111.7, 0.0)},
    "L": {"CB": (1.528, 110.3, -122.3), "CG": (1.528, 115.5, 0.0), "CD1": (1.520, 110.5, 0.0),
          "CD2": (1.521, 111.0, 123.2)},
    "M": {"CB": (1.529, 110.5, -122.5), "CG": (1.518, 113.6, 0.0), "SD": (1.804, 113.7, 0.0),
          "CE": (1.790, 100.5, 0.0)},
    "N": {"CB": (1.529, 110.6, -123.4), "CG": (1.511, 112.6, 0.0), "OD1": (1.232, 120.9, 0.0),
          "ND2": (1.325, 116.5, 180.0)},
    "P": {"CB": (1.533, 103.2, -119.9), "CG": (1.495, 104.3, 0.0), "CD": (1.506, 105.1, 0.0)},
    "Q": {"CB": (1.529, 110.4, -122.7), "CG": (1.520, 113.5, 0.0), "CD": (1.511, 112.8, 0.0),
          "OE1": (1.233, 121.0, 0.0), "NE2": (1.325, 116.4, 180.0)},
    "R": {"CB": (1.529, 110.5, -122.6), "CG": (1.519, 113.7, 0.0), "CD": (1.518, 112.2, 0.0),
          "NE": (1.458, 111.7, 0.0), "CZ": (1.328, 124.8, 0.0), "NH1": (1.326, 120.9, 0.0),
          "NH2": (1.326, 119.5, 180.0)},
    "S": {"CB": (1.527, 110.4, -122.3), "OG": (1.416, 111.0, 0.0)},
    "T": {"CB": (1.534, 110.9, -122.5), "OG1": (1.431, 109.2, 0.0), "CG2": (1.519, 111.6, -121.1)},
    "V": {"CB": (1.540, 111.3, -123.1), "CG1": (1.522, 110.8, 0.0), "CG2": (1.523, 110.7, 123.4)},
    "W": {"CB": (1.531, 110.4, -122.5), "CG": (1.496, 114.2, 0.0), "CD1": (1.365, 127.1, 0.0),
          "CD2": (1.432, 126.6, 180.0), "NE1": (1.372, 110.2, 180.0), "CE2": (1.409, 107.2, 180.0),
          "CE3": (1.399, 133.9, 0.0), "CZ2": (1.394, 122.3, 180.0), "CZ3": (1.382, 118.7, 180.0),
          "CH2": (1.369, 117.4, 0.0)},
    "Y": {"CB": (1.532, 110.6, -122.4), "CG": (1.509, 113.7, 0.0), "CD1": (1.388, 121.0, 0.0),
          "CD2": (1.387, 121.0, 180.0), "CE1": (1.385, 121.2, 180.0), "CE2": (1.385, 121.3, 180.0),
          "CZ": (1.379, 119.7, 0.0), "OH": (1.377, 119.8, 180.0)},
}
CB_IC = IDEAL_IC["A"]["CB"]   # generic ideal CB (Gly -> X, missing CB)
