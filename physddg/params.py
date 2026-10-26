"""Per-atom-type parameters and term hyper-parameters for the PhysDDG energy function (engineer E2).

Units: Å, Å^3, kcal/mol, elementary charge.

Sources (heavy-atom / united-atom model, polar H not explicit). Every number below was verified against a
public copy of the original parameter file (downloaded by scripts/44_fetch_energy_param_sources.sh, compared
in tests/test_energy.py::test_params_match_sources):
- vdW: CHARMM19 united-atom force field param19.inp (Brooks et al. 1983 J Comput Chem 4:187; Neria, Fischer &
  Karplus 1996 J Chem Phys 105:1902), NONBONDED section: Rmin/2 (Å), well depth ε (kcal/mol) and the special
  1-4 values (ε 0.1, Rmin/2 1.9 for the extended carbons). Combination: Rmin_ij = Rmin/2_i + Rmin/2_j,
  ε_ij = sqrt(ε_i ε_j). verified: param19.inp (VMD trunctraj plugin copy).
- Solvation: EEF1, Lazaridis & Karplus 1999 Proteins 35:133, Table I (CHARMM19 types): volume V (Å^3), ΔG_ref,
  ΔG_free (kcal/mol), λ (3.5 Å; 6.0 Å for charged groups), R_i = Rmin/2 of param19.
  verified: OSPREY3 config/eef1parm.dat (verbatim Table I, all 17 CHARMM19 types); ΔG_ref additionally agrees
  with PLUMED src/colvar/EEFSolv.cpp (CHARMM22/36 port of the same table). ΔH/ΔCp are not needed (298 K).
- History: the first version (10-26) used recalled values; verification corrected ΔG_free (EEF1 uses ΔG_free ≠
  ΔG_ref for most types, e.g. NH1 -8.9 vs -5.95, CH3E 1.50 vs 1.089) and NH2 ΔG_ref (-5.45, was -5.95).
  Radii, well depths and volumes were already correct.

Adaptations to the 20-type PhysDDG vocabulary (topology.ATOM_TYPES):
- CA -> CH1E for all residues (Gly CA is CH2E in CHARMM19; merged).
- C_aro -> CR1E (aromatic CH); ring carbons without H (Phe/Tyr CG, Tyr CZ, Trp CD2/CE2/CG, His CG) are type
  C/CR in CHARMM19 (V 14.7/8.3, ΔG ~0/-0.9); merged into CR1E (majority of ring carbons carry an H).
- C_sp2 (Asp CG, Glu CD, Asn CG, Gln CD, Arg CZ) -> C (carbonyl carbon).
- N_his -> NR (unprotonated ring N); in CHARMM19 the protonated N is NH1 (ΔG -5.95): merged (tautomer unknown).
- N_trp -> NH1; N_arg (NE, NH1, NH2) -> NC2 (CHARMM19 types NE as NH1: merged); O_amide -> O; OXT -> OC.
- S_cys -> SH1E (free Cys; disulfide S in CHARMM19 is type S, merged); S_met -> S.
- Polar/apolar class (for the two EEF1 terms): N, O = polar; C, S = apolar (as requested in the design).
H-bond roles and charges come from topology.py (E1), not from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .topology import ATOM_TYPES

# ----------------------------------------------------------------------------- per CHARMM19 type
# name: (Rmin/2 [Å], eps [kcal/mol], V [Å^3], dG_ref, dG_free [kcal/mol], lambda [Å])
# vdW verified: param19.inp; V, dG_ref, dG_free, lambda verified: OSPREY3 eef1parm.dat (= EEF1 Table I)
CHARMM19 = {
    "C":    (2.100, 0.1200, 14.7,   0.000,   0.00, 3.5),
    "CH1E": (2.365, 0.0486, 23.7,  -0.187,  -0.25, 3.5),
    "CH2E": (2.235, 0.1142, 22.4,   0.372,   0.52, 3.5),
    "CH3E": (2.165, 0.1811, 30.0,   1.089,   1.50, 3.5),
    "CR1E": (2.100, 0.1200, 18.4,   0.057,   0.08, 3.5),
    "NH1":  (1.600, 0.2384,  4.4,  -5.950,  -8.90, 3.5),
    "NR":   (1.600, 0.2384,  4.4,  -3.820,  -4.00, 3.5),
    "NH2":  (1.600, 0.2384, 11.2,  -5.450,  -7.80, 3.5),
    "NH3":  (1.600, 0.2384, 11.2, -20.000, -20.00, 6.0),
    "NC2":  (1.600, 0.2384, 11.2, -10.000, -10.00, 6.0),
    "N":    (1.600, 0.2384,  0.0,  -1.000,  -1.55, 3.5),
    "OH1":  (1.600, 0.1591, 10.8,  -5.920,  -6.70, 3.5),
    "O":    (1.600, 0.1591, 10.8,  -5.330,  -5.85, 3.5),
    "OC":   (1.600, 0.6469, 10.8, -10.000, -10.00, 6.0),
    "S":    (1.890, 0.0430, 14.7,  -3.240,  -4.10, 3.5),
    "SH1E": (1.890, 0.0430, 21.4,  -2.050,  -2.70, 3.5),
}
# CHARMM19 special 1-4 vdW (Rmin/2, eps) for the extended carbons (param19 "1-4" columns); others unchanged
LJ14 = {"C": (1.90, 0.1), "CH1E": (1.90, 0.1), "CH2E": (1.90, 0.1), "CH3E": (1.90, 0.1), "CR1E": (1.90, 0.1)}
VERIFIED = {k: "vdW: param19.inp; EEF1: OSPREY3 eef1parm.dat (Table I), dG_ref also PLUMED EEFSolv.cpp"
            for k in CHARMM19}

# PhysDDG type -> (CHARMM19 type, polar?)
TYPE_MAP = {
    "N_bb": ("NH1", True), "N_pro": ("N", True), "CA": ("CH1E", False), "C_bb": ("C", False),
    "O_bb": ("O", True), "CH1": ("CH1E", False), "CH2": ("CH2E", False), "CH3": ("CH3E", False),
    "C_aro": ("CR1E", False), "C_sp2": ("C", False), "N_amide": ("NH2", True), "N_his": ("NR", True),
    "N_trp": ("NH1", True), "N_lys": ("NH3", True), "N_arg": ("NC2", True), "O_amide": ("O", True),
    "O_carboxyl": ("OC", True), "O_hydroxyl": ("OH1", True), "S_cys": ("SH1E", False), "S_met": ("S", False),
}
assert set(TYPE_MAP) == set(ATOM_TYPES)

COLS = ("rmin_half", "eps", "volume", "dg_ref", "dg_free", "lam", "polar", "rmin14_half", "eps14")


def type_table() -> np.ndarray:
    """[n_types, len(COLS)] float32 table in topology.ATOM_TYPES order."""
    rows = []
    for t in ATOM_TYPES:
        c19, polar = TYPE_MAP[t]
        v = CHARMM19[c19]
        rows.append(list(v) + [float(polar)] + list(LJ14.get(c19, v[:2])))
    return np.asarray(rows, np.float32)


# ----------------------------------------------------------------------------- term hyper-parameters
@dataclass
class EnergyParams:
    """Functional-form constants (not fitted weights). Distances in Å, energies in kcal/mol."""
    # Lennard-Jones (CHARMM form eps[(Rmin/r)^12 - 2(Rmin/r)^6]) split at r = Rmin (WCA-like)
    lj_on: float = 6.0            # attraction switched (CHARMM switch) from lj_on ...
    lj_off: float = 8.0           # ... to 0 at lj_off
    rep_cap: float = 3.0          # repulsion per atom pair becomes linear above this value (backbone flexibility)
    rep_scale: float = 0.9        # radius scale for the repulsive branch. D14 (PI, 10-26): 0.9 = structural
    #                               calibration only (45c: native chis recognised best; CHARMM19 united-atom radii
    #                               are ~10% too large for heavy-atom-only crystal contacts); no ΔΔG data used
    rep_soft: float = 0.0         # > 0 adds term lj_rep_soft = s*log1p(raw/s) per atom pair (raw = uncapped repulsive
                                  # branch eps((Rr/r)^6-1)^2, log-saturating; s in kcal/mol). Default off (E1, D9 variants)
    w14: float = 0.5              # scale of LJ for 1-4 pairs (3 bonds apart; CHARMM19 1-4 vdW parameters)
    hb_rmin: float = 3.0          # Rmin used for donor-acceptor pairs (implicit polar H, cf. Rosetta)
    # EEF1 solvation
    solv_on: float = 7.0
    solv_off: float = 9.0
    # H-bonds, geometric model (side chains): heavy-atom D..A distance and angle windows
    hb_d_min: tuple = (2.3, 2.55)  # ramp up (Å)
    hb_d_max: tuple = (3.2, 3.6)   # ramp down (Å)
    hb_don_bis: tuple = (75.0, 40.0)      # explicit-H donors: angle(D->H, D->A) full <= 40°, zero >= 75°
    hb_don_cone: tuple = (15.0, 35.0, 95.0, 115.0)  # rotatable/NH2 donors: angle(B->D, D->A) window (°)
    hb_acc: tuple = (100.0, 70.0)  # acceptors: angle(B->A, A->D) full <= 70°, zero >= 100°
    # H-bonds, backbone-backbone: DSSP electrostatic model (Kabsch & Sander 1983)
    dssp_q: float = 0.084 * 332.0  # q1*q2*f (kcal/mol·Å)
    dssp_floor: float = -3.0       # energy floor per H-bond
    dssp_on: float = 5.0           # switch on r(O..N)
    dssp_off: float = 6.0
    nh_len: float = 1.01
    co_len: float = 1.231
    # electrostatics (formal charges from topology)
    elec_modes: tuple = ("dh", "bur", "dd")   # each mode is returned as its own term elec_<mode>; dh = default (D10)
    coulomb: float = 332.0636
    eps_dd: float = 4.0            # ε = eps_dd * r
    eps_dh: float = 80.0           # Debye-Hückel: solvent dielectric ...
    debye: float = 7.85            # ... and Debye length (Å): 3.04/sqrt(I) at 25 °C, I = 0.15 M
    eps_in: float = 4.0            # burial-dependent (also DH-screened) ε_ij = eps_in + (eps_out-eps_in)(1 - sqrt(b_i b_j))
    eps_out: float = 80.0
    elec_on: float = 10.0
    elec_off: float = 12.0
    elec_rmin: float = 2.0         # distance floor for Coulomb (Å)
    burial_n: tuple = (8.0, 24.0)  # CB neighbour count (10 Å) mapped linearly to burial 0..1
    # disulfide geometry: SG-SG distance, CB-SG-SG angles, CB-SG-SG-CB dihedral (Gaussians)
    ss_d: tuple = (2.04, 0.10)
    ss_ang: tuple = (104.0, 15.0)
    ss_dih: tuple = (90.0, 30.0)
    ss_bond_max: float = 2.5       # SG-SG closer than this = covalent (excluded from non-bonded terms)
    # numerics
    chunk_elems: int = 1 << 24     # max slot pairs per chunk
    extra: dict = field(default_factory=dict)
