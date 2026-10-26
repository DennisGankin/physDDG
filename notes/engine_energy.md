# Engine: energy function (E2) — physddg/params.py, physddg/energy.py

Status 2026-10-26. Tests: `tests/test_energy.py` (21 pass). Validation scripts `scripts/45a-d_*.py`, outputs in
`results/engine_energy/`.

## API (torch, float32, GPU; every call returns per-term energies `[..., n_terms]`, never summed)
Terms (`energy.term_names(p)`): `lj_atr, lj_rep, solv_apolar, solv_polar, hb_sc_bb, hb_sc_sc, hb_bb_bb,
elec_dh, elec_bur, elec_dd, disulf`. kcal/mol, Å.

Representation: **residue blocks** of A = 15 atom slots (E1's atom14 template order plus OXT in slot 14).
`Blocks(xyz[...,15,3], aa, mask, pos, prev_c, nterm, burial, cap=None)` with any leading batch dims.
`pos` is chain-aware (peptide-bonded neighbours differ by exactly 1, breaks by 3).
- `blocks_from_protein(structure.Protein, device)` adapter for E1's loader (or `blocks_from_arrays`).
- `decompose(blocks[L])` -> `Decomposition(one_body [L,T], pair [L,L,T])`; `.per_residue`, `.total`.
- `make_env(blocks, exclude=[sites])` -> fixed environment (features computed once).
- `env_energy(mov[...], env, env_mask=None)`: one residue (any number of variants/rotamers, any mix of sites)
  vs the fixed environment. Variants are grouped by site (`pos`); per site the env blocks within
  cutoff + extent are gathered once and broadcast. Optional `env_mask` `[E]` or `[..., E]`.
- `intra_energy(mov)` intra-residue (chi-dependent pairs + EEF1 dG_ref of the residue's atoms).
- `pair_energy(a, b)` rotamer–rotamer energies with broadcasting (e.g. `[Na,1]` x `[1,Nb]` -> `[Na,Nb,T]`).
- `local_energy(mov[B,R], env)` -> `(one_body [B,R,T], two_body [B,R,R,T])` = everything a mean-field /
  Boltzmann packer needs: one-body = intra + vs fixed env, two-body = between flexible residues.
- `atom_solvation(blocks)` per-atom EEF1 solvation `[L,15]` (burial descriptor).
- `Blocks.cap` per-residue repulsion cap (D9: `c0 + c1*flex(site)`); pair cap = max of the two atoms' caps.
- `StatTerms` interface for Ramachandran (`rama(aa, phi, psi)`, periodic bilinear) and side-chain entropy
  (`sc_entropy(aa)`); `StatTerms.from_library()` loads E1's tables once `physddg/rotlib.py` exists.
- `enable_compile()` optional torch.compile of the short-range kernel (~2x; ~40 s compile per shape pattern).
Consistency: local/env results equal the whole-structure decomposition to < 4e-6 kcal/mol (tested).

## Functional forms (constants in `params.EnergyParams`, frozen)
- LJ (CHARMM form eps[(Rmin/r)^12 - 2(Rmin/r)^6]) split at Rmin (WCA): attraction = -eps inside Rmin, LJ beyond,
  CHARMM-switched 6->8 Å; repulsion = eps(u-1)^2, u=(Rmin/r)^6, linear continuation where it exceeds
  `rep_cap` (3 kcal/mol/pair default; analytic r_c, C1-continuous). `rep_scale` scales repulsive radii.
  Donor–acceptor pairs use Rmin <= 3.0 Å (implicit polar H). 1-4 pairs use CHARMM19 1-4 vdW, scaled w14 = 0.5.
- EEF1: dG_i = dG_ref_i - sum_j f_i(r) V_j, f_i = dG_free_i/(2 pi^1.5 lambda_i r^2) exp(-((r-R_i)/lambda_i)^2),
  switched 7->9 Å; apolar term = C/S atoms, polar term = N/O atoms. dG_ref enters via intra/one-body.
- Side-chain H-bonds (geometric, FoldX/Rosetta-like): f(d_DA: 2.3-2.55 up, 3.2-3.6 down) x donor angle
  (bisector donors: angle(D->H, D->A) full <= 40°, 0 >= 75°; rotatable/NH2 donors: angle(B->D, D->A) window
  35–95° full, 15/115° zero) x acceptor angle (angle(B->A, A->D) full <= 70°, 0 >= 100°); E = -f (max of the two
  directions), split into sc–bb / sc–sc. Needs >= 4 bonds separation. Donor/acceptor roles, bases and H
  geometry from `topology.hbond_roles` (E1).
- Backbone–backbone: DSSP electrostatic energy (Kabsch & Sander), amide H on the C(i-1)/CA bisector, carbonyl C
  reconstructed from the C=O direction, floored at -3 kcal/mol, switched on r(O..N) 5->6 Å, |i-j| >= 2;
  Pro has no donor; chain-start N treated as NH3+ cone donor.
- Electrostatics: formal charges from topology (His neutral; termini charged at chain starts / OXT); `dh`
  Debye–Hückel eps 80, Debye length 7.85 Å (0.15 M) [default, D10]; `bur` DH-screened with
  eps_ij = 4 + 76(1 - sqrt(b_i b_j)), b = CB-count burial (10 Å, 8->24 mapped to 0->1); `dd` eps = 4r.
  CHARMM switch 10->12 Å, r floor 2 Å, intra-residue and <= 1-4 pairs excluded.
- Disulfide: -g(d_SG-SG; 2.04, 0.10) g(CB-SG-SG; 104°, 15°)^2-ish (both angles) g(|chi_SS|; 90°, 30°).
  SG-SG < 2.5 Å is treated as a covalent bond (non-bonded exclusions applied; dynamic, discontinuous at 2.5 Å).
- Exclusions: 1-2/1-3 pairs; intra-residue pairs whose distance does not depend on any chi (rigid groups: they
  only add per-aa constants, absorbed by reference energies; all Pro intra pairs); backbone–backbone pairs of
  peptide-bonded neighbours (fixed by the backbone, aa-independent).
- Deterministic: dense block evaluation, plain tensor sums, `index_put` without accumulation. Forward bitwise
  reproducible; gradients bitwise reproducible (also under `torch.use_deterministic_algorithms(True)`).

## Parameter sources (params.py)
- vdW: CHARMM19 united-atom (param19; Brooks 1983, Neria et al. 1996): Rmin/2, eps; 1-4 values for carbons.
- Solvation: EEF1, Lazaridis & Karplus 1999 Proteins 35:133 Table I (V, dG_ref, dG_free, lambda 3.5/6.0 Å).
- Verification: no copy of param19.inp / solvpar.inp was reachable (CHARMM distribution only; web search
  found none). dG_ref/dG_free/lambda were cross-checked against the EEF1 table in PLUMED
  (`src/colvar/EEFSolv.cpp`, same Lazaridis–Karplus dG values for CHARMM22/36 types): all agree except SH1E
  (free Cys, -2.05 recalled; PLUMED's S is -3.24), flagged in `params.DG_VERIFIED`. Volumes and vdW values
  are CHARMM19 numbers from memory (`verified=False` in the docstring). They enter only through term-level
  weights; PI may want them checked against a CHARMM c28+ distribution.
- Type merges (20-type vocabulary): CA->CH1E (Gly too), C_aro->CR1E (incl. ring C without H), C_sp2->C,
  N_his->NR, N_trp->NH1, N_arg->NC2 (NE too), O_amide->O, OXT->OC, S_cys->SH1E, S_met->S.

## Validation
(a) Gradients (45a; 1BNL, 40 residues, float64, 0.05 Å noise, central FD h=1e-4): max relative error
<= 9e-7 for every term with nonzero gradient (hb_sc_sc and disulf have no instance in that region; covered by
the targeted FD tests in tests/test_energy.py). float32 vs float64 totals differ by <= 7e-5 kcal/mol.

(b) Native structures (45b; 30 FireProt X-ray structures, 60–450 residues):
- Backbone H-bonds (E < -0.5) vs Kabsch–Sander map with DSSP-style H: precision 0.976, recall 0.986
  (4094 DSSP H-bonds). Differences come from H placement (bisector vs DSSP convention) and the -3 floor/switch.
- EEF1 per-atom solvation fraction dG_i/dG_ref_i vs freesasa atom SASA: Spearman 0.825 (polar atoms,
  n=16893), 0.866 (apolar, n=10630). Per-residue (symmetric decomposition) desolvation vs rSA: apolar 0.51,
  polar -0.30 (backbone N/O dominate the polar part).
- Per-residue energies (mean, kcal/mol; `45b_by_aa_burial.csv`): buried hydrophobic LJ -7.9 (atr -11.0,
  rep +3.1), apolar desolvation -0.66, polar desolvation +6.0 (backbone), H-bonds -2.1, elec 0;
  exposed charged: elec_dh -0.04, elec_dd -0.31 (~neutral), LJ -3.7, H-bond -0.9. Buried > exposed for
  LJ attraction and apolar burial in every aa; Trp/Phe/Tyr/Arg most favourable LJ.
- Note: lj_rep has a positive floor (~1.5–3 kcal/mol per residue) from 1-4 and close native contacts
  (CHARMM19 united-atom radii are generous for heavy-atom-only crystal contacts) — see (c).

(c) Native vs perturbed chis (45c; 3425 well-ordered residues, 27,400 perturbations; rigid rotations of
downstream atoms). Fraction of perturbations where the native scores better (ties excluded):
| term | large (±120°) | small (15–40°) | large, clash-free | small, clash-free |
|---|---|---|---|---|
| lj_rep | 0.93 | 0.85 | 0.30 | 0.28 |
| lj_atr | 0.41 | 0.49 | 0.72 | 0.73 |
| solv_apolar | 0.31 | 0.38 | 0.66 | 0.70 |
| solv_polar | 0.78 | 0.65 | 0.29 | 0.29 |
| hb (all) | 0.75 | 0.75 | 0.69 | 0.71 |
| elec_dh | 0.63 | 0.55 | 0.66 | 0.61 |
| disulf | 0.99 | 1.00 | 0.94 | 1.00 |
| unit-weight total | 0.95 | 0.88 | 0.47 | 0.44 |
| total without rep | 0.82 | 0.73 | 0.67 | 0.66 |
"clash-free" = perturbations with dE_rep < 0.5 kcal/mol (attraction/solvation/H-bonds only).
Reading: repulsion drives recognition of clashing alternatives; in the clash-free set attraction, apolar
burial, H-bonds and electrostatics each prefer the native in 66–73 % of cases, whereas polar desolvation
(perturbed polar groups pointing to solvent) and repulsion (native contacts slightly inside CHARMM19 Rmin)
prefer the decoy, so unit weights are near chance there. Term weights must be fitted (polar desolvation
needs to be balanced by H-bonds, as in all EEF1-type functions); w14 (0/0.5/1) and cap (1/3/10) barely change
the fractions. Repulsion-radius sensitivity: over all perturbations the unit-weight total prefers the native in
0.945 / 0.881 (large / small) with rep_scale 1.0, 0.944 / 0.893 with 0.9, 0.919 / 0.818 with 0.8; with 0.9
repulsion prefers the native in 61–63 % of its own clash-free set (vs 28–30 % at 1.0). So the CHARMM19
radii are ~10 % too large for heavy-atom-only crystal contacts. Default kept at 1.0 (literature value);
rep_scale = 0.9 is the recommended candidate, PI decision (structure-only calibration, no ΔΔG data used).

(d) Speed (45d; RTX 4090, float32, shared GPU):
| test | time | throughput |
|---|---|---|
| whole-protein decomposition, 295 res / 2358 atoms, forward | 9.9 ms | 101 / s |
| same, forward + backward | 29 ms | 34 / s |
| 10k variants x 3 flexible residues (37 atoms) vs 2000 env atoms (one- + two-body) | 0.95 s | 10.5k variants/s |
| same, `enable_compile()` | 0.41 s | 24.7k variants/s |
| 10k variants of one site (random aa + chis via E1 builder) vs 2344-atom protein | 0.19 s | 53k variants/s |
| same, `enable_compile()` | 0.12 s | 83k variants/s |
Main speed devices: per-site neighbour lists shared across variants, trimming empty slot tails (sorting variants
by size within a site), compact charged-atom lists for electrostatics.

## Open issues
1. Parameter verification (volumes / vdW / SH1E dG) against an actual param19/solvpar file.
2. lj_rep floor and clash-free repulsion preference: rep_scale 0.9 fixes most of it (45c); PI to decide
   whether to adopt it as the default.
3. sp2_2H donors (Asn/Gln/Arg NH2) use a cone window, the in-plane H constraint is ignored; His tautomers both
   donor+acceptor; Asn/Gln/His flips not resolved (packer's job).
4. EEF1 cutoff 9 Å truncates lambda = 6 Å charged-group Gaussians (EEF1 standard practice).
5. Dynamic disulfide detection is discontinuous at SG-SG = 2.5 Å.
6. StatTerms waits for E1's tables.

## Requests to E1
- R1: add an explicit intra-residue bond list (e.g. `topology.BONDS[aa]`) so energy.py can drop its private
  `_SC_BONDS` (currently tested for consistency with `IC_SPEC`).
- R2: rotamer-library module: expose `RAMA_GRID` [20 (+ pre-Pro rows), n, n] as -kT ln P(phi,psi|aa) on a
  periodic grid starting at -180°, and `SC_ENTROPY` [20] (kcal/mol); `StatTerms.from_library()` picks them up.
- R3: builder output → energy: `blocks_from_arrays` takes atom14 + OXT; packer variants can build `Blocks`
  directly from `build_sidechains` output (see scripts/45d_energy_speed.py, OXT slot appended as absent).
