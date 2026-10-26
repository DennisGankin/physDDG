# Engine: energy function (E2) — physddg/params.py, physddg/energy.py

Status 2026-10-26 (v2 after PI follow-ups: verified parameters, D14 rep_scale 0.9, StatTerms wired).
Tests: `tests/test_energy.py` (23 pass). Validation scripts `scripts/45a-d_*.py`, outputs in
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
- `StatTerms()` (E1's physddg/data/pdbstats.npz via rotlib.PDBStats), torch/differentiable, kcal/mol (nats x kT
  0.5925): `rama(aa, phi, psi, next_is_pro=None)` (pre-Pro map for non-Gly/Pro), `env(aa, ss3, cn)` (D4
  environment potential, cn = structure.smooth_cn), `sc_entropy(aa, ss)`. Tested equal to PDBStats.
- `enable_compile()` optional torch.compile of the short-range kernel (~2x; ~40 s compile per shape pattern).
Consistency: local/env results equal the whole-structure decomposition to < 4e-6 kcal/mol (tested).

## Functional forms (constants in `params.EnergyParams`, frozen)
- LJ (CHARMM form eps[(Rmin/r)^12 - 2(Rmin/r)^6]) split at Rmin (WCA): attraction = -eps inside Rmin, LJ beyond,
  CHARMM-switched 6->8 Å; repulsion = eps(u-1)^2, u=(Rmin/r)^6, linear continuation where it exceeds
  `rep_cap` (3 kcal/mol/pair default; analytic r_c, C1-continuous). `rep_scale` = 0.9 scales repulsive
  radii (D14, structural calibration on native chis only, see (c)).
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

## Parameter sources (params.py) — all verified
- vdW: CHARMM19 united-atom param19.inp NONBONDED (Rmin/2, eps; carbon 1-4 values eps 0.1 / Rmin/2 1.9):
  verified against a public copy of param19.inp (VMD trunctraj plugin, CVS mirror on GitHub).
- Solvation: EEF1 Table I (Lazaridis & Karplus 1999) for CHARMM19 types: V, dG_ref, dG_free, lambda, R_i verified
  against OSPREY3 `config/eef1parm.dat` (verbatim Table I incl. dH/dCp); dG_ref also agrees with PLUMED
  `EEFSolv.cpp`. `scripts/44_fetch_energy_param_sources.sh` downloads the three files (data/raw/param_sources,
  SHA256SUMS) and `tests/test_energy.py::test_params_match_sources` checks every number.
- Corrections found by the verification: dG_free ≠ dG_ref in EEF1 for most types (I had set them equal from
  the CHARMM22 port): CH1E -0.25, CH2E 0.52, CH3E 1.50, NH1 -8.9, NR -4.0, NH2 -7.8, N -1.55, OH1 -6.7, O -5.85,
  S -4.1, SH1E -2.7; NH2 dG_ref -5.45 (was -5.95). Radii, well depths, volumes, SH1E dG_ref were correct.
  The paper itself (Wiley) could not be fetched (Cloudflare); the OSPREY file reproduces its table.
- Type merges (20-type vocabulary): CA->CH1E (Gly too), C_aro->CR1E (incl. ring C without H), C_sp2->C,
  N_his->NR, N_trp->NH1, N_arg->NC2 (NE too), O_amide->O, OXT->OC, S_cys->SH1E, S_met->S.

## Validation
Numbers below are for the final parameters (verified values, rep_scale 0.9).

(a) Gradients (45a; 1BNL, 40 residues, float64, 0.05 Å noise, central FD h=1e-4): max relative error
<= 9e-7 for every term with nonzero gradient (hb_sc_sc and disulf have no instance in that region; covered by
targeted FD tests in tests/test_energy.py). float32 vs float64 totals differ by <= 7e-5 kcal/mol. Forward and
backward bitwise deterministic.

(b) Native structures (45b; 30 FireProt X-ray structures, 60–450 residues):
- Backbone H-bonds (E < -0.5) vs Kabsch–Sander map with DSSP-style H: precision 0.976, recall 0.986
  (4094 DSSP H-bonds). Differences come from H placement (bisector vs DSSP convention) and the -3 floor/switch.
- EEF1 per-atom solvation fraction dG_i/dG_ref_i vs freesasa atom SASA: Spearman 0.859 (polar atoms,
  n=16893), 0.868 (apolar, n=10630). Per-residue (symmetric decomposition) desolvation vs rSA: apolar 0.51,
  polar -0.35 (backbone N/O dominate the polar part).
- Per-residue energies (mean, kcal/mol; `45b_by_aa_burial.csv`): buried hydrophobic LJ -10.6 (atr -11.0,
  rep +0.4), apolar desolvation -0.91, polar desolvation +7.7 (mostly backbone), H-bonds -2.1, elec 0;
  exposed charged: elec_dh -0.04, elec_dd -0.31 (~neutral), LJ -5.9, H-bonds -0.9. Burial makes LJ attraction
  and apolar burial more favourable for every aa; Trp/Phe/Tyr/Arg most favourable LJ. With rep_scale 0.9 the
  native repulsion floor dropped from ~1.5–3 to ~0.2–0.45 kcal/mol per residue.

(c) Native vs perturbed chis (45c; 3425 well-ordered residues, 27,400 perturbations; rigid rotations of
downstream atoms). Fraction of perturbations where the native scores better (ties excluded):
| term | large (±120°) | small (15–40°) | large, clash-free | small, clash-free |
|---|---|---|---|---|
| lj_rep | 0.93 | 0.86 | 0.61 | 0.63 |
| lj_atr | 0.41 | 0.49 | 0.64 | 0.63 |
| solv_apolar | 0.31 | 0.39 | 0.59 | 0.59 |
| solv_polar | 0.78 | 0.65 | 0.44 | 0.44 |
| hb (all) | 0.75 | 0.75 | 0.66 | 0.69 |
| elec_dh | 0.63 | 0.55 | 0.64 | 0.59 |
| disulf | 0.99 | 1.00 | 0.97 | 1.00 |
| unit-weight total | 0.94 | 0.89 | 0.70 | 0.71 |
| total without rep | 0.83 | 0.74 | 0.69 | 0.67 |
"clash-free" = perturbations with dE_rep < 0.5 kcal/mol under the same setting.
Reading: repulsion drives recognition of clashing alternatives; among clash-free alternatives attraction,
repulsion, apolar burial, H-bonds and electrostatics each prefer the native in 59–69 % of cases; only polar
desolvation prefers decoys (perturbed polar groups pointing to solvent; must be balanced by H-bond weight).
Sensitivity (unit total, large / small): rep_scale 1.0 0.944 / 0.877, 0.9 0.944 / 0.889, 0.8 0.920 / 0.819;
w14 0/0.5/1 and cap 1/3/10 change fractions by <= 0.016. With rep_scale 1.0 repulsion preferred the decoy in
the clash-free set (0.28–0.30) — the basis of D14.

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
1. sp2_2H donors (Asn/Gln/Arg NH2) use a cone window, the in-plane H constraint is ignored; His tautomers both
   donor+acceptor; Asn/Gln/His flips not resolved (packer's job).
2. EEF1 cutoff 9 Å truncates lambda = 6 Å charged-group Gaussians (EEF1 standard practice).
3. Dynamic disulfide detection is discontinuous at SG-SG = 2.5 Å.
4. Ring carbons without H merged into CR1E (EEF1 has C/CR for them); could be split if E1 adds a type.

## Requests to E1
- R1 (open): add an explicit intra-residue bond list (e.g. `topology.BONDS[aa]`) so energy.py can drop its
  private `_SC_BONDS` (currently tested for consistency with `IC_SPEC`).
- R2 (done 10-26): statistical tables — wired via `StatTerms` to rotlib.PDBStats.
- R3: packer variants can build `Blocks` directly from `build_sidechains` output (see scripts/45d_energy_speed.py,
  OXT slot appended as absent). Requests from mutate.py go to notes/engine_mutate.md; none posted yet.
