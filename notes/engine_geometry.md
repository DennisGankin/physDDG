# Engine geometry (E1): topology, structure prep, rotamer library, PDB statistics, NeRF builder

Status 2026-10-26. Code: `physddg/{topology,structure,builder,rotlib}.py`; data: `physddg/data/{rotlib_bbdep,pdbstats}.npz`
(5.8 + 0.24 MB); scripts `40_build_rotlib.py` (extract + build, ~1 min on 48 CPUs + 1 GPU), `41_ideal_geometry.py`,
`42_validate_geometry.py`, `43_dssp_crosscheck.py`; tests `tests/test_geometry.py` (11 pass).

## Decisions
- **G1 Rotamer library is derived, not Dunbrack.** The Dunbrack 2010 bbdep library is free for academic use only
  under its own licence and cannot be redistributed in our repo; ours is derived reproducibly from the PISCES list
  (licence-free). It matches Dunbrack-style definitions (3 sp3 bins; non-rotameric terminal chis in 30° bins; 10° φ/ψ grid).
- **G2 Data set**: PISCES cullpdb pc30, resolution ≤ 1.8 Å, R ≤ 0.25, X-ray, list of 2026-10-20 (9266 chains; all
  found in the local mirror). Individual list files return HTTP 403, so the script fetches the dated
  `pisces_lists_*.tar.gz` (2 GB, deleted after extracting the one list). Chains are split by md5(PDB id): 90% train
  (8362 chains, 2.02 M residues: library + statistics), 10% test (904 chains: all validation below).
- **G3 Residue filters**: no altloc in the residue, all occupancies ≥ 0.99, backbone B < 40, not a modified residue,
  φ and ψ defined; for side-chain statistics additionally complete side chain and max side-chain B < 40.
- **G4 Ideal geometry** = medians over well-ordered residues (side-chain B < 30) of 1500 train chains. Bonds agree with
  Engh & Huber to ≤ 0.005 Å, angles ~1°. Per-aa CB geometry (Pro CB: N-CA-CB 103.2°). Planar offsets snapped to 0/180;
  branch offsets measured (Ile CG2 −124.2, Leu CD2 +123.2, Thr CG2 −121.1, Val CG2 +123.4 relative to chi).
  Table: `results/geometry/ideal_ic_measured.csv`.
- **G5 Smoothing**: von Mises kernels on φ/ψ, σ 10° (fine) backed off to σ 30° and then to the backbone-independent
  distribution (pseudo-count 10 for probabilities, 5 for chi moments); sd floor 3°. Rotamers seen < 5 times are dropped
  (their mass is dropped; P sums to ≥ 0.99 except Lys ≥ 0.96, Arg ≥ 0.92 at the sparsest grid cells). Rotamers per aa: C/S/T/V 3, P 2, I/L 9, D/F/Y 18, M 24, W 30, H 35,
  N 36, E 53, K 63, R 68, Q 98.
- **G6 DSSP**: own Kabsch–Sander implementation (8 states, classic priority, no β-bulge merging), sparse (KD-tree) so
  it handles 5000-residue chains. Versus pydssp on 26k test residues: 99.4% of our H and 99.7% of our E are H/E in
  pydssp; the remaining disagreement is our G (3-10) and B (isolated bridge) that pydssp lumps into H/E.
- **G7 Atom typing** (vocabulary fixed for E2): CB of Ile/Thr/Val = CH1, Ala CB = CH3, Pro CD = CH2, Gly CA = CA, His
  ring C = C_aro, Phe/Tyr CG = C_aro, OXT = O_carboxyl. His both ring N flagged donor and acceptor (tautomer unresolved),
  neutral by default. Cys SG / Met SD are not H-bonding.
- **G8 Packing (PI D6)**: mutated site uses library rotamers only for WT and mutant; neighbours may add their native
  rotamer (`RotamerLibrary.assign` gives its library index; native chis come from `structure.side_chain_chis`).
- Use `builder.build_sidechains(..., cb=native_CB)` for WT-site rebuilding and mutants of non-Gly residues: keeping the
  native CB halves the rebuild error (below). For Gly→X use the ideal CB.
- Leakage note: the statistics come from a pc30 PDB set that will contain homologues of benchmark proteins. These are
  generic conformational statistics (same as any rotamer library / Rama potential), not fitted to ΔΔG.

## Validation (904 held-out test chains, 159k well-ordered side chains; `results/geometry/validation.json`)
Builder: rebuild native side chains from native backbone + native chis (heavy atoms CB onwards, min over exact symmetry swaps).

| CB used | median RMSD | mean | 95th pct | fraction < 0.15 Å |
|---|---|---|---|---|
| ideal CB | **0.101 Å** | 0.136 | 0.369 | 0.69 |
| native CB | **0.042 Å** | — | 0.204 | — |

Per aa (median, ideal / native CB): A .036/0, S .068/.016, V .079/.029, T .085/.030, P .072/.039, C .084/.029,
I .106/.042, D .109/.039, N .109/.034, L .113/.058, Q .139/.070, M .143/.078, E .144/.076, H .146/.058, K .169/.093,
F .169/.073, Y .217/.131, R .221/.133, W .235/.144 (`rmsd_per_aa.csv`). Long/aromatic residues are limited by natural
spread of CA-CB-CG angles (sd 1.6–2.5°), not by the builder. Old structures deviate more (1UBQ: 0.19 / 0.11 Å).

Rotamer coverage: native side chain matched within 30° on all chis (2-fold symmetric terminal chis modulo 180°),
pooled over the 18 aa with chis (weighted by residues), library from train chains only:

| candidates | top1 | top3 | top5 | top10 | top20 | all | cum 0.90 | cum 0.95 | cum 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| no expansion | 0.591 | 0.810 | 0.867 | 0.930 | 0.966 | 0.984 | 0.946 | 0.966 | 0.981 |
| ±1σ χ1/χ2 expansion (×9) | 0.601 | 0.820 | 0.876 | 0.937 | 0.971 | 0.990 | 0.953 | 0.973 | 0.987 |

Mean base rotamers kept per residue: 7.7 (cum 0.90), 9.8 (cum 0.95), 14.3 (cum 0.99). Worst: Arg (top20 0.78, all 0.90),
Lys (0.87 / 0.91), Gln (0.82 / 0.98) — four/three chis at a strict 30° criterion. Per aa: `coverage_per_aa.csv`.

Throughput (RTX 4090, float32, 2000 positions × 40 rotamers): 24.5 M residue-rotamers/s with mixed residue types
(Trp-length worst case), 56 M/s for Leu. Rotamer query (CPU): 300 positions × 20 aa in 18 ms (41 ms with expansion).
Structure load + annotation incl. DSSP + SASA: ~0.2 CPU-s per chain on average (9266 chains in 38 s on 48 CPUs).
Robustness: all 1292 project structures (Megascale, FireProt, 300 yeast AF) load and annotate; 7 FireProt files with
misplaced ANISOU records are read through a fallback that strips ANISOU.

PDB statistics sanity: Leu in helix mt 59% / tp 36%; Val in helix t 91%; Pro endo/exo 61/39 at PPII.
E_rama (−ln P(φψ|class)/P(φψ)): Gly at αL −0.87, Gly at (80,−170) −2.55, Pro at β(−120,130) +6.1.
P(aa | φ>0): Gly 0.66, Asn 0.083, Asp 0.051; P(φ>0 | aa): Gly 0.53, Asn 0.12, Asp 0.052, His 0.046.

## API summary
- `topology`: `ATOM_TYPES`, `TYPE_INDEX`, `AA1/AA3`, `TEMPLATES[aa]` (atoms, types, type_ids, charges, is_sidechain, chis),
  `atom_type(aa, atom)`, `atom_charges(aa, his_charged, nterm, cterm)`, `hbond_roles(aa)` → {atom: HB(donor nH, acceptor,
  base, base2, don_geom, acc_hyb)} ("-C" = C of i−1), `CHI_ATOMS`, `CHI_SYMMETRIC`, `CHI_AMBIGUOUS`, `SYMMETRIC_SWAPS`,
  `AMBIGUOUS_SWAPS`, `IC_SPEC`, `IDEAL_IC`, `CB_IC`, `MAX_ATOMS = 14` (atom14 layout = template order).
- `structure`: `load_protein(path, chains=None, keep_het=True)` → `Protein` (chain, resnum str with icode, seq, aa,
  resname, modified, xyz14/mask14/b14/occ14, oxt, missing_sc, missing_o, altloc, het_*; `.index(chain, resnum)`,
  `.atoms()` flat view, `.bfactor` = CA B / pLDDT). `annotate(p)` fills `p.ann`: phi, psi, omega, cis, chain_break, chi
  [L,4], ss8, ss3, disulfides, ncb8/10/12, ncen8/10/12, cn (smooth CB contact number), hse_up/down, sasa_atom14, sasa,
  sasa_sc, rsa (Tien 2013 max ASA). Helpers: `dihedral`, `place_atom`, `ideal_cb`, `backbone_h` (bisector or DSSP
  style), `breaks`, `backbone_dihedrals`, `side_chain_chis`, `disulfides`, `dssp`, `sasa`, `burial`,
  `smooth_cn(x, r0=10, width=1)` (numpy KD-tree or differentiable torch).
- `builder` (torch): `build_sidechains(bb [B,3|4,3], aa int|[B]|[B,R], chis [B,R,4] deg, cb=None)` → (xyz [B,R,14,3],
  mask [B,R,14]); `nerf`, `ideal_cb`, `amide_h(c_prev, n, ca)`, `donor_h(D, B1, B2, geom)` for bisector / sp2_2H donors.
- `rotlib`: `RotamerLibrary().query(aa, phi[B], psi[B], max_r=None, cum=None, expand=False)` → chis [B,R',4],
  logp [B,R'] (−inf for padding), mask, rot, sd; `.tables(aa, phi, psi)`; `.assign(aa, chis)`; `.n_rot(aa)`.
  NaN φ/ψ → backbone-independent distribution. `PDBStats()`: `rama_energy(cls, phi, psi, absolute=False)` (cls = aa
  letter or general/gly/pro/prepro), `rama_energy_all_aa(phi, psi, next_is_pro)` [B,20],
  `env_energy(ss3, cn=, rsa=, phi=, psi=, variant="cn_ss"|"cn_ss_bb"|"rsa_ss")` [B,20] with E = −ln P(aa|env)/P(aa)
  (8 equal-population cn bins × DSSP H/E/L [× basin αR/β/φ>0]; RSA bins 0/.05/.1/.2/.3/.4/.5/.7; pseudo-count 50·P(aa);
  counts stored as `env_*_counts`), `sc_entropy(kind, ss)` [20] nats (`sc_entropy_rotameric` = rotameric chis only,
  `sc_entropy` = all library bins, `sc_entropy_bbdep` = mean entropy of P(rot|φψ)), `d["p_aa_given_posphi"]`,
  `d["p_posphi_given_aa"]`, `d["posphi_energy"]`. All energies in kT units (×0.593 kcal/mol).

## Open issues
- Side-chain entropy depends on the bin definition (fine 30° terminal bins add ~1–2 nats for D/E/F/H/N/Q/W/Y);
  the rotameric-only variant is the more physical "rotamer count" measure. Pick one by CV in fitting.
- Env potentials use backbone-quality residues incl. those with missing side chains (avoids depleting exposed K/E);
  RSA of those residues is slightly underestimated. Statistics are from isolated chains (no partner chains in SASA/cn).
- No β-bulge handling in DSSP; His tautomer / flips (Asn/Gln/His) are not resolved (flagged in topology only).
- Termini charges are available via `atom_charges(nterm=, cterm=)` but `Protein` does not mark termini; use
  `ann["chain_break"]` (and residue before a break) to identify them.
