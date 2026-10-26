# PhysDDG — Literature and landscape review

Compiled 2026-10-26 (literature searched up to 2026-10). Author: lit agent.
Companion notes with longer extracts: `notes/lit_datasets.md`, `notes/lit_sota_numbers.md`,
`notes/lit_alphafold.md`, `notes/lit_speed_algorithms.md`.

Conventions used throughout this file: **ΔΔG > 0 = destabilising** (project decision D1).
Where a source uses the opposite sign, it is stated explicitly.
Everything marked **[unverified]** could not be checked against a primary source; treat as a lead, not a fact.
Numbers labelled **[own]** were computed here (scripts in `scripts/lit_*.py`), not taken from a paper.

---

## 0. Executive summary (read this first)

1. **The bar is low and high at the same time.** On the Megascale/ThermoMPNN test split, native
   FoldX 5.1 gets pooled Spearman **0.569** (per-protein median 0.633) **[own]**, while
   burial alone (1 − RSA) already gets **0.461 / 0.514** and a 7-parameter "trivial" model
   (burial, burial×Δhydrophobicity, burial×ΔVolume, Pro/Gly indicators) gets **0.536 / 0.600** **[own]**.
   Deep models sit at 0.725 (ThermoMPNN) – 0.74 (SPURS) pooled, 0.77–0.83 per-protein. So "beating FoldX"
   is worth ~0.03–0.07 Spearman over a model that contains no physics at all; the project is only interesting
   if it lands clearly above ~0.62 pooled. The internal PhysDDG v1.2 is at 0.640 — that, not FoldX, is the bar.
2. **You cannot get there by reweighting FoldX-style terms.** Refitting free linear weights on FoldX 5.1's
   own 22 per-term energies under protein-grouped 5-fold CV gives pooled 0.541 / per-protein 0.604 —
   *worse* than stock FoldX; with clash terms log-compressed, 0.567 / 0.628, i.e. a tie **[own]**.
   The missing accuracy is in the *terms and the sampling*, not in the weights.
3. **Reference (unfolded) state is the cheapest real win for calibration, not for ranking.** A 20-coefficient
   amino-acid-identity "mass-balance" correction (Rossi 2025) on top of FoldX lifts Pearson
   0.417 → 0.633 and cuts RMSE 0.96 → 0.82 kcal/mol on Megascale-test, while leaving per-protein
   Spearman unchanged (0.633) **[own]**. If the deliverable is calibrated kcal/mol (it should be), this term is mandatory.
4. **7 of FoldX's 22 terms are identically zero** on a monomeric single-mutant benchmark (loop entropies,
   water bridge, disulfide, electrostatic kon, partial covalent, complex entropy) and three more are
   effectively noise (cis-bond, ionisation, helix dipole: |ρ| ≤ 0.02) **[own]**. A PhysDDG engine needs ~8–10 terms, not 22.
5. **Megascale is the only dataset big enough to fit on, and it is the one with the most severe
   range and homology caveats** (assay dynamic range −1…5 kcal/mol, 28 test proteins only, and at least
   two test domains 98% identical to a training domain). FireProt/S669/Ssym are small, 80–95% destabilising,
   and partially contaminate every published predictor. Generalisation claims must be made on
   Domainome + S669/Ssym + FireProt-HF simultaneously, with the Lehner ddPCA kcal/mol sets (KRAS, PDZ, SH3)
   as the cleanest external check (§3.8).
6. **Two free wins a physics model gets and DL does not.** (a) **Antisymmetry by construction** — write ΔΔG as a
   difference of folded and unfolded energies and r(dir,inv) = −1, ⟨δ⟩ = 0 come out exactly; FoldX manages only
   −0.38 / −0.47, and training penalties demonstrably fail to fix it. (b) **Speed with interpretability** —
   but note KORPM (12 parameters, backbone-only, ~5 ms/mutation on one CPU core, best Ssym PCC 0.69) already
   occupies that slot, so speed alone is not a contribution.
7. **The closest prior art to the engine already exists and is not DL**: `tmol` (uw-ipd, Apache-2.0) is
   Rosetta's energy, Dunbrack packer, minimiser and FastRelax in batched PyTorch + CUDA. Evaluate it before
   writing kernels (§5.4).

---

## 1. Physics and empirical-energy predictors

### 1.1 FoldX (FOLDEF)

Primary: Guerois, Nielsen & Serrano, *J Mol Biol* 320:369–387 (2002), DOI 10.1016/S0022-2836(02)00442-4;
server/suite paper Schymkowitz et al., *Nucleic Acids Res* 33:W382–W388 (2005), DOI 10.1093/nar/gki387.
Current release used here: FoldX 5.1 (`/workspace/foldx_native/bin/foldx`).

**Energy function (eq. 1 of Guerois 2002), with the published fitted weights:**

```
ΔG = 0.20·ΔG_vdw + 1.40·ΔG_solvH + 1.25·ΔG_solvP + ΔG_wb + ΔG_hbond + ΔG_el
     + 1.00·T·ΔS_mc + 0.75·T·ΔS_sc
```

* Weights were fitted by grid search on **339 single mutants in 9 proteins** (barnase, CI2, spectrin,
  Src SH3, Sso7d, tenascin, FKBP, Ada2h, CheY), then blind-tested on 667 mutants + 82 complex mutants
  (1088 total). Reported: r = 0.83, σ = 0.81 kcal/mol, slope 0.76 over 95% of the database.
* ΔG_hbond was itself fitted: **−1.3 kcal/mol** polar–polar, **−1.4 kcal/mol** polar–charged
  (H-bond accepted on distance ≤ 3.6 Å plus angular criteria).
* **Solvation / burial is computed with an occupancy model, not SASA** — for atom *i*,
  `Occ(i) = Σ_{j, d<6 Å} V_j · exp(−d_ij²/2σ²)` with σ = 3.5 Å and fragmental volumes V_j; this is
  the key speed trick (no geometric surface calculation). Every term (solvation, vdW, electrostatics,
  entropies) is then scaled linearly between per-atom-type Occ_min/Occ_max limits (Holm & Sander):
  `S_fact(i) = (Occ − Occ_min)/(Occ_max − Occ_min)`, clamped to [0,1].
* Atomic parameters from transfer free energies: vdW **−0.082 kcal/mol per Å³** of buried atomic volume
  (vapour→water minus water→cyclohexane); solvation **−0.0314 kcal/mol·Å⁻³** for apolar atoms,
  **+0.8 kcal/mol** polar, **+1.44 kcal/mol** charged. Polar atoms that reach their maximum H-bond count
  are treated as fully desolvated (max penalty regardless of Occ).
* Electrostatics: Coulomb with Debye–Hückel screening, `E = 332 q_i q_j / (ε d_ij) · exp(−d_ij·K)`,
  **ε ramped 8 → 80 with S_fact**, I = 0.05 M, T = 298 K, standard pH-7 charges, 20 Å cutoff,
  only Asp/Glu/Arg/Lys/His + termini.
* Backbone entropy from secondary-structure propensities, with an explicit loop correction
  (÷3 if neither neighbour makes a backbone–backbone H-bond, ÷2 if one does; ÷2 for residues preceding Pro).
  Side-chain entropy scaled by S_fact, but charged at **full cost whenever the side chain makes an H-bond
  or salt bridge**; if entropy cost exceeds the interaction gain, both are dropped.
* Water bridges: explicit pseudo-waters placed geometrically, kept only if ΔG_wb < 0;
  S_wat(buried) = 0.92 kcal/mol, S_wat(exposed) = 2.5 kcal/mol.
* **Unfolded state is implicit, absorbed into the weights.** The authors say so explicitly, and they
  interpret W_sc = 0.75 as residual side-chain ordering in the unfolded state and W_vdw = 0.2 as
  compensation by vdW contacts with solvent in the unfolded chain. This is the central weakness
  the mass-balance literature (§1.9) attacks.

**Which terms matter (Guerois 2002, Table 4, % of |ΔΔG|):**

| term | buried non-polar | exposed non-polar | buried polar (H-bonded) | exposed polar |
|---|---|---|---|---|
| non-polar solvation | 48.4 | 35.2 | 15.9 | 15.7 |
| polar solvation | 11.2 | 19.7 | 26.9 | 23.2 |
| van der Waals | 21.1 | 18.8 | 9.2 | 8.6 |
| H-bond | – | – | 28.0 | 25.7 |
| electrostatics | – | – | 2.8 | 5.4 |
| main-chain entropy | 7.1 | 17.9 | 3.3 | 4.3 |
| side-chain entropy | 10.5 | 6.0 | 11.0 | 15.1 |

Quote worth keeping: for hydrophobic residues ΔΔG is 55–70% hydrophobic solvation + vdW
("this can explain why a simple method counting the contacts around hydrophobic residues can be quite successful"),
whereas for polar residues "no individual term dominates" — the H-bond + vdW gain almost exactly cancels
the polar desolvation + side-chain entropy cost. **Design consequence: the hard part of ΔΔG is polar/charged
chemistry, and it is a difference of large cancelling numbers.**

**Known weaknesses of FoldX**
* Fixed backbone; only the mutated side chain and neighbours are repacked (BuildModel moves neighbours
  by default). No backbone relaxation ⇒ systematically over-penalises large-to-larger substitutions and
  under-rewards strain relief.
* Strong sensitivity to input structure quality; `RepairPDB` is effectively mandatory and is itself a
  minimisation in FoldX's own force field. Guerois 2002 already reported higher success on structures
  better than 1.5 Å, and that GROMOS minimisation did *not* help ΔΔG.
* No pKa calculation (standard charges at pH 7) ⇒ blind to pH-dependent stability and to buried
  ionisable groups; no explicit treatment of metals/cofactors beyond its molecule library.
* Antisymmetry is poor: on Ssym, FoldX direct r = 0.63, reverse r = 0.39, r(dir,inv) = −0.38,
  bias ⟨δ⟩ = −0.47 kcal/mol (Montanucci 2019, Table).
* Stochastic: FoldX 5.1 is deterministic run-to-run on identical input **[own, from project record]**, but
  results depend on the repaired structure, which varies between FoldX versions (0.76 Å all-atom RMSD
  between a 2025 build and 5.1 on the same inputs **[own, project record]**).
* **Per-mutation uncertainty is enormous.** Sapozhnikov, Patel, Ytreberg & Miller,
  *BMC Bioinformatics* 24:426 (2023), DOI 10.1186/s12859-023-05537-0: on 1,187 mutations
  (672 folding, 515 binding) in 20 systems, the 95% prediction interval for a FoldX folding ΔΔG is
  **±2.9 kcal/mol** (±3.5 for binding). Averaging FoldX over **100 MD snapshots** (1 ns apart from a 100 ns
  trajectory) narrows the median interval from 3.78 → 2.89 kcal/mol. >75% of predicted-destabilising calls
  are right, but **more than half of predicted-stabilising calls (< −0.5 kcal/mol) are actually neutral or
  destabilising**. Design consequences: (i) structural ensemble averaging is a real, measurable accuracy
  lever for a fixed energy function; (ii) a single-structure point estimate should ship with an uncertainty.

**FoldX 5.1 term list as actually emitted** (from our own runs, `/foldx_native_clean/results/*.csv`):
total_energy, backbone_hbond, sidechain_hbond, van_der_waals, electrostatics, solvation_polar,
solvation_hydrophobic, van_der_waals_clashes, entropy_sidechain, entropy_mainchain, sloop_entropy,
mloop_entropy, cis_bond, torsional_clash, backbone_clash, helix_dipole, water_bridge, disulfide,
electrostatic_kon, partial_covalent_bonds, energy_ionisation, entropy_complex.

**[own] Per-term signal on Megascale-test** (28,172 variants / 28 proteins; `scripts/lit_foldx_term_analysis.py`;
Spearman vs experimental ΔΔG, sign-aligned to destabilising-positive):

| term | fraction non-zero | sd (kcal/mol) | Spearman |
|---|---|---|---|
| **foldx_ddg (total)** | 1.00 | 3.30 | **0.569** |
| solvation_hydrophobic | 1.00 | 1.25 | 0.225 |
| van_der_waals | 1.00 | 0.73 | 0.164 |
| van_der_waals_clashes | 0.95 | 3.17 | 0.161 |
| backbone_clash | 0.92 | 0.39 | **−0.123** |
| entropy_mainchain | 0.99 | 0.56 | 0.072 |
| torsional_clash | 0.98 | 0.30 | 0.070 |
| solvation_polar | 1.00 | 0.98 | 0.069 |
| backbone_hbond | 0.90 | 0.50 | 0.058 |
| sidechain_hbond | 0.69 | 1.05 | −0.057 |
| cis_bond | 0.002 | 0.05 | −0.019 |
| energy_ionisation | 0.13 | 0.04 | −0.008 |
| helix_dipole | 0.43 | 0.06 | 0.007 |
| electrostatics | 0.81 | 0.47 | 0.006 |
| entropy_sidechain | 0.99 | 0.80 | 0.001 |
| sloop/mloop entropy, water_bridge, disulfide, electrostatic_kon, partial_covalent, entropy_complex | 0.00 | 0 | n/a |

Reading: the total is far better than any part (0.569 vs best single 0.225) — the terms are genuinely
complementary — but *individually* the polar terms are near-zero-signal on this benchmark, the
side-chain-H-bond term has the **wrong sign**, and FoldX's `backbone_clash` is **anti**-correlated with
true destabilisation, i.e. it fires on modelling artefacts rather than on real strain.
`sidechain_hbond` negative is partly a composition effect (mutations away from H-bonding residues),
but it is a warning that a naive H-bond count term will hurt.

**[own] Headroom of "FoldX reweighted"** (protein-grouped 5-fold CV, linear):

| model | pooled ρ | per-protein median ρ |
|---|---|---|
| FoldX 5.1 as-is | 0.569 | 0.633 |
| free linear refit of its 14 non-constant terms | 0.541 | 0.604 |
| same, clash terms log-compressed | 0.567 | 0.628 |
| forward selection, best 10 terms | 0.538 | — |

⇒ **Reweighting FoldX's own decomposition does not beat FoldX.** Capping the total at +3/+5/+8 kcal/mol
changes ρ by ≤0.003, so the long clash tail is not what limits rank accuracy either.

**[own] FoldX 5.1 throughput on this machine** (64 CPUs, from the project's native-FoldX runs; FoldX is CPU-only,
GPUs are useless for it): FireProt 341 variants in 0.06 h; Megascale-test 28,172 in 1.49 h;
Megascale-train 215,731 in 5.4 h; Domainome 505,024 in 14.18 h with 52 parallel jobs.
That is roughly **5–10 CPU-seconds per mutation** end-to-end (RepairPDB amortised over a protein),
i.e. ~0.1–0.2 wall-seconds/mutation when 52-way parallel. A GPU PhysDDG at 1 ms/mutation would be
~10³–10⁴× faster per core-second; that is the speed claim to aim at and to measure honestly
(same hardware, same structure-preparation included).

### 1.2 Rosetta: ddg_monomer, cartesian_ddg, flex-ddG

* **ddg_monomer** — Kellogg, Leaver-Fay & Baker, *Proteins* 79:830–838 (2011),
  DOI 10.1002/prot.22921. Protocol rows 3 (fixed-backbone, soft-rep, repack-only) and 16
  (limited backbone minimisation with harmonic Cα restraints, 50 models, mean of lowest-3) are the ones
  people cite; row 16 is the accurate/slow one. Reported r ≈ 0.69 on their benchmark **[unverified — row/number
  taken from forum + docs, not from the paper itself]**.
* **cartesian_ddg** — Park et al., *J Chem Theory Comput* 12:6201 (2016) (score function / cart) and
  Frenz et al., *Front Bioeng Biotechnol* 8:558247 (2020) for the protocol.
  Docs: https://docs.rosettacommons.org/docs/latest/cartesian-ddG. Key facts (verified from docs):
  Cartesian-space local relaxation (`-ddg:bbnbrs 1`, i.e. residues i−1,i,i+1), typically 3–5 iterations,
  `ref2015_cart`, `-fa_max_dis 9.0` (must match the pre-relax), output in REU,
  **divide by 2.94 to get kcal/mol**. Needs a cartesian FastRelax of the input first (nstruct ~20).
  The official docs now say outright that ThermoMPNN "may perform better in some situations".
* **flex-ddG** — Barlow et al., *J Phys Chem B* 122:5389–5399 (2018), DOI 10.1021/acs.jpcb.7b11367.
  Backrub ensemble (35k backrub steps, 35 models) + torsion minimisation + generalized additive model
  reweighting of score terms; benchmarked on **1240 interface mutants** (SKEMPI), not monomer stability.
  Relevant to us for two transferable ideas: (i) ensemble averaging over backrub conformers beats a single
  structure, especially for small→large mutations; (ii) **refitting the weights of the energy terms
  in the ΔΔG task** (their GAM) is worth more than changing the sampling.
* Cost: cartesian_ddg is minutes–tens of minutes of CPU per mutation; flex-ddG is ~hours per mutation.
  Precise numbers in `notes/lit_speed_algorithms.md`.

### 1.3 PoPMuSiC / statistical potentials (Rooman–Pucci school)

Dehouck et al., *Bioinformatics* 25:2537 (2009); **PoPMuSiC 2.1**, Dehouck, Kwasigroch, Gilis & Rooman,
*BMC Bioinformatics* 12:151 (2011), DOI 10.1186/1471-2105-12-151. This is the closest published
relative of what PhysDDG should be, and the single most useful template:

* ΔΔG = Σ_i α_i(A)·ΔW_i + α₊(A)·ΔV₊ + α₋(A)·ΔV₋ + α₀(A), i.e. **13 statistical potentials + 2 volume terms + constant**.
* The 13 potentials are built from 4 descriptors — residue type (s), backbone torsion (t), solvent
  accessibility (a), inter-side-chain-centre distance (d): ΔW_st, ΔW_as, ΔW_sd, ΔW_sds, ΔW_stt, ΔW_sst,
  ΔW_aas, ΔW_ass, ΔW_ast, ΔW_asd, ΔW_std, ΔW_asdas, ΔW_stdst. Higher-order couplings did not help.
* **Every coefficient α_i is a sigmoid function of the solvent accessibility of the mutated residue**
  (4 parameters each: inflection, slope, scale, offset). 64 parameters pruned to **52**.
* Volume terms split by sign with a Heaviside function (ΔV₊ = creating a cavity vs ΔV₋ = overpacking)
  — i.e. an explicit asymmetry between large→small and small→large.
* Trained on **2648 mutations / 131 proteins** from ProTherm, excluding Pro mutations, heme proteins,
  pseudo-wild-types and ΔΔG > 5 kcal/mol. 5-fold CV: **r = 0.63, RMSE 1.15 kcal/mol**
  (r = 0.79, RMSE 0.86 after removing the worst 10%).
* Speed: **4,300–9,000 mutations/minute**, full saturation scan of a medium protein in < 1 min.
* **PoPMuSiC-sym** (Pucci, Bernaerts, Kwasigroch & Rooman, *Bioinformatics* 34:3659 (2018),
  DOI 10.1093/bioinformatics/bty348) enforces exact antisymmetry by training on the symmetrised dataset
  and using antisymmetric features; on Ssym it reaches direct r = 0.48 / inverse 0.48, r(dir,inv) = −0.77, ⟨δ⟩ = 0.03.
* Related: BeAtMuSiC (binding ΔΔG, Dehouck 2013), SNPMuSiC, ThermoMuSiC. Same machinery.

### 1.4 DDGun / DDGun3D — the "untrained" baseline everyone must beat

Montanucci, Capriotti, Dehouck, Rooman & Fariselli, *BMC Bioinformatics* 20:335 (2019),
DOI 10.1186/s12859-019-2923-1. Code: https://github.com/biofold/ddgun.

* Sequence version: `s_seq = 0.30·s_Bl + 0.43·s_Sk + 0.27·s_Hp` where s_Bl is a profile-weighted BLOSUM62
  difference, s_Sk a profile-weighted Skolnick pairwise potential over a ±2 sequence window, and
  s_Hp a profile-weighted Kyte–Doolittle hydrophobicity difference.
* 3D version adds a Bastolla–Vendruscolo contact potential over a 5 Å sphere:
  `s_3D = 0.20·s_Bl + 0.29·s_Sk + 0.18·s_Hp + 0.33·s_BV`, the whole thing **multiplied by (1.1 − RSA)**.
  That single multiplicative burial gate is the only structural information it uses beyond contacts.
* Weights are not fitted; they are set proportional to each feature's correlation with ΔΔG on VariBench.
* Multiple mutations: `max + min − mean` of the singles.
* **Exactly antisymmetric by construction** (every term is a difference): Ssym r(dir,inv) = −0.99, ⟨δ⟩ = −0.007/−0.02.
* Performance: S2648 r = 0.50 (seq) / 0.57 (3D); VariBench 0.50/0.54; Broom 0.52/0.62; p53 0.70/0.67;
  myoglobin 0.48/0.57; S669 r = 0.39 (seq) / 0.42–0.48 (3D) depending on the compilation.
  On S461 DDGun3D reaches Pearson ≈ 0.62 and Spearman ≈ 0.58.
* Single-component |Pearson| on S2648 / VariBench / Broom / p53 / myoglobin (their Table 1):
  BLOSUM62 0.284 / 0.269 / 0.354 / 0.636 / 0.148; Kyte–Doolittle 0.298 / 0.248 / 0.263 / 0.143 / 0.500;
  Skolnick 0.387 / 0.398 / 0.423 / 0.328 / 0.454; **Bastolla–Vendruscolo contact potential 0.497 / 0.452 /
  0.581 / 0.423 / 0.548**. A *single residue-level contact potential* reaches Pearson ≈ 0.5 on S2648 —
  that is the bar any atomistic machinery has to clear before it pays for itself.
* **This is the key sanity check for PhysDDG**: a 4-term, zero-trained-parameter model reaches ~85–90% of
  the Pearson of heavily trained predictors on the small benchmarks, with perfect antisymmetry.

### 1.4b KORPM — the closest competitor in spirit (12 parameters, 5 ms/mutation)

Hernández IM, Dehouck Y, Bastolla U, López-Blanco JR, Chacón P. *Predicting protein stability changes upon
mutation using a simple orientational potential.* **Bioinformatics 39(1):btad011 (2023)**,
DOI 10.1093/bioinformatics/btad011. Code: https://github.com/chaconlab/korpm.

* A residue-level **6D orientational knowledge-based potential** (KORP): each residue pair is described by
  the Cα–Cα distance, two polar angles per residue and one torsion, from **three backbone atoms only —
  side chains are ignored entirely**. Pair energy `E = −RT ln(P_obs/P_ref)` with a residue-type-independent reference.
* ΔΔG = Σ over native contacts of (mutant − wild-type) pair energies, each residue's contribution scaled by an
  **amino-acid-dependent weight**; the 20 weights (|w| ≈ 0.7–2.5) are collapsed to **12** by grouping residues
  (DE, VIL, FY, STNQ, KH, …) and fitted by direct MAE minimisation (BOBYQA).
* No explicit unfolded state (the paper says so, and lists it as the main approximation).
* Training: ThermoMutDB + ProTherm filtered to 10–40 °C, pH 5–9, monomers ≤ 2.5 Å, no ligand within 5 Å,
  capped at 15 mutations per amino-acid pair → **2,371 mutations / 129 families**; MMseqs 25% clustering,
  and test proteins with ≥25% identity excluded from the fit.
* Results: CV RMSE 1.38 ± 0.2, PCC 0.54 ± 0.1. **S461 PCC 0.57, RMSE 1.21.** Ssym (684 dir+rev)
  PCC 0.69, RMSE 1.33; direct-only PCC 0.57–0.58. Antisymmetry r(dir,rev) = 0.88, ⟨δ⟩ = −0.15 kcal/mol.
* **Runtime: 684 mutations in 3.5 s on a single i7 core** (≈5 ms/mutation) — ~1000× faster than FoldX.

Why this matters for us: KORPM shows that **12 interpretable parameters over a backbone-only statistical
potential already match or beat FoldX on the classical benchmarks at 1000× the speed**. PhysDDG's claim has
to be that adding real side-chain packing, desolvation and an explicit unfolded state buys accuracy that a
backbone-only potential cannot reach — especially on Megascale (where KORPM has not been evaluated) and on
AF structures. Benchmark against KORPM explicitly; it is cheap to run.

### 1.5 Other classical structure-based predictors (brief)

| method | principle | citation | notes |
|---|---|---|---|
| **MAESTRO** | ML (multi-agent: ANN + SVM + MLR) over statistical/structural descriptors | Laimer et al., *BMC Bioinformatics* 16:116 (2015) | fast, multi-point capable; strongly asymmetric (Ssym r_d−i = −0.34, ⟨δ⟩ = −0.58) |
| **CUPSAT** | torsion-angle + atom-atom mean-force potentials | Parthiban, Gromiha & Schomburg, *NAR* 34:W239 (2006) | old, weak on reverse mutations |
| **SDM / SDM2** | environment-specific substitution tables (knowledge-based, not energetic) | Worth, Preissner & Blundell, *NAR* 39:W215 (2011); Pandurangan 2017 | Ssym direct 0.51, inverse 0.32, ⟨δ⟩ = −0.32 |
| **mCSM / DUET / DynaMut2** | graph-based signatures (atom distance patterns) + ML; DynaMut2 adds normal-mode vibrational entropy | Pires 2014 *Bioinformatics*; Rodrigues, Pires & Ascher, *Protein Sci* 30:60 (2021) | DynaMut2: r = 0.72 (10-fold CV on S4022), 0.68 on S611 (RMSE 1.14). Heavy train/test overlap with ProTherm-derived sets |
| **ENCoM** | coarse-grained normal modes with an amino-acid-type-dependent non-bonded term; ΔΔG from vibrational entropy differences | Frappier & Najmanovich, *PLoS Comput Biol* 10:e1003569 (2014); server *NAR* 43:W395 (2015) | not the most accurate but "the most self-consistent and least biased" on 303 curated mutations — a useful low-bias entropy term |
| **EvoEF / EvoEF2** | physical energy function (vdW, solvation, H-bond, electrostatics, ref state) for design | Huang, Pearce & Zhang, *Bioinformatics* 36:1135 (2020), DOI 10.1093/bioinformatics/btz740 | EvoEF reported to beat FoldX on ΔΔG at **3–5× the speed**; EvoEF2 is retuned for sequence recovery (32.5% on 148 monomers) and is *worse* for ΔΔG than EvoEF. Open source (C), a realistic starting point for term definitions |
| **I-Mutant / INPS / INPS3D** | SVM on sequence/structure features | Capriotti 2005; Fariselli 2015 | sequence-dominated; INPS3D is a common S669 reference point |
| **PremPS, DeepDDG, ThermoNet, ACDC-NN** | ML/DL, see §2 | | |

### 1.6 Which energy terms actually matter

Synthesis of Guerois 2002 Table 4, the DDGun ablation, PoPMuSiC's term list and our own FoldX decomposition:

**Carry most of the signal**
1. **Burial-weighted hydrophobic (apolar) desolvation** — the single largest contributor for non-polar
   mutations (35–48% of |ΔΔG|) and the best single FoldX term on Megascale (ρ 0.225) **[own]**.
   Equivalent to "ΔASA_apolar × 25–30 cal/mol/Å²".
2. **Packing / van der Waals, including an explicit cavity (ΔV₋) vs overpacking (ΔV₊) asymmetry**
   (PoPMuSiC's Heaviside split; FoldX's vdw + clash). Must be soft-capped: raw LJ repulsion is the
   main source of catastrophic outliers.
3. **Side-chain conformational entropy**, scaled by burial, with the FoldX rule that an H-bonded or
   salt-bridged side chain pays the full cost.
4. **Backbone propensity / torsional strain**, with special handling of Gly (extra unfolded-state entropy)
   and Pro (backbone restriction + loss of NH).
5. **H-bonds with explicit geometry** (distance + angles), and the corresponding **polar desolvation
   penalty** — these two must be modelled as a *pair*, since they cancel to ~0.2–0.5 kcal/mol.

**Mostly noise at single-mutation resolution (on monomeric domains)**
* Long-range Coulomb electrostatics between formal charges (ρ ≈ 0.006 in FoldX **[own]**); matters only
  for salt bridges and for pH-dependent effects, and is highly sensitive to the dielectric model.
* Helix dipole, cis-bond, ionisation terms (|ρ| ≤ 0.02 **[own]**).
* Water bridges, disulfides, loop entropies: zero on 100% of our benchmark variants **[own]** — keep the
  capability but do not spend fitting effort there.
* Normal-mode vibrational entropy as a *primary* term (ENCoM is "least biased" but not most accurate);
  possibly useful as a small additive correction.

### 1.7 Unfolded-state / reference-state treatments — the most under-exploited lever

Thermodynamically, ΔΔG = [G_F(m) − G_F(w)] − [G_U(m) − G_U(w)]. Almost all fast structure-based methods
drop the second bracket (FoldX absorbs it in the weights; ProteinMPNN-style scores ignore it entirely).

Four published treatments, in increasing sophistication:

1. **Absorb it into fitted weights** (FoldX). Cheap; explains W_sc = 0.75 and W_vdw = 0.2.
   Breaks whenever amino-acid composition changes systematically (e.g. X→Gly, X→Pro, charged→apolar).
2. **Per-amino-acid reference energies** (Rosetta `ref` weights, EvoEF reference state). 20 numbers fitted
   so that the score reproduces native amino-acid frequencies. Rosetta's are fitted to sequence recovery,
   *not* to ΔΔG, which is a known mismatch.
3. **Mass-balance correction (MBC)** — Rossi, Barducci, Sanavia, Turina, Capriotti & Fariselli,
   *Protein Sci* 34(5):e70134 (2025), DOI 10.1002/pro.70134; arXiv:2504.06806; code
   https://github.com/compbiomed-unito/ddMBC. The correction is deliberately trivial:
   `ΔΔG = a₀·(S(m) − S(w)) + Σ_{i=1..20} a_i·O_i`, where O_i = +1 for the introduced residue, −1 for the
   removed one, i.e. **21 fitted coefficients on top of any existing score**, antisymmetric by construction.
   Fitted on VBS3322 (VariBench ∪ S2648, antisymmetrised), tested on S461. Results: improves ESM-IF1,
   FoldX and Pythia; Pythia/MBC becomes the top performer and beats DDGun3D (PCC 0.62); on the
   independent Megascale set Pythia + MBC gains **+0.07 PCC → ≈0.70 PCC, RMSE 1.43 kcal/mol**.
   Crucially, the fitted 20 coefficients correlate strongly with the **Rose** scale (mean ΔASA on folding)
   and with Kyte–Doolittle, so a 2-parameter variant `a₀·ΔS + a₁·ΔRose` works nearly as well.
   Methods that already see the mutation identity (Stability Oracle, DDGun3D, ACDC-NN) get *worse* with MBC —
   they already model it.
   **[own] replication on Megascale-test, FoldX as the base score** (protein-grouped CV):
   FoldX alone r = 0.417, RMSE 0.96 → +20-coef MBC r = 0.536, RMSE 0.89 → + burial r = 0.594, RMSE 0.85
   → + burial×AA (40 coef) **r = 0.633, RMSE 0.82**, with per-protein Spearman unchanged at 0.633.
   So MBC is a *calibration* fix, not a ranking fix, on Megascale. Expect a larger ranking effect on
   datasets with heterogeneous composition (S669/FireProt) and on reverse mutations.
4. **Explicit unfolded ensembles** — (a) Bastolla and coworkers' statistical-mechanics unfolded-state models;
   (b) alchemical FEP with an explicit unfolded peptide; (c) IFUM, "Protein folding stability estimation with
   explicit consideration of unfolded states", bioRxiv 2025.02.10.637420, published *Nat Commun* (2026),
   DOI 10.1038/s41467-026-68637-4 — models the unfolded ensemble as a Flory random-coil distance-map prior
   and jointly learns ΔG and both ensembles; joint learning "considerably enhances" accuracy over learning ΔG alone.
   (DL, so not directly usable here, but the *representation* — a random-coil contact prior giving each residue
   a sequence-dependent unfolded-state energy — is implementable in a physics engine.)
   (d) "Zero-shot protein stability prediction by inverse folding models: a free energy interpretation"
   (arXiv 2506.05596, 2025) makes the same point for inverse-folding scores and gets gains by sampling
   unfolded-state conformations (MD / BioEmu).

**Practical recipe for PhysDDG**: a sequence-local unfolded-state term of the form
`G_U(aa, local context) = c_aa + c_aa^(neighbour-Pro/Gly) + c_aa·(helix/coil propensity)`, fitted jointly
with the folded-state weights, antisymmetric by construction. ~20–40 parameters, and it is the only part
of the model that can fix the stabilising-mutation bias.

### 1.8 Recent (2023–2026) physics / statistical-potential work on Megascale

* **Rossi et al. 2025** (above) — the only clean "retrofit physics with a reference state" result on Megascale.
* **Pythia** (Sun et al. 2024, *J Chem Theory Comput*? **[unverified venue]**) — self-supervised GNN,
  "ultrafast zero-shot ΔΔG"; used as a potential-like baseline by Rossi 2025. DL, but its *role* in the
  literature (a fast per-residue scorer refined by a 21-parameter correction) is exactly PhysDDG's niche.
* **StrucNS** (PMC13232128, 2025/2026) — interpretable multi-scale topological features trained on
  Megascale; claims interaction-weighted network topology drives absolute stability. **[unverified in detail]**
* **"Large-scale discovery, analysis, and design of protein energy landscapes"** (bioRxiv 2025.03.20.644235) —
  reports that models from **manually engineered interpretable features with explicit structures** were the most
  accurate in one of its comparisons, with PLM embeddings best overall. **[unverified in detail]**
* **PhysDDG v1.2 (Beltrao lab, internal)** — a physics-based, no-DL predictor already evaluated in this group:
  Megascale-test pooled ρ **0.640** (per-protein 0.701), FireProt no significant difference from FoldX,
  Domainome 0.394–0.403 ≈ FoldX 0.392. It beats FoldX (0.569) significantly on Megascale and loses to the
  DL models (~0.72) by 0.06–0.08 **[own, project record; see memory note `physddg-comparison`]**.
  **This is the number the new engine must beat: pooled ρ > 0.64 on Megascale-test, and > 0.40 on Domainome.**
* No published purely-physics GPU ΔΔG engine was found. The nearest claims are ProteinZero's
  unsupervised scorer (matching FoldX PCC 0.60–0.62 at 236–760× the speed, on its own 342-mutation set)
  and ProteinMPNN-ddG (10,000 residues/s on one V100) — both neural. **The "fast + physics + no DL" slot
  is genuinely empty in the literature.**

---

## 2. DL / ML state of the art (what we are being compared to)

The full master table — 45 methods × 12 benchmark columns, with every source cited and unverified cells
flagged — is in `notes/lit_sota_numbers.md`. Condensed here to what changes design decisions.

### 2.1 Headline numbers per benchmark

| benchmark | best physics | best "simple/statistical" | best DL | notes |
|---|---|---|---|---|
| **Megascale test (ThermoMPNN split), pooled ρ** | FoldX **0.57**; Rosetta cartesian_ddg **0.56** | — (KORPM/DDGun not evaluated there) | **SPURS ρ 0.74** (per-protein median **0.83**); ThermoMPNN **0.725 ± 0.003**; RaSP retrained 0.67; Pythia zero-shot 0.602; ProteinMPNN zero-shot 0.49 | cartesian_ddg nearly matches FoldX at ~2,000× the cost |
| **S669 direct, Pearson** | FoldX **0.22**; Rosetta ddg_monomer 0.39 | MAESTRO 0.50; DDGun3D 0.43; PoPMuSiC 0.41 | JanusDDG 0.55; Pythia ρ 0.66; Stability Oracle 0.52; Mutate Everything ρ 0.56 | FoldX is *terrible* here (ρ 0.283) — the classical sets punish its scale and outliers |
| **S461, Pearson** | cartesian_ddg 0.597 (capped); EvoEF2 0.463 | **DDGun3D 0.635** (untrained!); PremPS 0.632; MAESTRO 0.630; KORPM 0.569 | JanusDDG 0.69; SPURS 0.64; Stability Oracle 0.618 | a zero-parameter model is within 0.06 of the 2026 SOTA |
| **Ssym antisymmetry r(dir,inv) / ⟨δ⟩** | FoldX −0.38 / −0.47; cartesian_ddg \|0.41\| / −3.12 | **DDGun3D −0.99 / −0.02**; KORPM 0.88 / −0.15; PoPMuSiC-sym −0.77 / +0.03 | ACDC-NN −0.98 / −0.02; JanusDDG exactly −1 / 0 | physics methods are the *worst* offenders; this is our free win |
| **FireProt-HF ρ** | FoldX 0.57 (lit.) / 0.449–0.481 on FireProt-test [own] | — | ThermoMPNN **0.657**; SPURS 0.67; PROSTATA 0.55 | |
| **Domainome ρ (per-domain median)** | FoldX 0.391 [own] | — | ESM-MSR 0.573; SPURS 0.54; ThermoMPNN 0.50 (0.57 excluding zinc fingers) | all predictors fail on zinc fingers (metal) |
| **ProteinGym stability ρ** | — | — | Kermut 0.817 (supervised per-assay CV); ProSST 0.653; ESM3 0.641; ESM-IF1 0.624 | 64/66 assays are Tsuboyama; 53 are ThermoMPNN-train → not independent |

### 2.2 Methods worth knowing in detail

* **ThermoMPNN** (Dieckhaus et al., *PNAS* 121:e2314853121, 2024) — ProteinMPNN embeddings + a light transfer head.
  The reference point on Megascale (0.725 pooled). Its *split* is the field standard. Also publishes ThermoMPNN-D for doubles.
* **SPURS** (2025, *Nat Commun* 2026, doi:10.1038/s41467-025-67609-4) — rewires ESM into ProteinMPNN; currently the
  best Megascale number (ρ 0.74 pooled, per-protein median 0.83). **Per-protein median 0.83 is the real ceiling to quote**
  — note it is above our site-only oracle (0.718), which means SPURS is exploiting within-site substitution structure.
* **RaSP** (Blaabjerg et al., *eLife* 12:e82593, 2023) — a 3D CNN **distilled from Rosetta cartesian_ddg**, 0.4 s/residue
  on a V100, 480–1036× faster than Rosetta at r ≈ 0.86 to its teacher. Proof that a cheap surrogate of a physics model
  is competitive; also the cleanest published xtal-vs-AF comparison (below).
* **Stability Oracle** (Diaz et al., *Nat Commun* 15:6170, 2024) — graph transformer on a "mutation-free" local
  environment + amino-acid embeddings; introduced **thermodynamic permutations** (turn n measurements at a site into
  n(n−1) ΔΔG pairs) — a data-augmentation idea we can reuse for fitting, and the reason its Ssym numbers are symmetric-ish.
* **Pythia** (Wu lab, bioRxiv 2023.08.09.552725) — 1.3 M-parameter self-supervised GNN, zero-shot, **25.2 M mutations
  in 3 min on one RTX 4090**. This is the speed/accuracy frontier PhysDDG is attacking: ρ 0.60 on Megascale, ρ 0.66 on S669.
* **JanusDDG** (*Commun Biol*, 2026, doi:10.1038/s42003-026-09632-9) — sequence-only, **antisymmetric by construction**,
  S461 Pearson 0.69. Demonstrates how much antisymmetry alone buys on the classical sets.
* **ProteinMPNN-ddG** (Dutton et al.) — adds an **unfolded-state reference** to inverse-folding log-odds; 9,800 res/s.
  Same conceptual fix as Rossi's MBC, from the DL side.
* **mutateeverything** (Ouyang-Zhang et al., NeurIPS 2023) — all singles + doubles of a 317-aa protein in 0.6 s.
* **METL**, **HERMES**, **ABYSSAL**, **UniStab**, **ESM-MSR**, **DDMut**, **PROSTATA**, **ACDC-NN**, **ThermoNet** — see the note.

### 2.3 What this means for a physics model

1. **The DL/physics gap is benchmark-dependent.** On Megascale it is large (0.57 → 0.74). On S461 it is small
   (0.635 untrained DDGun3D → 0.69). On antisymmetry physics *loses to a 4-term heuristic*.
2. **Rosetta cartesian_ddg buys nothing over FoldX on Megascale** (0.56 vs 0.57) for ~2,000× the compute.
   Expensive sampling is not the answer by itself.
3. **The two cheapest published fixes are both reference-state fixes** (ProteinMPNN-ddG, ddMBC) and both work.
4. **Everything above ρ 0.65 on Megascale is a learned model**; no published physics or statistical potential
   reaches it. That is the gap PhysDDG exists to close, and the internal PhysDDG v1.2 (0.640) is the current best attempt.

### 2.4 Four published results that should change our plan

1. **An antisymmetry *penalty* does not remove directional bias.** Shivram et al. (arXiv 2606.08100, 2026) added a
   Siamese (f→ + f←)² regulariser to SPURS: S461 ρ 0.653 → 0.711, but the forward over-estimate *grew*
   (+0.335 → +0.396 kcal/mol), because the penalty is minimised by shrinking both magnitudes. **Antisymmetry must be
   structural (a difference of two energies), not a loss term** — which is exactly what a physics model can do.
   They also show the in-distribution/OOD trade-off: Megascale ρ fell 0.749 → 0.713 while S669/S461 rose.
2. **Every method has *negative* Spearman in the strongly stabilising regime** (ΔΔG < −2 kcal/mol), because
   Megascale's range is roughly [−2, +5] (Tan et al., *Chem Sci* 17:18220, 2026, UniStab). If PhysDDG is meant for
   protein engineering (finding stabilising mutations) this is the regime that matters and no one can do it.
   Report precision@k for stabilising mutations separately.
3. **Physics + DL ensembles beat both.** StaB-ddG (Deng et al., ICML 2025, PMLR 267:13129) matches FoldX on SKEMPI
   binding ΔΔG (0.49 vs 0.49) at >1000× the speed, and **FoldX + StaB-ddG together reach 0.56**. Physics and DL make
   different errors; a fast accurate physics model has value even if DL stays ahead alone.
4. **The Domainome DL numbers are not robust.** Reeves & Kalyaanamoorthy (bioRxiv 2026) found ThermoMPNN retrained on
   their Megascale subset "performed significantly worse than all others, including its own zero-shot baseline" —
   so the ThermoMPNN 0.49–0.50 Domainome figure depends on the training subset. Re-run baselines ourselves rather
   than quoting.

---

## 3. Datasets

Full detail, with every download URL tested and every sign convention verified, is in
`notes/lit_datasets.md`. Condensed here.

### 3.1 Sign conventions (get this wrong and everything downstream is wrong)

| source | released column | sign of released column | flip for PhysDDG? |
|---|---|---|---|
| Tsuboyama 2023 Megascale | `ddG_ML` | **positive = stabilising** | **yes** |
| Tsuboyama `dG_ML` / `deltaG*` | ΔG of unfolding | positive = stable | n/a (absolute) |
| FireProtDB 1.0/2.0 and the ThermoMPNN export `4_fireprotDB_bestpH.csv` | `ddG` / `DDG` | **positive = destabilising** | no |
| ThermoMutDB | `ddG` | positive = stabilising | **yes** |
| ProThermDB / ProTherm | per publication | mixed | **check per row** |
| protddg-bench (S2648, VB1432, Ssym, S461, Broom, PTMUL, p53, myoglobin) | `DDG` | **negative = destabilising** | **yes** |
| Stability Oracle T2837 / C2878 / cDNA117K | `ddG` | positive = destabilising | no |
| ProteinGym `DMS_score` (Tsuboyama assays) | passthrough `ddG_ML` | positive = stabilising | **yes** |
| Domainome `normalized_fitness` | abundance fitness, WT = 0 | negative = destabilising | proxy, flip |
| Domainome Suppl. Table 4 `mean_kcalmol_scaled` (Zenodo 14356805) | inferred ΔΔG, rescaled | positive = destabilising | no (but not true kcal/mol) |
| Lehner ddPCA / MoCHI sets (GRB2, PSD95, GB1, KRAS, PDZ, Src) | ΔΔG_f kcal/mol | positive = destabilising | no |
| FoldX `total_energy` | kcal/mol | positive = destabilising | no |

Verified independently here: FireProtDB's ΔΔG is destabilising-positive (FoldX ρ = +0.481 on the
135 variants that merge cleanly) and its `pdb_position` is **0-based** **[own]**.

### 3.2 Tsuboyama 2023 "Megascale"

Tsuboyama K, Dauparas J, Chen J, …, Ovchinnikov S, Rocklin GJ, *Nature* 620:434–444 (2023),
DOI 10.1038/s41586-023-06328-6. Zenodo 10.5281/zenodo.7992926 (`Processed_K50_dG_datasets.zip`,
`AlphaFold_model_PDBs.zip` — the latter is already unpacked locally in `/workspace/data/megascale_pdbs`, 862 PDBs).

* cDNA display proteolysis: ΔG inferred from K50 (protease concentration at half-maximal cleavage) with
  trypsin and chymotrypsin, against a **sequence-based unfolded-state K50 model**. ΔΔG is therefore
  *inferred through a kinetic model*, not measured calorimetrically.
* **Dataset 1** = everything, 1,841,285 sequences. **Dataset 2** = quality-filtered for ΔG, 776,298
  (singles 566,180 over 478 WTs). **Dataset 3** = subset usable for ΔΔG (WT ΔG < 4.75 kcal/mol so that
  stabilising mutants stay in range), 607,839 (singles 448,788 over 412 WTs). Datasets 2 and 3 are in the
  same CSV; rows where the `_ML` columns are `"-"` failed filtering.
* **Conditions: pH 7.4, 25 °C, identical for every domain** — the single biggest advantage over ProTherm-derived sets.
* **Dynamic range −1 to 5 kcal/mol in ΔG**, hard-clipped; usable ΔΔG span ≈ −3 … +5 kcal/mol.
  Median 95% CI on ΔG inside the range is **0.14 kcal/mol**; trypsin-vs-chymotrypsin agreement
  Pearson **0.94 ± 0.04** over 478 domains; K50 replicate R = 0.97/0.99. Agreement with purified-protein
  literature ΔG is Pearson 0.75–0.96 per domain with systematic intercept offsets.
* Composition: 331 natural + 148 de novo designed domains (40–72 aa), plus 121 deliberately destabilised
  WT backgrounds. Structures are **AlphaFold / trRosetta models**, never crystals. Only small,
  monomeric, cooperatively folding, disulfide-free domains — no ligands, no oligomers, no large proteins.
* Doubles: 210,118 at 559 site pairs (Dataset 2) — the only large double-mutant resource.

### 3.3 ThermoMPNN splits (the de-facto standard)

Dieckhaus H, Brocidiacono M, Randolph NZ, Kuhlman B, *PNAS* 121(6):e2314853121 (2024),
DOI 10.1073/pnas.2314853121. `https://github.com/Kuhlman-Lab/ThermoMPNN/dataset_splits/mega_splits.pkl`
(confirmed to download; 16 kB). Curation: Dataset 3 → drop indels and multi-mutants → drop modified WT
backgrounds → **272,712 single substitutions over 298 proteins**; MMseqs2 at 25% identity → **239 train /
31 val / 28 test** proteins (215,731 / 27,328 / 28,172 mutations). Also ships `megascale_clusters.tsv`
(162 clusters), `fireprot_clusters.tsv`, `megascale_vs_fireprot_homologues.m8`, `s669_vs_megascale_homologues.m8`.

**The split is not homology-clean.** Independent re-check: 1TUC and 2KXD (test) are **98% identical
over 42–43 residues** to 2CDT (train); 2K1B/2K28 ≈ 40–44% to 2N88; 6EWU 46% to 6YSE; plus the
project's earlier finding of 9/28 test proteins with train homologs. Report a strict subset.

### 3.4 FireProtDB

v1.0: Stourac J et al., *NAR* 49(D1):D319 (2021), DOI 10.1093/nar/gkaa981 — 13,274 entries / 237 proteins.
**v2.0: Musil M et al., *NAR* 54(D1):D409 (2026), DOI 10.1093/nar/gkaf1211** — now an *aggregator*
(5.47 M experiments, 2,762 proteins) that **ingests Megascale and Domainome**; it is therefore no longer
an independent test set unless filtered on `SOURCE_DATASET`. REST API documented and working
(`/fireprotdb/api/search?query=...&format=csv`), exposing `DOMAINOME_DDG`, `MEGASCALE`, `SOURCE_DATASET`.
The benchmark everyone calls "FireProt" is ThermoMPNN's curated export (Zenodo 10.5281/zenodo.8169289):
**3,438 ΔΔG rows / 100 proteins**, splits 57/15/28 proteins (2,686/402/350 rows) plus
**FireProt-HF = 89 proteins / 2,578 rows with no Megascale homolog** — the right set for a Megascale-fitted model.
Biases: 24.4% mutations to alanine; two proteins supply ~1/3 of rows; heterogeneous pH/T; ΔΔG span −8.6…+12.

### 3.5 Classic benchmarks

One-stop, cluster-annotated download: **ProtDDG-Bench**, https://github.com/protddg-bench/protddg-bench
(`#SET,CLID,PDB,MUT,DDG`, **DDG negative = destabilising**, with ready-made homology-respecting folds and
matched training files that already exclude each test set's homologs).

**The classic sets circulate with both signs.** protddg-bench / VariBench / KORPM / GeoStab / DDGun distributions
are **negative = destabilising**; PremPS, ThermoNet (Q1744 etc.) and HotMuSiC-S1626 distributions are
**positive = destabilising**; the p53 set appears with both signs in different repositories. Check every file
numerically (fraction negative, a known cavity mutant such as T4 lysozyme L99A) before merging.

| set | n | chains | notes |
|---|---|---|---|
| S2648 | 2,648 | 132 | the classic training corpus; 77% destabilising |
| VariBench / VB1432 | 1,432 | 79 | overlaps S2648 (719 variants) |
| **Ssym** | 342 direct + 342 reverse | 15 | **57% of its direct variants and 13/15 proteins are inside S2648** — not independent of anything trained on S2648 |
| **S669** | 669 | 94 | Pancotti et al., *Brief Bioinform* 23:bbab555 (2022), DOI 10.1093/bib/bbab555; Zenodo 10.5281/zenodo.7568094. Built at <25% identity to S2648 ∪ VariBench. 20% of its values needed sign/value correction vs ThermoMutDB — a direct measure of literature-database error rate. **249/669 variants (37%) are homologous to Megascale domains**; use the 420-variant clean subset |
| S461 | 461 | 48 | ⊂ S669 (KORPM subset). Never report both as independent |
| Broom | 605 | 58 | |
| PTMUL | 914 | 90 | multi-point |
| p53 / myoglobin | 42 / 134 | 1 / 1 | single-protein sets |
| **T2837 / C2878 / cDNA117K** | 2,837 / 2,878 / 116,641 | 129 / 151 / 116 | Stability Oracle (Diaz et al., *Nat Commun* 15:6170 (2024), DOI 10.1038/s41467-024-49780-2). **T2837 = S669 ∪ Ssym ∪ p53 ∪ myoglobin deduplicated**, so T2837 results are not new evidence; cDNA117K ⊂ Megascale Dataset 1 |

### 3.6 Human Domainome 1

Beltran A, Jiang X, Shen Y, Lehner B, *Nature* 637:885–894 (2025), DOI 10.1038/s41586-024-08370-4.
Zenodo 10.5281/zenodo.13629491 (v3). **563,534 variants / 522 domains** (503 human, 127 Pfam families)
measured by abundance PCA in yeast. `normalized_fitness`: WT = 0, 2.5th percentile of missense = −1,
**more negative = more destabilising**. Fitness↔ΔΔG is sigmoidal and saturates for ΔΔG ≳ 3 kcal/mol.
Published quality figures: **median replicate Pearson r = 0.85**; median Spearman vs in-vitro ΔΔG **0.73**
(n = 10 domains) and vs Tsuboyama proteolysis **0.65** (n = 13). (Our project note quotes a 0.84 ceiling — that is
a recomputation; the paper's number is 0.85.) **Inferred ΔΔG *is* released**: Supplementary Table 4 (`mean_kcalmol_scaled`, positive = destabilising but
*rescaled*, not true kcal/mol), Zenodo concept 11043642 → record **14356805**. The paper fits MoCHI Boltzmann
models per domain family (r = 0.78 CV on homeodomains) and additionally releases *predictions* for 4.1 M variants
in 7,271 domains (`Extended_data_Table_4_homolog_predictions.txt`, 319 MB). FireProtDB 2.0 re-exposes
Domainome-derived ΔΔG as `DOMAINOME_DDG`. MaveDB holds only the fitness scores.

### 3.7 ProteinGym stability subset, MaveDB, VAMP-seq

* **ProteinGym** (Notin et al., NeurIPS 2023 D&B; https://proteingym.org): `coarse_selection_type == Stability`
  = **66 of 217 assays / 134,794 variants**, of which **64 are Tsuboyama domains**.
  `DMS_substitutions.csv` reference table downloads directly from the repo. `DMS_score` for those assays is
  raw `ddG_ML` (positive = stabilising). **53 of the 64 are ThermoMPNN *training* domains**, so
  "ProteinGym stability" is essentially Megascale-train — nearly useless as an independent benchmark here.
* **VAMP-seq**-style abundance DMS (PTEN/TPMT Matreyek 2018; NUDT15 Suiter 2020; CYP2C9 Amorosi 2021; …,
  on MaveDB) are abundance proxies normalised WT ≈ 1 / nonsense ≈ 0, saturating at both ends and confounded
  by degradation-pathway usage. Use for AUROC-style validation, not regression.
* **MGnify Stability (Rocklin lab 2026)** — Cho Y, Tsuboyama K, …, Rocklin GJ, bioRxiv
  doi:10.64898/2026.05.19.726285, Zenodo 10.5281/zenodo.19411306 (~8 GB). cDNA-display proteolysis on
  **1.8 M domains of 60–80 aa from >200,000 metagenomic families**, recalibrated to a **6 kcal/mol** usable
  ΔG window, with the authors' own train/val/test split in the file and a new curated ThermoMutDB ΔΔG
  benchmark (`S1724`, 35 WT proteins) in `benchmarks.zip`. **This is an absolute-ΔG resource**: it is the
  natural place to test whether PhysDDG's *folded-state* energy is on a physical scale, and the ins/del
  benchmarks are a stress test no ΔΔG-only model has ever been held to.
* Meltome/Tm datasets are not ΔΔG; keep them out of fitting.

### 3.8 Lehner-lab ddPCA / MoCHI datasets — the only other ΔΔG-in-kcal/mol resources at scale

These release **folding ΔΔG directly in kcal/mol with per-variant errors**, on larger and different domains than
Megascale, and most validate against in-vitro ΔΔG at r = 0.79–0.95. Sign in the released tables:
**positive = destabilising** (80–89% of single mutants are > 0; Pro substitutions have the largest positive means);
"confident" = Monte-Carlo 95% CI < 1 kcal/mol. Tool: MoCHI (Faure & Lehner, *Genome Biol* 25:303, 2024).

| dataset | citation | size (single-mutant folding ΔΔG) | source |
|---|---|---|---|
| GRB2-SH3, PSD95-PDZ3, GB1 (ddPCA) | Faure et al., *Nature* 604:175 (2022), DOI 10.1038/s41586-022-04586-4 | 1,056 / 1,587 / 1,045 (898–1,148 confident) | Suppl. Table 7 xlsx; github.com/lehner-lab/doubledeepms. PSD95-PDZ3 vs in vitro r = 0.79 (n=30) |
| **KRAS 4B (1–188)** | Weng, Faure, Escobedo & Lehner, *Nature* 626:643 (2024), DOI 10.1038/s41586-023-06954-0 | 3,453 over 187 positions (2,339 confident); **in vitro r = 0.95** | MaveDB `00000115` (folding = `-a-7`), Suppl. Table 5 |
| PDZ family (5 domains) | Martí-Aranda & Lehner, *Nat Commun* 17:5487 (2026) | 9,064 ΔΔG_f + 12,738 ΔΔG_b | Zenodo 15978782 |
| PDZ3 extensions | Hidalgo-Carcedo et al., *Nat Commun* 17:2934 (2026) | >190,000 variants, in vitro r = 0.88 | Zenodo 16411462 |
| Src kinase | Beltran et al., *Sci Adv* 12:eaea2726 (2026) | >50,000 single+double | Zenodo 17165108 |
| KINASE-MAPS (SRC, FGR, JNK2, ZAK, TSSK2) | bioRxiv 10.64898/2026.08.04.742685 | >160,000 variants | Zenodo 23261855 |

**These are the best available external ΔΔG tests for PhysDDG**: real kcal/mol, independent of Megascale's assay
and size range (SH3, PDZ, KRAS, kinases rather than 40–72 aa mini-domains), with binding ΔΔG alongside folding ΔΔG
so stability and function can be separated. Use them as held-out sets, never for fitting.

**MaveDB warning**: the Tsuboyama data is *also* in MaveDB (~1,082 "trypsin digestion" score sets, mostly with no
linked publication). Harvesting MaveDB broadly will silently double-count Megascale.
(MaveDB current paper: Rubin et al., *Genome Biol* 26:13, 2025, DOI 10.1186/s13059-025-03476-y.)

---

## 4. AlphaFold structures for ΔΔG

Full detail and citations in `notes/lit_alphafold.md`. The decision-relevant content:

### 4.1 What the evidence says

1. **High-confidence AF2 models are roughly as good as crystal structures** for FoldX, Rosetta, RaSP, mCSM-type and
   ThermoMPNN-type predictors (Akdel et al., *Nat Struct Mol Biol* 29:1056 (2022); Blaabjerg et al., *eLife* 12:e82593 (2023);
   Sora 2023; Diaz 2024; Gerasimavicius 2023). RaSP: mean Pearson between predictions on two crystal structures 0.97
   vs crystal-vs-AF **0.94**; but only **0.66–0.96 for sites with pLDDT < 0.9**.
2. **The degradation is concentrated below pLDDT 70, and partly a dynamic-range artefact** (low-pLDDT sites genuinely
   have smaller |ΔΔG|).
3. **FoldX is far more fragile to input-structure error than Rosetta cartesian_ddg** (Valanciute et al. 2023): with
   homology models below ~45% template identity, FoldX and ddg_monomer collapse to the structure-free null while
   cartesian_ddg stays near crystal performance. **The lesson for PhysDDG: local Cartesian relaxation of WT *and*
   mutant is what makes a method robust to model error.** A rigid, occupancy-scaled FoldX-style energy is the worst case.
4. **Do not repack AF side chains globally.** AF2/AF3 native side chains beat SCWRL4, FASPR, Rosetta packer and the DL
   packers when re-packed on an AF backbone (Vangaru & Bhattacharya, *Brief Bioinform* 26:bbaf297, 2025). Repack only
   the mutated site and its 6–8 Å shell, identically for WT and mutant.
5. **AF accuracy by band** (Terwilliger et al., *Nat Methods* 21:110 (2024); Tunyasuvunakool 2021): median Cα error
   0.6 Å (pLDDT > 90), 1.1 Å (80–90), 1.5 Å (70–80); χ1 within 40° **80%** of the time at pLDDT > 90; **20% of side
   chains in pLDDT ≥ 70 regions differ by ≥ 1.5 Å RMSD** from the deposited structure (vs 6% between two crystal forms).
   ⇒ **backbone-derived burial is trustworthy well below the pLDDT where side-chain contacts are not.**
6. **Mutant AF models are useless**: Buel & Walters 2022; Pak et al. 2023 (ΔpLDDT vs ΔΔG r = −0.17); Stability Oracle
   loses 2–4% using AF mutant structures. Never run AF on the mutant.
7. **Missing context is the dominant systematic failure**: metals (all predictors fail on Domainome zinc fingers),
   cofactors, oligomeric partners (Gerasimavicius 2022: AUC 0.652 → 0.677 with full assemblies), membranes.
8. **The Megascale structures are themselves AF2 models**, with termini trimmed by contact count, and designed domains
   folded in single-sequence mode. So anything fitted on Megascale is already fitted on AF geometry — and our FoldX
   0.569 baseline is *already* a "FoldX on AF structures" number. The AF mode matters for *transfer* to large natural
   multi-domain proteins (Domainome, proteome-scale), not for Megascale itself.
9. **The one published head-to-head on AF inputs** is Degn et al. 2025 (bioRxiv 10.1101/2025.03.28.645695), set **S612**
   (= the 612 S669 variants with an AFDB model at pLDDT > 70): MutateX/FoldX PCC 0.44, Rosetta cartesian 0.44,
   RaSP 0.42, DDGun3D 0.38, **ThermoMPNN 0.51**; on the 332 variants <25% identical to ThermoMPNN's training data all
   drop (FoldX 0.33, ThermoMPNN 0.36). Also: an **MD ensemble as input beats a single static AF model and FoldX gains
   most (+0.07 PCC)**, while coarse-grained (CABS-flex) ensembles *degrade* results.
10. **Field conventions** (never benchmarked against experiment, but universal): report only pLDDT > 70; trim termini /
    linkers below pLDDT 70 (MAVISp) or 10-residue windows below 50 (Jänes et al. 2024, the Beltrao-lab ProtVar pipeline
    that produced the FoldX resources on this machine); RepairPDB / minimise side chains before scanning.
11. **No published study quantifies "over-idealised / over-packed AF cores" for ΔΔG, and no AF3-vs-AF2 ΔΔG benchmark
    exists** (as of 2026-10). Both are open questions we can answer ourselves with paired xtal/AF sets.

### 4.2 Recommended AF mode (condensed from `notes/lit_alphafold.md` §6)

* Prefer an experimental structure when one covers the site; otherwise AF2/AF3, recording `struct_source`, AFDB version and pLDDT.
* Trim terminal/linker segments with pLDDT < 70 and internal 10-residue windows with mean pLDDT < 50 — **unless the
  segment is genuinely packed** (contact density test, Williams et al. 2025 "barbed wire" vs "near-predictive" criterion).
* Crop multi-domain models at PAE domain boundaries; for Domainome use the assayed construct crop.
* Keep AF side chains; repack only site + shell; apply a **restrained local Cartesian minimisation with restraint
  strength scaled by pLDDT**, identically to WT and mutant.
* Use **Cβ-count / half-sphere-exposure burial rather than all-atom SASA** in the 70–90 pLDDT band.
* Shrink the prediction toward neutral as site pLDDT falls: `ΔΔG_AF = w(pLDDT)·ΔΔG_phys + (1−w)·c`, w logistic,
  2 extra fitted parameters. Report metrics stratified by pLDDT band.
* Average over the 5 AF2 models / AF3 seeds where available — cheap on GPU, and yields a per-mutation uncertainty.
* Flag metal sites (Cys/His clusters, zinc-finger Pfams), cofactor sites and interfaces; either add the ligand
  (AlphaFill / AF3) or report those variants separately.

---

## 5. Speed, packing algorithms and GPU

Full tables with licences, URLs and verification marks: `notes/lit_speed_algorithms.md`.

### 5.1 Side-chain packers

| packer | algorithm | accuracy | speed |
|---|---|---|---|
| **SCWRL4** (Krivov, Shapovalov, Dunbrack, *Proteins* 77:778, 2009) | bbdep KDE library, anisotropic H-bond, soft vdW, **tree decomposition** | 86% χ1, 75% χ1+2 within 40° | ~12 s per CASP13 target |
| **FASPR** (Huang, Pearce, Zhang, *Bioinformatics* 36:3758, 2020) | **DEE + tree decomposition**, Dunbrack 2010 | 69.1% of all χ within 20° (SCWRL4 68.8) | **379 proteins in 34.3 s** (~0.09 s each); fastest classical packer; MIT C++ |
| **Rosetta packer** | simulated annealing over precomputed one-/two-body tables | best on core residues, fewest clashes | ~124 s per CASP13 target (~150× FASPR) |
| OPUS-Rota2, RASP, CIS-RR, OSCAR-star/o, Sccomp, TreePack | various | within ~1 pt of each other | OSCAR-o ~100× slower than OPUS-Rota2 |
| DL packers (AttnPacker, DLPacker, PIPPack, DiffPack, FlowPacker) | — | 0.1–0.25 Å better RMSD | excluded (DL) — and **worse than AF's own side chains on AF backbones** |

**Conclusion: accuracy differences between SCWRL4/FASPR/Rosetta are ≤ 0.04 Å RMSD and ≤ 1.5 points of rotamer
recovery — noise compared with ΔΔG error.** Choose the packer by batchability, not accuracy. DEE + exact small-graph
search (FASPR style) or **batched mean-field / belief propagation over rotamer marginals** (naturally tensorised,
and BP additionally yields side-chain conformational *entropy*, which we need as an energy term anyway:
Kamisetty, Xing & Langmead, *J Comput Biol* 15:755, 2008).

### 5.2 Rotamer libraries (and the licence trap)

* **Dunbrack 2010 smoothed backbone-dependent** (Shapovalov & Dunbrack, *Structure* 19:844, 2011) is the standard.
  It is **CC BY 4.0 since 2019**, but the official download at dunbrack.fccc.edu requires a free application and
  approval — **a possible access delay; flag it**. Copies are already redistributed inside **FASPR**
  (`dun2010bbdep.bin`, MIT repo) and **tmol** (`dunbrack.bin`, Apache-2.0 repo), usable under CC BY attribution.
* **Richardson Top8000 rotamer distributions** (Hintze et al., *Proteins* 84:1177, 2016) are on GitHub
  (rlabduke/reference_data) under CC BY 4.0, backbone-independent.
* **Top2018** (Williams, Richardson & Richardson, *Protein Sci* 31:290, 2022) — ~3 M residue-filtered residues on
  Zenodo under CC — enough to **derive our own backbone-dependent library** (von Mises KDE of P(rot|φ,ψ) on a 10°
  grid, circular statistics for mean/σ per bin, continuous treatment of terminal sp2 χ). Budget ~1 day.
* Use `−log P(rot|φ,ψ)` as an explicit energy term (Rosetta's `fa_dun`, weight 0.7).

### 5.3 Energy-function parameter sources we can lift directly

* **Rosetta REF2015** (Alford et al., *JCTC* 13:3031, 2017): 19 weighted terms — fa_atr 1.0, **fa_rep 0.55
  (linearised below 0.6σ)**, fa_sol 1.0 (Lazaridis–Karplus), lk_ball_wtd 1.0 (anisotropic polar desolvation),
  fa_elec 1.0 (sigmoidal ε 6 → 80, truncated at 5.5 Å), four hbond terms at 1.0, fa_dun 0.7, rama_prepro 0.45,
  p_aa_pp 0.6, omega 0.4, pro_close 1.25, dslf_fa13 1.25, yhh_planarity 0.625, **ref 1.0 (20 per-aa reference energies)**.
  The LK atom parameters (`atom_properties.txt`: LJ_RADIUS, LJ_WDEPTH, LK_DGFREE, LK_LAMBDA = 3.5 Å, LK_VOLUME) are a
  ready-made solvation table. REU → kcal/mol ≈ ÷2.9.
* **EEF1** (Lazaridis & Karplus, *Proteins* 35:133, 1999): Gaussian solvent-exclusion, group parameters from model
  compounds, ionic side chains neutralised + distance-dependent dielectric. Pairwise with a ~9 Å cutoff → cheap on GPU.
* **GB/SA** (OBC, GBn2) is available in OpenMM but needs Born radii (an extra pairwise pass) and is noisier for ΔΔG
  than LK unless ensemble-averaged.
* **H-bonds**: Kortemme, Morozov & Baker, *JMB* 326:1239 (2003) orientation-dependent statistical potential — the
  standard, and better than a Coulomb-only H-bond.
* **SASA**: if ever needed, LCPO (Weiser, Shenkin & Still, *J Comput Chem* 20:217, 1999) is the analytic fast version;
  otherwise prefer FoldX-style occupancy or Cβ neighbour counts.

### 5.4 GPU tooling

* **tmol** (https://github.com/uw-ipd/tmol, Apache-2.0, actively developed as of 2026-10) — *"Rosetta on the GPU"*:
  all-atom scoring with the beta2016/REF2015-family energy, a **Dunbrack-based rotamer packer**, minimisation and
  FastRelax, implemented in **PyTorch + custom CUDA kernels**, with batched padded `PoseStack`s, gradients through
  coordinates and CUDA-graph replay. **This is the closest prior art to the PhysDDG engine and contains no neural
  network.** Decide early whether to use it as a dependency, as a reference implementation, or as a baseline — but
  do not reimplement its kernels blindly. (No peer-reviewed paper found.)
* **OpenMM 8** (Eastman et al., *J Phys Chem B*, 2024) — the fastest route to a restrained Amber relax of AF models
  (AF2's own relaxation step uses OpenMM + Amber99sb). Already installed in this project's venv.
* **TorchMD** (Doerr et al., *JCTC* 17:2355, 2021) is a cautionary tale: **~60× slower than a native MD engine,
  attributed to the lack of neighbour lists**. Dense pair loops in PyTorch are slow; use fixed-size local
  environments or cell lists.
* **JAX-MD**, **DMFF** — differentiable energies with cell lists; **OSPREY 3.0** (GPU gives >10× on provable design);
  **Force Field X** (many-body rotamer optimisation with OpenMM GPU energies; took AF2 models from clashscore 20.75 → 0.11).
* **No GPU FoldX-like tool exists.** MutateX only parallelises FoldX by process.

### 5.5 Batching recipe for a single-site ΔΔG engine

1. **Fixed-size local environments, not global neighbour lists**: gather all residues with any atom within 10–12 Å of
   the site, pad to K ≈ 32–64, and evaluate everything as dense `[n_mut, n_rot, K, atoms, 3]` tensors.
2. **Precompute one-body and two-body rotamer energy tables** for the shell once per site; the 19 substitutions share
   the environment and only the mutated position's rotamer set changes. Packing then becomes a small discrete
   optimisation (DEE + exhaustive/tree search, or batched mean-field/BP on GPU).
3. Smooth, truncated terms (5.5–6 Å electrostatics, LK ~6 Å, occupancy 6 Å) keep K small.
4. Occupancy / Cβ-neighbour-count burial instead of SASA (one radius count per atom).
5. Autograd + L-BFGS for shell minimisation; CUDA graphs for repeated fixed shapes; float32 with
   **linearised repulsion** to avoid r⁻¹² overflow; group proteins by size; one process per GPU.

### 5.6 Runtime reference points

| tool | runtime per mutation | hardware |
|---|---|---|
| **FoldX BuildModel** | **≈ 34 CPU-s** (Dunham, Beltrao & AlQuraishi, *Genome Biol* 2023: 24,187 Spike variants in 232 h; 161,825 variants in 64.5 CPU-days); **5–10 CPU-s [own]** on our 64-core runs; 395–807 s per *saturated residue* on 64 cores (RaSP Table 4) | CPU only |
| **Rosetta cartesian_ddg** | ≈ 20–25 CPU-min (3 iterations) + 677–17,211 s one-off relax | CPU only |
| flex-ddG | CPU-hours (35 backrub models × 35,000 trials) | CPU |
| PoPMuSiC 2.1 | all single mutants of a medium protein in < 1 min (~10 ms/mutation) | CPU |
| **KORPM** | **~5 ms** (684 mutations in 3.5 s, one core) | 1 CPU core |
| RaSP | 0.4 s per *position* (all 19), length-independent | V100 |
| ThermoMPNN | full SSM 2 s (105 aa) – 8 s (693 aa) | V100 |
| Stability Oracle | ~50 ms per residue | GPU |
| Pythia | 25.2 M mutations in 3 min | 1 × RTX 4090 |

**Target**: FoldX-class-or-better accuracy at **≲ 1–10 ms per mutation on a GPU** (≈ 3,000–30,000× FoldX CPU time),
which puts a full human-proteome saturation scan (~2 × 10⁸ variants) within ~1 GPU-week on the 4 GPUs here.

---

## 6. Evaluation pitfalls (read before reporting any number)

1. **Pooled vs per-protein correlation.** On Megascale-test, FoldX scores pooled ρ = 0.569 but per-protein
   median ρ = 0.633, range 0.461–0.754 **[own]**. Pooled correlations are inflated/deflated by between-protein
   differences in ΔΔG spread (designed mini-proteins have a different dynamic range from natural domains).
   Report **both**, and bootstrap over proteins, not over variants.
   Beware a subtle trap: when a model is fitted in CV folds, per-fold intercepts/scales differ, so the
   *pooled* Spearman of a cross-validated prediction is not comparable to the pooled Spearman of a
   single global model (we measured 0.545 vs 0.569 for the same FoldX scores **[own]**). Per-protein medians
   are immune to this.
2. **Homology leakage.** 9 of the 28 Megascale-test proteins have homologs in Megascale-train (2 at 100% id);
   strict test = 19 proteins **[own, project record]**. FireProtDB overlaps the training sets of nearly every
   published predictor (15/25 proteins and 94/341 mutations in one internal check **[own]**). ProteinMPNN/ESM-IF
   were trained on the PDB, which contains S669/Ssym proteins. **Always report a strict non-homologous subset.**
3. **Leakage can produce absurd numbers and nobody notices.** Diaz et al. 2024 audited PROSTATA: retrained on its
   own 75%-identity-filtered sets, **93/93 S669 proteins and 15/15 Ssym proteins were still leaked**, giving a
   Pearson of **0.98** on S669. Trained on the clean Q1744 the same model gives 0.49/0.51. Any S669/Ssym number
   above ~0.7 should be assumed to be leakage until proven otherwise.
4. **Antisymmetry and the destabilising bias.** S2648/Q3421 contain ~3.2–3.7× more destabilising than
   stabilising mutations (Pucci et al., *Bioinformatics* 34:3659 (2018)). Models trained on them predict
   destabilisation by default, and reverse-mutation performance collapses
   (FoldX: direct 0.63 → inverse 0.39; MAESTRO 0.52 → 0.32; SDM 0.51 → 0.32; DDGun 0.48 → 0.48).
   Metrics to report on Ssym: r_dir, r_inv, **r(dir,inv)** (should be −1) and **⟨δ⟩ = mean(ΔΔG_dir + ΔΔG_inv)** (should be 0).
   A physics model built on a folded-state difference plus an antisymmetric reference term is antisymmetric
   *by construction* — this is a free win and should be enforced, not fitted.
5. **Assay dynamic range / saturation.** Tsuboyama ΔG estimates are bounded by the proteolysis assay
   (ΔG roughly −1 to 5 kcal/mol; estimates near the edges, and any with 95% CI > 0.5 kcal/mol, are flagged
   low-confidence in the source data). Consequences: ΔΔG for very destabilising mutations is **censored**,
   not merely noisy; our Megascale-test ΔΔG spans −2.92 to +5.72 with only 28.9% of variants above
   +1 kcal/mol **[own]**. Over the whole runtime table (271,231 single substitutions, destabilising-positive):
   mean +0.73, **sd 1.00**, median +0.48, min −3.10, max +5.72 **[own]**. An sd of 1.0 kcal/mol means an
   RMSE of 1.0 is the trivial "predict the mean" baseline, and nothing below ~0.6 is credible. Fitting squared error on censored data biases the slope; consider a censored
   likelihood, or a robust (Huber) loss, or simply fitting ranks + a separate calibration.
6. **Noise ceiling.** Fariselli and coworkers estimate the best attainable Pearson on classical ΔΔG datasets
   at **0.7–0.8** given experimental noise and spread. For Domainome-style fitness assays the ceiling is
   ρ ≈ 0.84 (project estimate), and the best DL model reaches ~73% of it. Do not chase correlations past
   the ceiling; report how close you are to it.
7. **Duplicates and averaging.** ProTherm-derived sets contain the same mutation measured several times,
   at different pH/temperature, sometimes with opposite signs. Average per
   (protein, position, wt, mut) *after* harmonising sign, and record the spread. FireProtDB's
   "best pH" selection is one such harmonisation and is not neutral.
8. **pH and temperature.** Classical ΔΔG values come from denaturation curves at assorted pH (2–9) and T;
   Megascale is a single condition (pH 7.4, 25 °C **[verify in dataset note]**). A pH-blind model (like FoldX)
   has an irreducible error on the classical sets. Either filter to pH 6–8 or add pH as an input to the
   ionisation term.
9. **Trivial baselines must be reported.** **[own, `scripts/lit_trivial_baselines.py`]**, Megascale-test:

   | baseline | pooled ρ | per-protein median ρ |
   |---|---|---|
   | burial, 1 − RSA (Shrake–Rupley, Tien max-ASA) | 0.461 | 0.514 |
   | ΔKyte–Doolittle alone | 0.304 | 0.354 |
   | burial × ΔKyte–Doolittle | 0.303 | 0.357 |
   | burial × \|ΔVolume\| | 0.311 | 0.336 |
   | to-Pro indicator | 0.160 | 0.187 |
   | 7-feature linear model (grouped CV) | **0.536** | **0.600** |
   | FoldX 5.1 | 0.569 | 0.633 |

   A predictor that does not clearly beat 0.60 per-protein on Megascale has not demonstrated that its
   physics does anything.
10. **How much is "site" and how much is "substitution"?** **[own, `scripts/lit_site_vs_substitution.py`]**
   Oracle decomposition of Megascale-test (1,562 sites, 18.0 substitutions/site):

   | oracle | pooled ρ | per-protein median ρ |
   |---|---|---|
   | leave-one-out **per-site mean** ΔΔG (knows nothing about the substituted residue) | **0.757** | 0.718 |
   | global per-**substituted-amino-acid** mean only | 0.174 | 0.190 |
   | additive site + amino acid | **0.800** | **0.777** |

   Two consequences. (a) **A perfect "how sensitive is this site" model already beats every published
   predictor** (ThermoMPNN ≈ 0.72 pooled): the task is dominated by identifying buried, tightly packed,
   H-bonded positions, not by the chemistry of the incoming residue. (b) The purely **additive**
   (site + substitution) model class has a ceiling of ρ ≈ 0.80, which is above the noise-limited
   performance of any current method — so site×substitution interaction modelling (the thing that is
   expensive in a packing engine) is *not* the limiting factor. Spend the compute on getting burial,
   packing density and H-bond environment right per site, and on a clean additive substitution scale.
   (c) How far are we from the site oracle? **[own, `scripts/lit_site_sensitivity_proxies.py`]**, predicting
   the *site-mean* ΔΔG over 1,562 sites: −RSA pooled 0.606 / per-protein 0.675; weighted contact number
   (CB, sigmoid at 9 Å) 0.602 / 0.667; contact count < 10 Å 0.594 / 0.685; **FoldX site mean 0.651 / 0.722**.
   So FoldX adds only ~0.05 over a contact count at the site level, and there is ~0.35 of pooled correlation
   still unclaimed between FoldX's site estimate and the site oracle. **Per-site sensitivity is where the
   remaining accuracy lives.**

11. **Reverse mutations as training data.** Augmenting with antisymmetrised copies (as Rossi 2025 and
   PoPMuSiC-sym do) removes bias but also halves the effective information; for a model that is
   antisymmetric by construction it is redundant.
12. **Per-dataset metric conventions differ.** S669/S461 papers report **Pearson**; Megascale papers report
    **Spearman**; Ssym papers report antisymmetry. Reporting only your best metric on each set is the
    standard way published numbers get inflated — report Spearman, Pearson, RMSE and AUROC(ΔΔG > 1) everywhere.
13. **Structure provenance is part of the method.** Numbers computed on crystal structures are not comparable
    to numbers on AF2 models; Rossi 2025 found ESM-IF1 and Pythia do *better* on AF models than on
    experimental ones (because >90% of their training structures were AF models). See §4.
14. **A benchmark's difficulty is mostly its stabilising fraction.** Stabilising mutations are 12.7% of S669,
    5.0% of S461, 11.4% of Ssym-direct, **4.2% of Megascale** and <5% of Megascale doubles. Every method's apparent
    advantage shrinks on balanced subsamples (Degn et al. 2025 on S4038). A model that simply predicts
    "destabilising" is right 80–95% of the time — which is why ⟨δ⟩ and balanced-subset metrics are mandatory.
15. **Stability predictors transfer poorly to functional DMS.** Gerasimavicius, Livesey & Marsh,
    *Protein Sci* 32:e4688 (2023): 9 stability tools against 49 DMS datasets (170,940 variants) give a mean
    |Spearman| of only **0.26** (FoldX and Rosetta best; their mean, "Foldetta", beats both), versus 0.36 for
    variant-effect predictors. Høie et al. 2022 put Rosetta-vs-DMS at 0.25. Do not expect Domainome/VAMP-seq
    correlations in the ΔΔG range, and do not interpret a low DMS correlation as a stability failure.
16. **Random splits on DMS data measure nothing.** In ProteinGym the one-hot supervised baseline scores 0.582
    under a random split and **0.022 under a modulo split**.
17. **Designed vs natural domains.** Megascale is ~50% de-novo designed mini-proteins, which are idealised,
    hyper-stable and have unusually clean hydrophobic cores. A model tuned on them can transfer badly to
    natural domains; always break results down by `natural vs designed`.

---

## 7. Implications for PhysDDG design (prioritised)

Ordered by expected return on effort. Each item says what to do and what evidence it rests on.

### Targets to beat (set these in the harness now)

| benchmark | metric | FoldX 5.1 | best classical / statistical non-DL | best DL | PhysDDG v1.2 (internal) | **PhysDDG target** |
|---|---|---|---|---|---|---|
| Megascale test (ThermoMPNN split, 28 proteins, 28,172 variants) | pooled ρ | **0.569** [own] | Rosetta cartesian_ddg 0.56 | **SPURS 0.74**; ThermoMPNN 0.725; RaSP 0.67; Pythia (zero-shot) 0.602 | 0.640 | **≥ 0.68** |
| Megascale test | per-protein median ρ | 0.633 [own] | — | SPURS **0.83**; ThermoMPNN 0.77 | 0.701 | **≥ 0.73** |
| Megascale test | RMSE (kcal/mol) | ~0.96 after a global rescale [own]; 3.9 raw | — | ThermoMPNN 0.708 | — | **≤ 0.75** |
| S669 direct | Pearson (Spearman) | 0.22 (ρ 0.283) | MAESTRO 0.50; DDGun3D 0.43 | JanusDDG 0.55; Pythia ρ **0.66** | — | **≥ 0.50 Pearson** |
| S461 | Pearson | — | **DDGun3D 0.635**; PremPS 0.632; KORPM 0.569 | JanusDDG 0.69; SPURS 0.64 | — | **≥ 0.62** |
| Ssym | r(dir,inv) / ⟨δ⟩ | −0.38 / −0.47 | DDGun3D **−0.99 / −0.02**; KORPM 0.88 / −0.15 | ACDC-NN −0.98 / −0.02 | — | **−1.00 / 0.00 (by construction)** |
| Ssym | direct / inverse Pearson | 0.63 / 0.39 | KORPM 0.57 / 0.49 | Stability Oracle 0.72 / 0.72 | — | **≥ 0.60 / ≥ 0.55** |
| FireProt-test / FireProt-HF | ρ | 0.449–0.481 [own] / 0.57 (lit.) | — | ThermoMPNN 0.657; SPURS 0.67 | ~0.62 (clean subset) | **≥ 0.55** |
| Domainome (522 domains) | per-domain median ρ | 0.391 [own] | — | ESM-MSR 0.573; SPURS 0.54; ThermoMPNN 0.50 | 0.394–0.403 | **≥ 0.45** |
| S612 (S669 on AF2 models, pLDDT > 70) | Pearson | MutateX/FoldX 0.44 | DDGun3D 0.38 | ThermoMPNN 0.51 | — | **≥ 0.48** |
| speed | per mutation | 5–10 CPU-s [own]; 395–807 s per saturated residue on 64 cores (RaSP Table 4) | **KORPM ~5 ms/mutation, 1 core**; PoPMuSiC ~10 ms | Pythia 25.2 M mutations in 3 min on one RTX 4090 | — | **≤ 1 ms/mutation on one GPU** (and ≤ 50 ms on one CPU core) |

Interpretation: on the *classical* benchmarks we only need to reach DDGun3D/KORPM territory to be competitive, and
antisymmetry is a guaranteed win. On **Megascale the gap is real** — 0.57 (FoldX) → 0.64 (PhysDDG v1.2) → 0.72–0.74
(DL) — and that is where the project is decided. On **speed**, KORPM (12 parameters, 5 ms/mutation, one core) and
Pythia (10⁵ mutations/s on a 4090) define the frontier; "fast" alone is not a contribution any more.

### Recommendations

Ordered by expected return on effort. Each says what to do and the evidence behind it.

1. **Build the engine around per-site sensitivity, not per-substitution chemistry.** A site-only oracle scores
   0.757 pooled / 0.718 per-protein on Megascale-test — above every published model — and an additive
   site+substitution oracle reaches 0.800 / 0.777 [own §6.10]. The best available site proxies reach only
   0.59–0.65 against the site oracle (−RSA 0.606, contact number 0.594, FoldX site mean 0.651 [own]).
   **Most of the remaining accuracy is in "how sensitive is this position", and additivity is not the bottleneck.**
2. **Make ΔΔG a difference of two energies, with an explicit unfolded state:**
   `ΔΔG = [E_fold(mut) − E_fold(wt)] − [E_unf(mut) − E_unf(wt)]`, with `E_unf` a sum of per-residue terms depending
   on amino-acid identity and local sequence context. This is **antisymmetric by construction** (r(dir,inv) = −1,
   ⟨δ⟩ = 0), which no empirical-energy competitor achieves (FoldX −0.38/−0.47) and which **cannot be obtained by a
   training penalty** (Shivram 2026: the Siamese penalty made the bias worse). Target |⟨δ⟩| < 0.15, r(dir,inv) ≤ −0.90.
3. **Fit a 20-coefficient unfolded-state (mass-balance) term, initialised from the Rose scale.** Rossi et al. 2025
   report +0.18 Pearson on S461 for FoldX and +0.07 on Megascale for Pythia; our own replication on FoldX gives
   Pearson 0.417 → 0.633 and RMSE 0.96 → 0.82 on Megascale-test [own §1.7]. Costs nothing at runtime. Expect
   calibration gains first; ranking gains on composition-heterogeneous sets and on reverse mutations.
4. **Implement ~8–10 energy terms, in this order**: (i) burial-scaled apolar desolvation; (ii) packing/vdW with a
   **separately parameterised cavity (large→small) vs overpacking (small→large) asymmetry** (PoPMuSiC's Heaviside
   ΔV± split); (iii) burial-scaled side-chain conformational entropy, charged at full cost when the side chain is
   H-bonded or salt-bridged (FoldX rule); (iv) backbone torsional/Ramachandran propensity with explicit Gly and Pro
   handling; (v) a geometric H-bond term **paired** with polar desolvation (they cancel to a few tenths of a
   kcal/mol — never ship one without the other); (vi) screened Coulomb for salt bridges only, with a burial-dependent
   dielectric. **Do not implement** water bridges, disulfide, loop entropies, helix dipole, cis-bond or ionisation in
   v1 — they are identically zero or |ρ| ≤ 0.02 on our benchmark [own §1.1].
5. **Lift parameters rather than inventing them**: Rosetta REF2015 weights and the LK atom table
   (LJ_RADIUS/WDEPTH, LK_DGFREE, λ = 3.5 Å, LK_VOLUME) for solvation; Kortemme 2003 for orientation-dependent
   H-bonds; FoldX's occupancy model and transfer-derived atomic energies (−0.082 kcal/mol/Å³ vdW,
   −0.0314 kcal/mol/Å³ apolar solvation, +0.8 polar, +1.44 charged); Dunbrack `−log P(rot|φ,ψ)` as `fa_dun`.
6. **Use occupancy-style burial, not SASA, in the inner loop.** `Occ(i) = Σ_j V_j exp(−d²/2σ²)`, σ = 3.5 Å, 6 Å
   cutoff — differentiable, one radius query, trivially batched. Our weighted contact number already matches −RSA
   at predicting site sensitivity (0.602 vs 0.606 [own]). In the AF 70–90 pLDDT band prefer Cβ counts / half-sphere
   exposure over all-atom measures, because AF backbones are far more reliable than its side chains.
7. **Soft-cap repulsion.** FoldX's clash terms have sd 3.2 kcal/mol, its `backbone_clash` is *anti*-correlated with
   true ΔΔG, and capping the total at +3 kcal/mol changes ρ by 0.003 [own §1.1]. Use Rosetta-style linearised
   `fa_rep` below 0.6σ and fit the softening parameter. Essential on AF inputs, where small clashes are model error.
8. **Relax WT and mutant identically, locally, in our own energy.** Valanciute 2023: with poor models FoldX collapses
   to the structure-free null while Rosetta cartesian_ddg (local Cartesian minimisation of a 6 Å shell) stays near
   crystal performance. Guerois 2002 found GROMOS minimisation did *not* help FoldX — minimising in a different
   potential than you score with adds a systematic offset. One restrained relax per WT structure, then identical
   local moves for WT and mutant.
9. **Average over a small ensemble.** Sapozhnikov 2023 narrowed FoldX's 95% interval from 3.78 to 2.89 kcal/mol by
   averaging over 100 MD snapshots; Degn 2025 found an MD ensemble input gives FoldX +0.07 PCC on AF structures
   (and coarse-grained CABS-flex ensembles *hurt*); flex-ddG's entire advantage is backrub ensembles. 8–32 perturbed
   copies per mutation is cheap on GPU and yields a per-prediction uncertainty — a real differentiator, since
   FoldX's own 95% interval is ±2.9 kcal/mol and >50% of its predicted-stabilising calls are wrong.
10. **Keep the parameter count in the 20–60 range and fit with protein-grouped CV.** PoPMuSiC fits 52 (13 potentials
    + 2 volume terms, each weighted by a 4-parameter sigmoid in solvent accessibility), KORPM 12, FoldX 5, DDGun 0.
    Evidence that more is worse: refitting FoldX's own 22 terms on 215k training variants *lowers* out-of-sample
    performance (0.569 → 0.541 pooled) [own §1.1].
11. **Fit on Megascale-train only; hold out everything else.** Subsample to ~50–200 variants per domain so large
    scans do not dominate, drop modified WT backgrounds, exclude any MMseqs 25% cluster that appears in an
    evaluation set, and clip targets to |ΔΔG| ≤ 4 kcal/mol to avoid fitting assay saturation. Use a robust/censored
    loss: the ΔΔG sd is only 1.00 kcal/mol and the assay range is clipped [own §6.5].
12. **Benchmark against the right competitors, not just FoldX.** Run **KORPM** (12 parameters, ~5 ms/mutation,
    Ssym PCC 0.69) and **DDGun3D** (zero fitted parameters, best classical on S461 at Pearson 0.635) on our own
    tables. If PhysDDG does not clearly beat both, it is not a contribution. Also beat the internal PhysDDG v1.2
    (Megascale 0.640 pooled / 0.701 per-protein).
13. **Report trivial baselines in every table**: burial alone 0.461/0.514 and a 7-parameter model 0.536/0.600 on
    Megascale-test [own §6.9]; RSA alone 0.468 on Megascale and 0.456 on S461 (independent measurement);
    a single Bastolla–Vendruscolo contact potential gets Pearson ≈ 0.50 on S2648.
14. **AF mode: trim, keep AF side chains, relax with pLDDT-scaled restraints, shrink at low confidence.**
    Trim termini/linkers below pLDDT 70 and 10-residue windows below 50 *unless packed*; crop at PAE domain
    boundaries; repack only site + 6–8 Å shell (never globally — AF side chains beat every packer on AF backbones);
    `ΔΔG_AF = w(pLDDT)·ΔΔG_phys + (1−w)·c` with w logistic (2 parameters); report stratified by pLDDT band.
    Validate on paired xtal/AF sets and on **S612** (target: Pearson > 0.45, and no worse than on crystals).
    Never run AF on the mutant; never use ΔpLDDT as signal.
15. **Flag missing context rather than silently mispredicting it**: metal sites (Cys/His clusters, zinc-finger
    Pfams — *all* predictors fail on Domainome zinc fingers), cofactors, interfaces, membrane segments.
    A "low-confidence, missing cofactor" flag is more useful than a wrong number.
16. **Evaluate tmol before writing CUDA.** https://github.com/uw-ipd/tmol (Apache-2.0, PyTorch + CUDA) already
    provides batched REF2015-family scoring, a Dunbrack packer, minimisation and FastRelax on GPU, with no neural
    network. Use it as a dependency, a reference implementation, or at minimum a baseline — do not reinvent it blindly.
17. **Architecture for speed: fixed-size local environments + precomputed pair tables.** Gather the ≤ 12 Å shell,
    pad to K ≈ 32–64, and evaluate `[n_mut, n_rot, K, atoms, 3]` dense tensors; precompute one-body and two-body
    rotamer energies once per site and reuse across all 19 substitutions; solve the small packing problem with
    DEE + exact search or **batched mean-field / belief propagation** (which also yields side-chain entropy directly).
    Avoid TorchMD's mistake: dense pair loops without neighbour/cell lists cost ~60×. float32, linearised repulsion,
    CUDA graphs, one process per GPU. **Target ≤ 1–10 ms/mutation**; KORPM already does 5 ms on one CPU core,
    so "fast" alone is not a contribution.
18. **Rotamer library: start from the copies in FASPR (MIT) or tmol (Apache-2.0)** — both redistribute Dunbrack 2010
    (CC BY 4.0) — and **apply for the official dunbrack.fccc.edu licence in parallel** (free but manually approved;
    flag this to the user now). Fallback: derive our own from Top2018 (Zenodo, CC) with a von Mises KDE of
    P(rot|φ,ψ); ~1 day of work.
19. **Measure the stabilising-mutation regime separately.** Only 4.2% of Megascale is stabilising, and *every*
    published method has **negative** Spearman below ΔΔG < −2 kcal/mol (UniStab). Report precision@k and AUROC for
    stabilising mutations; a physics model with a correct reference state is the most likely method to fix this.
20. **Break every result down by class from day one**: natural vs designed, buried vs exposed, apolar vs polar vs
    charged, to-Pro / from-Gly, large→small vs small→large, crystal vs AF, pLDDT band. Guerois's Table 4 shows the
    term balance differs qualitatively between these classes, so a pooled number hides exactly the failures that matter.
21. **Consider shipping a FoldX/PhysDDG consensus as a secondary product.** StaB-ddG + FoldX beats either alone on
    SKEMPI (0.56 vs 0.49/0.49), and Gerasimavicius' "Foldetta" (mean of FoldX and Rosetta) beats both on DMS data.
    If PhysDDG's errors are decorrelated from FoldX's, say so and quantify it.
