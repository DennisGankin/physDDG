# PhysDDG engine design (v0, 2026-10-26, PI draft — to be critiqued)

## Model form
ΔΔG_pred = g( Σ_k w_k · ΔE_k  +  Δref(wt→mut) ;  context )

- ΔE_k = E_k(mutant model) − E_k(WT model), computed on the *same* local region with the *same* protocol
  (WT is re-modelled with the identical repacking procedure, so packing artefacts cancel).
- Δref = r(mut) − r(wt): unfolded-state reference energies (20 values, shrunk towards a physical prior
  derived from the per-atom EEF1 reference solvation of each side chain).
- g: monotone saturating link (soft-clip at the assay's dynamic range) + context factors
  (AF mode: pLDDT-dependent attenuation). Fitted with a robust loss.
- Target size: ≤ ~15 term weights + 20 reference energies (+ a handful of link / AF parameters).
  Every parameter must have a physical reading (kcal/mol per unit, or dimensionless scale).

## Representation
- Heavy atoms only + geometrically placed polar hydrogens where needed for H-bond angles
  (backbone N–H from the bisector rule; side-chain donors use heavy-atom geometry).
- Per-protein tensors on GPU: coords [N,3], atom type id, residue index, backbone/side-chain flag, charge.
- Atom types (~20): C_bb, CA, N_bb, O_bb, CH1/CH2/CH3 aliphatic, C_aro, C_carbonyl(sc), N_amide, N_aro(His/Trp),
  N_charged(Lys NZ, Arg NE/NH), O_hydroxyl, O_amide, O_carboxylate, S (Cys SG, Met SD).
  Parameters: vdW radius/well (CHARMM19/EEF1-like), EEF1 (V, ΔG_ref, ΔG_free, λ), partial charge (formal charges on ionisables only),
  H-bond donor/acceptor flags.

## Mutant modelling (fixed backbone, local flexibility)
1. Rotamer library: backbone-dependent, derived from a high-resolution non-redundant PDB set in /rsyncPDB
   (licence-free, reproducible; same data also gives a Ramachandran potential and per-aa side-chain entropies).
   Include the native rotamer of every residue as an extra candidate ("include current").
2. Side chains built on GPU by NeRF from chi angles (vectorised over positions × aa × rotamers).
3. Energies decomposed into one-body (rotamer vs fixed backbone + intrinsic rotamer probability) and two-body
   (rotamer–rotamer between neighbouring positions). Precompute for all positions × 20 aa × R rotamers once per protein;
   WT–WT pair tables are shared by all 19·L mutations, only pairs involving the mutated site with the mutant aa are new.
4. Packing sub-problem per mutation: mutated site + neighbours within ~8 Å (side-chain flexible), everything else fixed at
   the WT-packed state. Solved for all mutations in one batch on GPU (DEE pruning + batched iterated conditional modes /
   simulated annealing; problems are small: ~10–20 residues × ≤ 20 rotamers).
5. Optional continuous refinement: a few gradient steps on chi angles (autograd on the soft potential).
6. Backbone flexibility approximated by capped/softened repulsion (a fitted cap) — the main known failure of
   fixed-backbone methods is over-penalising small→large substitutions.

## Energy terms (each reported separately per mutation; these are the features ΔE_k)
| k | term | form | notes |
|---|------|------|-------|
| 1 | vdW attraction | LJ 6-12 attractive branch (or LK-style well), 8 Å cutoff | |
| 2 | vdW repulsion | capped, linear-ish beyond overlap | soft for backbone flexibility |
| 3 | apolar solvation / burial | EEF1 Gaussian exclusion on C/S atoms (or ΔSASA × σ_apolar) | hydrophobic effect |
| 4 | polar desolvation | EEF1 on N/O atoms | penalty for burying polar groups |
| 5 | H-bond sc–bb | distance + angle (DSSP/Rosetta-like) | |
| 6 | H-bond sc–sc | same functional form, separate weight | |
| 7 | electrostatics | Coulomb, ε = 4r or Debye–Hückel screened, 12–15 Å | ionisable groups + backbone dipoles optional |
| 8 | side-chain entropy | per-aa S_conf (from rotamer-library entropy) × burial fraction | loss of rotamer freedom on folding |
| 9 | backbone torsion | −kT ln P(φ,ψ | aa) at site (and i−1 for X→Pro, pre-Pro map) | Gly/Pro/β-branched |
| 10 | helix propensity | Pace–Scholtz-like, applied if site in α-helix (DSSP) | possibly subsumed by 9 |
| 11 | cavity / packing defect | Δ buried volume or loss of contacts on large→small | mostly covered by 1/3 |
| 12 | special cases | disulfide broken, cis-Pro lost, Gly in positive-φ, metal ligand lost | indicator × fitted penalty |
| 13 | helix dipole / capping | charge near helix N-/C-terminus | |

## Datasets & fitting
- Fit on Megascale-train (ThermoMPNN split) + FireProt-trainval jointly (assay-specific offset/scale as nuisance),
  grouped CV over homology clusters for all model selection.
- Held-out evaluation: Megascale-test (all + strict non-homologous), FireProt-test, S669/S461, Ssym (antisymmetry),
  Domainome (AF structures, fitness proxy, per-domain Spearman), paired xtal/AF sets.
- Report FoldX 5.1 native on identical rows, and trivial baselines (RSA, hydrophobicity×burial, BLOSUM).

## AlphaFold mode
- Use pLDDT: (a) trim/neutralise residues with pLDDT < 50 (treated as solvent-exposed coil),
  (b) pLDDT-dependent attenuation of the predicted ΔΔG at the site (fitted sigmoid, 2 params),
  (c) re-pack all side chains with the same protocol before scoring (AF side chains are biased),
  (d) construct cropping for isolated-domain assays.
- Evaluate on paired xtal/AF variants and Domainome; AF mode must not hurt on high-pLDDT xtal-like regions.

## Speed
- One GPU batch per protein (all 19·L mutations); structure prep on CPU in parallel workers; multiple GPUs by sharding proteins.
- Target: full saturation mutagenesis of a 300-residue protein in < 1 min on one RTX 4090 (FoldX: ~hours).

## Staging
- Stage 1 (lite): mutant side chain packed into a fixed WT environment (no neighbour repacking), all energy terms,
  fitting + evaluation end-to-end → first numbers.
- Stage 2: neighbour repacking (batched GPU packer), continuous refinement.
- Stage 3: AF mode, extra terms, ablations, speed work.

---
# v1 revisions after independent critique (notes/critique_design_v0.md), 2026-10-26

Accepted (PI decisions D4–D10):
- **D4 Environment-dependent statistical potential** E_env(aa | burial × SS [× φ/ψ basin]) from the culled PDB set,
  one fitted weight; table entries are structural statistics, not ΔΔG-fitted. Highest-priority addition.
- **D5 Boltzmann averaging** over rotamers (F = −kT ln Σ e^{−E/kT}) instead of min-energy; side-chain entropy comes
  out of it (term 8 dropped as a separate term, kept only as an ablation). Packer = batched mean-field / Gibbs, not DEE.
- **D6 Symmetric modelling**: mutated site uses library rotamers only for both WT and mutant (no "include current"
  at the site); neighbours may carry their native rotamer (identical in both states). Flexible set = union of WT and
  mutant neighbourhoods. Hard test: WT→WT returns exactly 0.
- **D7 Reference energies** r = r_prior(EEF1) + δ_shared (strong shrinkage) + δ_Megascale (nuisance, dropped at
  inference) because Tsuboyama ΔG already contains a PSSM unfolded-state model; X→Pro / Pro→X handled with care
  (proteolysis channel).
- **D8 Censoring-aware objective**: Megascale fitted in ΔG space with interval (Tobit-style) likelihood using
  ΔG_wt and CI where available; per-domain-centred robust loss + within-domain rank surrogate; dataset weights by
  clusters not rows; per-assay scale/offset nuisance. Domainome only as validation (sigmoidal per-domain link),
  not for fitting energy weights.
- **D9 Site-dependent repulsion cap** cap = c0 + c1·flex(site) (contact number; B-factor or pLDDT) — the fixed-backbone
  small→large fix; backbone ensembles only if residuals demand it.
- **D10 Electrostatics** screened (Debye–Hückel ~0.15 M) with burial-dependent attenuation; His neutral; must be shown
  to help in ablation or reported as not helping. Helix dipole/capping, cavity and helix-propensity terms deferred.

Frozen-parameter rule: radii, well depths, EEF1 values, H-bond geometry, cutoffs are taken from literature and listed in
physddg/params.py with provenance; anything tuned after seeing a test number counts as fitted and is disclosed.

## Baseline ladder & pre-registered gates (identical rows; grouped CV; natural-only Megascale primary)
B0 residue-pair lookup · B1 RSA · B2 hydrophobicity×burial+Δref · B3 statistical potential + burial (2 weights) ·
B4 Stage-1 lite (all terms, Boltzmann mutant side chain in fixed WT environment) · B5 +neighbour repacking ·
B6 +backbone ensemble · B7 +continuous χ refinement · X FoldX 5.1 native.
- G1 physics earns its keep: B4 ≥ X + 0.08 per-domain Spearman (Megascale-test natural) and ≥ +0.05 Pearson FireProt-test; B4 − B3 ≥ 0.03 (if < 0.02 say so).
- G2 repacking: B5 − B4 ≥ 0.02 overall and ≥ 0.03 on buried small→large, in ≥ 70% CV folds, within speed target; else drop.
- G3 backbone ensemble only if buried volume-change residual slope significant; then B6 − B5 ≥ 0.015.
- G4 continuous refinement: ≥ 0.01 at ≤ 2× runtime.
- G5 AF mode: paired xtal/AF per-site agreement r ≥ 0.9 after uniform repacking; pLDDT attenuation ships only if Domainome +0.02 and Megascale-test ≥ −0.005.
- G6 transfer: weights fitted on Megascale-only vs FireProt-only — no sign flips, no > 3× changes; failing terms shrunk/deleted.
- G7 leakage: every headline reported on full and strict non-homologous (< 25–30% id) subsets; if gap > 0.05 the strict number is the headline.
Note (PI): the critic proposes Stage 2 only after G1/G6. I keep the GPU packer on the critical path in parallel because
(a) it is needed for AF uniform repacking (G5) and (b) Boltzmann averaging over a single site already needs most of it.
