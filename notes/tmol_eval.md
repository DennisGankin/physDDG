# tmol as a ΔΔG baseline and as a per-term feature source (2026-10-26)

tmol (https://github.com/uw-ipd/tmol, Apache-2.0) is the IPD's PyTorch/CUDA port of the Rosetta energy, packer and
minimiser. It contains no neural network. Code: `scripts/60_tmol_ddg.py` (ΔΔG + per-term ΔE, runs in `.venv_tmol`),
`scripts/60b_tmol_queue.sh` (all runs reported here), `scripts/61_tmol_refit.py` (linear refit, main `.venv`).
Outputs: `results/tmol/` (per-variant parquet/CSV files are git-ignored and can be regenerated; metrics JSON/CSV, speed CSVs and
`refit_{pack,min}.json` are committed).

## 1. Install: worked first time, no build step
- `uv venv --python 3.12 .venv_tmol && uv pip install torch==2.14.1 "tmol[cu13]==0.2.0"`, which took about 1 min.
  PyPI ships prebuilt CUDA operators (`tmol-ops-cu13==0.2.0`), so no nvcc or CUDA toolkit was needed (this machine
  has neither). `tmol.has_cuda()` returns True on the RTX 4090. Exact versions are in `requirements_tmol.txt`.
  The repository was also cloned to `ext/tmol` (git-ignored) at commit `60b0d77` (2026-10-23), only to read docs and source.
- The install brings in heavy dependencies (atomworks, biotite 1.6, rdkit, openbabel-wheel, numba). Keeping them out of the main venv was the
  right call.
- Cosmetic problems: atomworks prints `PDB_MIRROR_PATH` / `CCD_MIRROR_PATH` warnings. The first pose build in a process
  takes 20–110 s (database initialisation, slower when the CPUs are busy); later proteins load in about 1 s.

## 2. Protocol (`60_tmol_ddg.py`)
- Input chain, protein residues only (HETATM, water and ligands dropped). tmol adds hydrogens. Score function `beta2016`
  (beta_nov2016_cart; this is the only full-atom weight set tmol ships, so REF2015 itself is not available).
- Shell = residues with any heavy atom within 8 Å of any heavy atom of the WT site residue (a Gly site adds a virtual CB).
  The same shell is used for the WT reference and for all 19 mutants at that site.
- Packing: everything outside the shell is frozen. Shell residues repack with their identity fixed (Dunbrack rotamers plus the current
  rotamer). The site gets the WT or mutant identity with Dunbrack rotamers only and no current rotamer (DESIGN D6). Simulated
  annealing uses a fixed seed. Three seeds gave identical ΔΔG on 1GYZ (Spearman 1.000), so the annealer converges on these small
  problems and `--seeds 1` is used.
- `--min` adds kinematic minimisation of the side-chain torsions of all shell residues (LBFGS-Armijo, max 50 iterations), applied identically to WT and mutant.
- The whole pose is scored. ΔΔG = E(mut) − E(WT reference of that site), with weighted per-term ΔE in `d_<term>` (they sum to `ddg_reu`).
  `ddg_pred = ddg_reu / 2.94`.
- Batching: the WT and mutant poses of a protein go into one `PoseStack` (chunks of 512). Per-pose `PackerTask` masks set the shell,
  the site identity and the samplers.
- Terms tmol gives: fa_ljatr, fa_ljrep, fa_lk, fa_elec, hbond (one combined term; tmol has no bb/sc split, which would need
  `render_block_pair_scoring_module`), lk_ball_iso/lk_ball/lk_bridge/lk_bridge_uncpl, rama (includes p_aa_pp), omega,
  dunbrack_rot/rotdev/semirot, ref, cart_* (the fixed backbone still changes slightly at the site), disulfide.
- Failures:
  - 3 of 347 FireProt-test rows: tmol cannot mutate a disulfide-bonded Cys (CYD; 2TRX C32A and C35A), and for 1RN1 Y45W no Trp was
    placed at the site. These rows are marked `pack_failed`.
  - tmol silently drops terminal residues that lack main-chain atoms (1QGV, 1RBP). The script maps residues through tmol's own labels.
  - Megascale: 0 failures.

## 3. Accuracy (identical rows to FoldX 5.1 native; bootstrap over clusters, n = 300)
Megascale-test: 28 proteins, 28,172 variants, 100% coverage. FireProt-test: 25 proteins, 338 rows where both predictors exist.

| predictor | MS Spearman | MS per-prot median | MS Pearson | MS AUROC(>1) | FP Spearman | FP Pearson |
|---|---|---|---|---|---|---|
| FoldX 5.1 | 0.569 | 0.633 | 0.447 | 0.829 | 0.454 | 0.417 |
| tmol raw, repack only | 0.463 | 0.507 | 0.213 | 0.768 | 0.256 | 0.011 |
| tmol raw, repack + χ-min | 0.492 | 0.535 | 0.184 | 0.784 | 0.356 | 0.116 |
| tmol refit, 12 terms (min) | 0.575 | 0.598 | 0.611 | 0.821 | 0.432 | 0.443 |
| **tmol refit, 12 terms + 20 ref (min)** | **0.596** | **0.626** | **0.629** | **0.833** | **0.493** | **0.486** |
| same, repack only | 0.591 | 0.621 | 0.618 | 0.829 | 0.460 | 0.477 |

- **Raw tmol ΔΔG is clearly worse than FoldX.** Megascale paired Δ Spearman −0.076 [−0.13, −0.03] with χ-min. The cause is
  fa_ljrep: 17% of variants have ΔE > 20 REU (p99 = 470 REU) from fixed-backbone clashes, so Pearson collapses (0.18–0.21).
  χ-minimisation helps: +0.029 [+0.016, +0.040] Spearman on Megascale, +0.10 on FireProt.
- **Refit** (`61_tmol_refit.py`, `physddg/fit.py`, Gaussian loss, proteins weighted equally):
  - Fitted on a Megascale-train subset: 40 proteins, one per homology family, 37,753 rows.
  - All selection used only 5-fold family-grouped CV on that subset: the repulsion transform (clip at 1/2/5/10/20/none vs signed log1p),
    pack vs min, and the variant. Signed log1p(fa_ljrep) was best in both modes. min + ref was best (CV per-protein 0.643 vs 0.631 for pack).
  - On Megascale-test the refit matches or slightly beats FoldX: Spearman +0.027 [0.000, +0.055], per-protein median −0.007 [−0.05, +0.07],
    Pearson 0.63 vs 0.45.
  - On the strict subset with no train homologue (19 proteins, 20,156 rows): 0.595 vs 0.571 Spearman, per-protein median 0.620 vs 0.629.
  - FireProt-test: 0.493 vs 0.454 (CI includes 0).
- Fitted weights (kcal/mol per REU, min + ref): atr 0.11, log-rep 0.20/log-unit, sol (fa_lk) 0.13, lk_ball 0.09, elec 0.13,
  hbond 0.07, dun 0.03, rama 0.10, omega 0.09, cart ≈ 0, ref ≈ 0 (absorbed by the 20 fitted reference energies).
  All weights are positive and similar in size: about 1/3 of the nominal 1/2.94 = 0.34, which is the usual shrinkage of a noisy predictor.
  The 20 fitted reference energies add about +0.02 (consistent with D7).
- **Complementarity and comparison with the baseline ladder** (`notes/baseline_ladder.md`).
  - B3+ is a statistical potential + burial + extras, 10 weights + ref, fitted on all of Megascale-train. On Megascale-test it scores
    0.603 / 0.649 / r 0.631 (Spearman / per-protein / Pearson). tmol-refit is about equal: 0.596 / 0.626 / r 0.629.
  - On FireProt-test (338 rows) tmol-refit has the higher Pearson: 0.486 vs 0.456.
  - The two correlate at Spearman 0.76. An unfitted equal-weight sum of z-scores gives:
    - B3+ + tmol: 0.635 / 0.677 / r 0.669 (+0.03 over B3+);
    - FoldX + tmol: 0.635 / 0.691 (tmol and FoldX correlate at 0.66).
  - The z-scores were computed on the test pool, so these sums are indicative only.

## 4. Speed (RTX 4090, float32; "poses" = WT references + mutants, about 1.05 per variant)
| workload | repack only | repack + χ-min |
|---|---|---|
| 1GYZ (59 aa) SSM, GPU to itself | 128 poses/s | 38 poses/s |
| Megascale-test, 2 jobs sharing GPU 3 (43–72 aa) | 57 poses/s per job | 47 poses/s per job |
| 2ADA (349 aa) full SSM, 6,980 poses (CPU busy with other jobs) | 37.5 poses/s, i.e. 3.1 min | 9.5 poses/s, i.e. 12.2 min |

That is 8–25 ms per mutation for small domains and 27–105 ms per mutation for 350 aa. FoldX takes 5–34 CPU-s per mutation, so one GPU running tmol ≈ 64 FoldX cores up to about 10× that.
Megascale-test took about 10 min per mode. The DESIGN target (a 300-aa SSM in under 1 min) is missed by 3× without minimisation and 12× with it.
Bottlenecks, from cProfile and tmol's verbose stage timings:
1. Every mutant pose rebuilds its own rotamer set and interaction graph from scratch. The WT one- and two-body tables are not
   reused across the 19 substitutions or across overlapping shells; in each packing chunk, building the rotamers and the graph is about 50% of the time.
2. The packer splits the batch into chunks of 25 poses (`_SMALL_PACKING_POSE_CHUNK`), with about 0.3–0.5 s of Python and launch overhead per
   chunk. `TMOL_PACK_MAX_POSES_PER_CHUNK=128` gave only +30%.
3. Minimisation and final scoring evaluate the whole pose for every mutant (no locality), so cost grows with protein length × number of mutants.

## 5. tmol vs our own engine (E1 builder/rotlib, E2 energy)
- **What tmol has that our stack does not have (yet):** Rosetta's mature terms, i.e. lk_ball anisotropic polar desolvation,
  a combined rama/p_aa_pp backbone term and Dunbrack fa_dun, plus a validated simulated-annealing packer and a χ/Cartesian minimiser.
  With only 12 weights it already reaches FoldX level on held-out data.
- **What we already have:** `physddg/mutate.py` (stage 1 Boltzmann site; stage 2 batched mean-field with flexible neighbours, WT neighbour
  tables shared across all sites) runs a 490-aa saturation scan (9,234 variants) in 18.7 s and a 245-aa scan in 9.7 s
  (`results/mutate/runtime_s2_*.csv`). That is about 500 variants/s, or **13–50× faster than tmol repack-only**, because
  it implements the shared-table design that tmol lacks.
  E2's energy reaches 53–83k single-site variant evaluations/s (`notes/engine_energy.md`).
  So tmol is not needed for speed or for packing capability. Its value is the energy function, and as an independent reference.
- **Why tmol should not be the core engine dependency:**
  - (a) Speed: tmol's design repacks full poses independently and cannot meet the speed target without rewriting its packer orchestration.
  - (b) Our design brief is a few physically interpretable terms from frozen literature parameters. beta2016's roughly 1000 atom-type parameters were
    themselves fitted in the Rosetta optimisation (Park et al. 2016). Whether stability ΔΔG data were among its targets is
    **[unverified]**. If they were, that is a leakage-like caveat for any tmol-based headline.
  - (c) Heavy dependency stack, and beta2016 only (no REF2015 weights file).
  - (d) Licence check needed: tmol bundles `dunbrack.bin`. LITERATURE §5.2 says Dunbrack 2010 is CC BY 4.0; E1's G1 says
    academic-only, no redistribution. Resolve this before anything tmol-dependent ships.

## 6. Recommendations
1. **Use tmol as the reference baseline, not as the engine.** Add "tmol-refit (12 terms + ref, χ-min)" to the baseline ladder.
   It is a Rosetta-physics rung that is about as good as B3+ and FoldX (Megascale-test 0.596 / 0.626 per-protein).
   If our engine (B4/B5) does not beat tmol-refit, our own physics adds nothing over off-the-shelf Rosetta terms.
2. **Use tmol as an oracle to validate E1/E2.**
   - Score the same tmol-packed WT and mutant structures with `physddg/energy.py` and correlate the matched terms
     (atr↔lj_atr, fa_lk↔EEF1, hbond↔hb_*, fa_elec↔elec).
   - Compare our packer's χ recovery and energies against tmol's packer on identical shells.
   - `60_tmol_ddg.py` already writes all per-term ΔE.
3. **Port or design-in what the refit says matters:**
   - (i) Soft or log-saturating repulsion as the ΔE feature. Signed log1p beat every hard cap in CV, which supports D9.
   - (ii) Anisotropic polar desolvation (lk_ball), weight about 0.09, not absorbed by fa_lk.
   - (iii) A backbone term with p_aa_pp + rama (weight 0.10) plus omega.
   - (iv) Fitted reference energies (+0.02).
   - (v) χ-minimisation of the shell (+0.03 raw). Our G4 gate should expect a similar gain.
   - fa_dun and the cart terms carry little once these are present.
4. **Ensembling or features:** tmol adds information beyond both B3+ and FoldX (naive sums +0.03 and +0.06 ρ).
   A proper test is to add tmol ΔE (or just its log-rep, lk_ball and elec) to the B3+ feature set in `fit.py` and use grouped CV.
   This needs `60_tmol_ddg.py` on all of Megascale-train (about 1 h per mode on one GPU).
   Any such model must be reported as a separate variant ("+Rosetta terms"), not as PhysDDG's own energy.
5. **If tmol is used beyond baselines:** use it as a dependency in `.venv_tmol` only, called through `60_tmol_ddg.py`. Do not port its CUDA kernels.
   Our dense residue-block design with shared WT pair tables is the faster architecture for single-site scans.

Reproduce: `scripts/60b_tmol_queue.sh pack` and `... min` (GPU 3, about 35 min each including the train subset); then
`PYTHONPATH=. .venv/bin/python scripts/61_tmol_refit.py --mode {pack,min}`; then `scripts/30_evaluate.py --pred results/tmol/<file>
--compare results/baselines/foldx_<bench>.csv:ddg_pred`.
