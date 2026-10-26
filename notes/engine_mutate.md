# Mutation / packing engine (E1) — physddg/mutate.py, scripts/55–57

Status 2026-10-26. Tests: `tests/test_mutate.py` (7 pass). Features: `data/features/phys_s{1,2}_<dataset>.parquet`
(git-ignored; regenerate with `scripts/55_phys_features.py --stage 1|2`, per-structure cache in
`data/features/cache/phys_s<stage>_<tag>/`, tag = hash of MutateConfig + energy/geometry sources + statistics tables).

## Protocol
For every needed site and **all 20 residue types (WT included, identical code path)**:
- Candidates: library rotamers at the site's (φ, ψ) (`RotamerLibrary.query`, cum ≥ 0.97, ≤ 30 per type; no
  native "include current", D6), built by `builder.build_sidechains` with the **native CB** (ideal CB if WT is Gly or
  exactly one of WT/target is Pro). Gly/Ala: one state.
- Energies (E2, `energy.py`, rep_scale 0.9 per D14, cap 3 kcal/mol/pair; `cap0 + cap1·(1−burial)` exposed for D9):
  intra-residue + interaction with the protein without the site residue (`env_energy` with a 1-D env mask per site).
  EEF1 reference solvation (dG_ref) is **excluded** from all energies and reported separately as `dref_apolar/polar`
  (unfolded-state reference, D7). Packing energy U = unit-weight sum of lj_atr, lj_rep, solv_apolar, solv_polar,
  hb_sc_bb, hb_sc_sc, hb_bb_bb, elec_dh, disulf, plus −kT ln p(rot|φ,ψ) (renormalised over the kept rotamers), kT = 0.593.
- **Stage 1** (fixed environment): F(a) = −kT ln Σ_r exp(−U_r/kT); Boltzmann-averaged per-term ⟨E_k⟩, site entropy,
  ⟨prior⟩, min-U versions (`dEmin_*`, `dU_min`) for ablation.
- **Stage 2** (neighbour flexibility): flexible set N(i) = residues with chis (not Pro, complete side chain) that have
  a movable atom (beyond CB) within **6 Å of any candidate atom (CB onwards) of any of the 20 residue types** at i
  (so WT and all mutants share it; ≤ 16 nearest). Mean 10.6 neighbours per site (Megascale), 12.9 (FireProt).
  Neighbour candidates: native side chain (exact coordinates) + ≤ 12 library rotamers (cum 0.9), identical in all
  states; prior = library probability (native gets its bin's probability), renormalised.
  Site rotamers: top ≤ 20 per type. Shared per protein: neighbour one-body vs the whole protein, candidate–native
  pair tables (used to remove the flexible residues from the fixed environment) and candidate–candidate tables;
  per site: site-rotamer × neighbour-candidate pairs. Batched damped mean-field (30 sweeps, damping 0.5, fixed
  initialisation → deterministic) over all (site, aa) of a 48-site chunk. Output: mean-field free energy of the local
  region F = Σ⟨U⟩ − kT·S_MF (site + neighbours + their interactions with the fixed rest) and mean-field-averaged
  per-term energies over the same region.
- Statistical / context features (same rows): ΔE_rama (site; pre-Pro map if i+1 is Pro), ΔE_rama_prev (i−1 switches
  to the pre-Pro map when the target is Pro), ΔE_env (cn×SS, cn×SS×basin, RSA×SS; D4), Δ(kT·S_rotameric),
  Δposphi; context: φ, ψ, ss3, cn, ncb10, B/pLDDT, HSE-up, RSA; stage 2 also n_nb.

Feature columns: `s<stage>_dF`, `s<stage>_dE_<term>` (11 terms), `s1_dEmin_<term>`, `s1_dU_min`, `s<stage>_dTS`,
`s<stage>_dprior_avg`, `s<stage>_dref_apolar/polar`, `nrot_wt/mut`, `d_rama`, `d_rama_prev`, `d_env_cn_ss`,
`d_env_cn_ss_bb`, `d_env_rsa_ss`, `d_sc_entropy`, `d_posphi`, `ctx_*`, flags `site_found`, `wt_match`.
d<x> = x(site, mut) − x(site, wt). Rows = `features_simple.dataset_rows` (deduplicated), all keys preserved.
`wt_match` = structure residue equals `wt` (paired_af reverse rows on AF models: 0.962; features still defined).

## Hard checks (tests/test_mutate.py)
- WT→WT ΔE = 0 exactly (same tensor entry), both stages; bitwise deterministic reruns (stage 1 and 2).
- Batch independence: a site's values do not depend on which other sites are in the batch (< 1e-4 / 1e-3 kcal/mol).
- Stage 1 Ala ⟨E⟩ equals a direct intra + env_energy evaluation (< 1e-4).
- Stage 2 with an empty flexible set reproduces stage 1 exactly (MF on one variable = exact Boltzmann; < 2e-3).
- F ≤ U_min and F ≥ U_min − kT ln n_rot; neighbour sets identical for all 20 aa; native CB kept / ideal CB for Gly.

## Coverage, runtime (all 9 tables: megascale incl. censored, fireprot, s669, s461, ssym, paired_af (AF + xtal),
abundance_dms, domainome = 1713 structures, 51,737 sites, 983k variants)
- site found 100% (s669 99.7%: 2 rows; paired_af_xtal 99.95%), wt_match 100% except paired_af AF side.
- Stage 1: 248 s wall on 2 RTX 4090 (GPU 0–1) under heavy CPU contention from other jobs (load avg 70–120 on 64 CPUs);
  524 GPU-process-seconds → **≈ 1,900 mutations/s per GPU** (≈ 0.5 ms/mutation; 0.6 s for a 86-residue domain
  saturation scan); peak GPU memory 1.6 GB.
- Stage 2: 912 s wall; 1,781 GPU-process-seconds → **≈ 550 mutations/s per GPU** (1.8 ms/mutation; 63-residue domain:
  1.9 s, 30 ms/site); peak GPU memory 4.8 GB. Costs: site one-body env evaluation ~45%, pair tables ~25%, Python glue.
- FoldX reference: 5–34 CPU-s per mutation.

## First sanity numbers (raw ΔF, unit weights, nothing fitted; scripts/57_raw_dF_sanity.py)
| | Megascale val ρ | Megascale test ρ | test per-protein median ρ (natural) | FireProt-test ρ (r) |
|---|---|---|---|---|
| stage 1 ΔF | 0.521 | 0.507 | 0.535 (0.535) | 0.276 (0.09) |
| stage 2 ΔF | 0.533 | 0.519 | 0.557 (0.561) | 0.308 (0.16) |
| FoldX 5.1 (baseline ladder) | – | 0.569 (MS all) | 0.629 (MS natural) | 0.449 (0.411) |
Stage 2 adds +0.012 (pooled) / +0.02 (per protein) / +0.03 (FireProt) over stage 1 with unit weights. Pearson is low
because unit weights give huge outliers (X→Pro ≈ +35 kcal/mol, clashes). Adding the EEF1 dG_ref back (dF + dref)
destroys the correlation (≈ 0), i.e. reference energies must be fitted (D7), as expected.
Ssym antisymmetry of raw ΔF is poor (direct vs −reverse r = 0.13 / 0.04; mean bias +8.0 / +6.3 kcal/mol): both
directions look destabilising because the fixed backbone/environment is adapted to the residue present — the main
known failure of fixed-backbone methods; the cap (D9) and fitted weights must address it.
Per-term single-feature Spearman (FireProt-test, stage 1): lj_atr 0.26, solv_apolar 0.26, elec 0.15, lj_rep −0.20,
solv_polar −0.17; statistical: env_cn_ss_bb 0.31, env_cn_ss 0.27, rama 0.23.

## Rotamer recovery (structural calibration; scripts/56_rotamer_recovery.py; held-out PISCES test chains)
60 held-out chains (≤ 300 residues), 6,511 well-ordered residues with chis; site re-predicted from library rotamers
only (no native candidate). Within 40°:
| | chi1 all | chi1+2 all | chi1 buried (rSASA<0.25) | chi1+2 buried | chi1 exposed | chi1+2 exposed |
|---|---|---|---|---|---|---|
| prior only (top library rotamer at φ,ψ) | 0.746 | 0.599 | 0.750 | 0.604 | 0.740 | 0.593 |
| stage 1 (min U, fixed native environment) | **0.915** | **0.845** | 0.963 | 0.937 | 0.853 | 0.738 |
| stage 2 (max MF probability, flexible neighbours) | 0.905 | 0.830 | 0.957 | 0.927 | 0.839 | 0.717 |
By aa (stage 1 chi1 / chi1+2): F .99/.98, W .98/.94, Y .98/.94, L .98/.95, I .99/.95, V .98, T .96, M .93/.85,
H .93/.69, D .90/.87, P .93/.89, N .88/.74, K .86/.72, R .85/.69, E .84/.74, Q .84/.69, C .92, **S .71** (hydroxyl
without explicit H; weakest). Stage 2 is slightly below stage 1 because neighbours move too (native context partly
lost) and mean field is approximate. Per-aa/burial table: `results/mutate/rotamer_recovery_summary.csv`.
Runtime here (single GPU, all residues of a chain): stage 1 mean 1.1 s, stage 2 mean 3.3 s for ~160-residue chains.

## Open issues / next
- Unit-weight packing energy: polar desolvation without the H-bond balance and the lj_rep floor shape the Boltzmann
  weights; a refit of term weights inside the packer (structure-only, e.g. on recovery) is possible but not done.
- Stage 2 is a mean-field approximation (no correlation between neighbours); `nb_native_frac` (WT state, neighbour's
  most probable candidate is its native side chain) is only 0.56, partly because the native and its library-bin twin
  split the probability.
- No backbone flexibility; D9 cap parameters (cap0, cap1) are exposed but features were computed with a uniform cap
  (3 kcal/mol/pair); changing them requires a rerun (Stage 1 ≈ 5 min, Stage 2 ≈ 15 min for all tables).
- torch.compile (energy.enable_compile) not used (compile overhead per shape; could give ~1.5–2×).

## Requests from the fitting agent (2026-10-26)
- **R-fit1 (test hygiene, audit item 11): the WT→WT test in `tests/test_mutate.py:43-45` is tautological.**
  `dF` is defined as `F − F[wt]`, so the WT column is 0 by construction and the test cannot fail. It therefore does
  not check D6. Please replace it with a test of the property that actually matters: (a) the candidate rotamer *set*
  at the site is identical for WT and every mutant (library-only, no native rotamer), e.g. assert that re-running
  the site scan with the WT residue relabelled as a "mutation" to its own identity reproduces F(wt) exactly, and
  (b) that `F(a)` computed from a scan seeded with a different resident residue at the site (same backbone, same
  environment) equals `F(a)` from the WT-seeded scan, within tolerance. (b) is the real symmetry claim and is what
  the Ssym diagnostic needs; if it does not hold, that difference *is* the native-adaptation bias and we should
  measure it directly rather than infer it from Ssym.
- **R-fit2 (already in progress):** capped-repulsion variants (`_capA`/`_capB`) and reverse-scoring columns. When
  they land I will select the cap setting by grouped CV on Megascale-train + val only (never on test) and report it
  in `notes/ladder_physics.md`.

## v2 additions (2026-10-26, requests from the fitting lead via PI)

### (1) D9 repulsion-cap variants + softened repulsion column
- `MutateConfig.cap0/cap1`: per-residue cap = cap0 + cap1·rigidity (pair cap = max of the two atoms' residues);
  rigidity = clip((smooth CB contact number − 8)/16, 0, 1), or a per-residue array passed as
  `SiteScan(prot, cfg, rigidity=...)` (AF-mode hook, e.g. pLDDT/100). c1 > 0 = stiffer repulsion at buried/confident
  sites; c1 < 0 is allowed.
- `EnergyParams.rep_soft` (energy.py, default 0 = unchanged): extra term `lj_rep_soft` = s·log1p(raw/s) per atom pair
  on the *uncapped* repulsive branch raw = eps((R_r/r)^6 − 1)^2 (r² floored at 0.01 Å²), 1-4 scaled like lj_rep;
  appended after hb_bb_bb. `MutateConfig.rep_soft = 1.0` (kcal/mol) turns it on for the feature runs; it is a feature
  only (the packing energy U still uses the hard-capped lj_rep).
- Feature files `data/features/phys_s{1,2}_<dataset>_cap{A,B,C}.parquet` (`55_phys_features.py --variants`;
  runtime per variant: stage 1 ≈ 4–5 min, stage 2 ≈ 15–20 min wall on GPUs 0–1):
  capA c0=1,c1=0; capB c0=3,c1=0 (= old features + soft column); capC c0=1,c1=2. Unit-weight quick look:
  | variant | Ssym lj_rep r(dir,−rev) / bias | lj_rep_soft r / bias | raw dF bias | MS-val natural ρ_prot (raw dF) |
  |---|---|---|---|---|
  | capA | 0.35 / +4.2 | 0.34 / +3.2 | +4.6 | 0.562 |
  | capB | 0.29 / +7.5 | 0.35 / +3.2 | +8.0 | 0.552 |
  | capC | 0.29 / +7.2 | 0.35 / +3.2 | +7.6 | 0.559 |
  The softer cap halves the repulsion bias but does not remove it: the asymmetry is a clash artefact of each structure
  being packed around its resident residue, not a tail artefact.

### (2) Reverse scoring on the modelled mutant environment (stage 2)
After the mean-field solve, for every site the 20 residue types are re-scored (Boltzmann over the site's rotamers) in
each of the 20 frozen mean-field environments (neighbour distributions relaxed around residue type e):
X[s, e, a] for F, per-term ⟨E_k⟩, ⟨prior⟩, site entropy. Columns:
- `rev_s2_*` = X[s, mut, wt] − X[s, mut, mut]: mut→wt scored on the modelled mutant environment;
- `fz_s2_*`  = X[s, wt, mut] − X[s, wt, wt]: wt→mut in the frozen WT-relaxed environment (same definition, so
  ½(fz − rev) is a proper symmetrised score); `sym_s2_dF` = ½(fz_s2_dF − rev_s2_dF).
Without flexible neighbours rev = −fz exactly (test). First look (capA, unit weights): Ssym r(dir, −rev) of
sym_s2_dF = 0.15 (s2_dF 0.12), bias +3.5: symmetrising on one structure does **not** recover cross-structure
antisymmetry, because the modelled mutant environment stays close to the WT crystal environment (fixed backbone,
neighbours move little) while the real mutant crystal is repacked around the mutant. MS-val natural ρ_prot (raw):
s2_dF 0.558, fz_s2_dF 0.566, sym_s2_dF 0.540. Cost: one extra [S, 20, 20, R, K, C] contraction per chunk,
< 5% of stage-2 runtime. Note the site's own backbone is not relaxed, so X→Pro stays ≈ ±35 kcal/mol in both directions.

### (3) Backbone ensemble (G3 pre-test; `stage1_ensemble`, scripts/58_backbone_ensemble.py; Megascale-val only)
Native + K = 8 backbones; each residue rigidly translated by a smooth random displacement field (GP sample, RBF length
6 Å, random Fourier features) with RMS amplitude 0.3 or 0.6 Å (peptide C–N length changes: RMS 0.09 Å, max 0.3 Å at
0.3 Å amplitude; the energy has no bonded terms). φ/ψ and rotamer candidates are kept from the native structure.
Aggregates: mean over backbones, Boltzmann over backbones (no strain energy: upper bound on relaxation).
B4 weights applied unchanged (not refitted) to the ensemble stage-1 features (n = 17,908 natural val rows; buried
small→large n = 1,402):
| amp | aggregate | raw dF ρ_prot | raw pooled | raw buried s→l | B4 ρ_prot | B4 pooled | B4 buried s→l |
|---|---|---|---|---|---|---|---|
| – | native | 0.552 | 0.554 | 0.310 | **0.7265** | 0.7027 | 0.349 |
| 0.3 | mean | 0.534 | 0.552 | 0.304 | 0.7257 | 0.7021 | 0.351 |
| 0.3 | Boltzmann | 0.550 | 0.561 | 0.315 | 0.7253 | 0.7025 | 0.358 |
| 0.6 | mean | 0.521 | 0.535 | 0.292 | 0.7233 | 0.6992 | 0.342 |
| 0.6 | Boltzmann | 0.542 | 0.556 | 0.318 | 0.7188 | 0.6969 | 0.355 |
Verdict: **no gain overall** (−0.001 to −0.008 ρ_prot); buried small→large +0.009 at most (B4, Boltzmann, 0.3 Å),
below the G3 bar. Cost: 9× stage 1 (≈ 200 s for 1,510 val sites on one shared GPU, ~15 ms per site per backbone).
Random perturbations do not find relevant relaxations; a directed relaxation (minimising the shell around the mutant)
would be the next thing to try, but this pre-test gives no reason to prioritise it.

### (4) topology.BONDS (E2 request R1)
`topology.BONDS[aa]` (backbone + IC build tree + `RING_CLOSURES`); `energy.residue_bonds` now uses it (private table
removed; identical bond sets, all energy tests pass).

Note: `physddg/fit.py` sets torch's global default dtype to float64 on import; all SiteScan entry points now run under
a float32 guard (`_float32`).

## v3 (2026-10-26): packaged predictor, speed, tests
- `physddg/predict.py`: `PhysDDG(weights=None|json, stage=1|2, device, mode='xtal'|'af').predict(structure, mutations |
  scan_all=True, sites=..., terms=True)`; CLI `scripts/physddg predict ...` (single structure, `--scan-all`, `--table`,
  `--gpus` sharding); `run_jobs` for many structures (one process per GPU, CPU preparation in a pool per GPU process).
  Engine arrays/columns come from the same functions scripts/55 uses (`site_arrays`, `engine_columns`; reassembly
  from the caches reproduces the committed feature files exactly); structure-statistics features from
  `features_simple.site_table/variant_features`; transform mirrors `52.derive`; linear model `fit.predict`.
  Check: predictions reproduce the fitting agent's stored B4 predictions exactly (max |Δ| = 0 on two Megascale-test
  proteins in-process, ≤ 1.0e-5 through run_jobs) and B5 to ≤ 6e-6. WT→WT = 0 exactly by definition (the intercept
  is not added); per-term contributions sum to ddg. Defaults: B4 (stage 1), B5 (stage 2 / `--mode af`).
- Flags: wt_mismatch, site_not_found, site_missing_atoms, missing_atoms_8A, hetatm_5A, chain_break_10A, terminal,
  low_plddt (AF mode), plus afmode.flags when available.
- Tests (tests/test_mutate.py): the tautological WT→WT array check was replaced by (a) site-native independence:
  moving the WT side chain (beyond CB) at the site leaves stage-1 F/⟨E⟩ unchanged to 1e-6 (stage 2: ≤ 1e-4, float32
  rounding of the subtract-native-neighbour bookkeeping) while moving a neighbour's side chain changes them;
  (b) predict() WT→WT = 0.000 and reproduction of the fitted model; (c) bitwise determinism across runs and across
  GPUs 0/1 (stage 1 and 2).
- Speed (scripts/59_speed_benchmark.py, results/mutate/speed_benchmark.csv): see README "Speed". Profile, 269-residue
  stage-1 scan: engine 2.57 s (≈ 89% in per-site `env_energy` dense kernels, GPU-bound), CPU 0.28 s (load, annotate,
  SASA, structure-statistics table). Tried / decided:
  * torch.compile of the energy kernel: stage 1 no gain (2.47 → 2.85 s, dynamic shapes recompile), stage 2 −15%
    (7.1 → 6.1 s), 60 s compile, ΔΔF drift 1e-4 → **not enabled** by default.
  * multi-site batching in one kernel (per-candidate neighbour lists): 2× slower (padding to the largest list).
  * fp16/bf16: not applied — r^-12 repulsion, Gaussian solvation and 1/r electrostatics overflow/lose precision in
    half precision; the kernels are elementwise (no tensor-core matmuls), so the expected gain is small.
  * CUDA graphs: shapes change per site (candidate and neighbour counts), not applicable without fixed padding,
    which costs more than the launch overhead saved.
  * Implemented wins: CPU preparation in a process pool overlapped with GPU work; one process per GPU with greedy
    balancing; single-threaded BLAS in workers (64-thread OpenBLAS per process exhausted the process limit).
    2 processes per GPU: no gain (GPU-bound).
- Issue for the fitting agent: `physddg/fit.py` sets torch's global default dtype to float64 at import; pytest
  collects tests/test_fit.py before test_energy.py runs, so `tests/test_energy.py::test_site_dependent_cap` fails in a
  full `pytest tests/` run (passes alone). mutate/predict are guarded; the fix belongs in fit.py (local dtype).
- **R-fit3 (loader, D19 final models):** the shipped models are now
  `results/ladder_physics/models/final_stage2.json` (= B6g, default), `final_stage1.json` (= B6gs1, fast mode) and
  `final_af.json` (= B6gaf, AF inputs only). Schema details in `results/ladder_physics/models/README.md`. Two things
  `physddg/predict.py` must handle:
  1. **New feature columns** `ncap_pref` and `helix_dipole`, produced by `physddg/features_extra.py`
     (`site_table` for the site descriptors `x_ncap`, `x_n1..3`, `x_c1..3`, then `candidate_features`). `ncap_pref`
     reads the PDB-derived table `physddg/data/ncap_pref.json` (built by `scripts/53_ncap_stats.py`). A loader that
     does not compute them raises a KeyError — please do not fill them with 0 silently.
  2. **Attenuated models** carry `scale_kind` (currently only `"plddt"`) and an embedded core model `scale_base`.
     Prediction is two-stage: `core = predict(scale_base, x)`, then `x["scale_plddt"] = core * flag_lowplddt`, then
     `predict(model, x)`. `flag_lowplddt = 1 - pLDDT/100` **only** for AF models whose per-structure mean B-factor is
     in [30, 100]; it is 0 for crystal structures and for design models with B = 0. Branch on `scale_kind`; never apply
     an attenuated model to crystal inputs (the term is 0 there anyway, but the intent should be explicit).
  The old `B6*.json` files (B6, B6s1, B6af, B6af_s1, B6hb, B6d15, B6j) are removed; `B6x.json` (with site_scsc) is
  kept for transparency only. `B4.json` / `B5.json` are unchanged in schema.

## v4 (2026-10-26): B6 models, AF mode (D18), final benchmark
- Weights: `physddg.predict.default_weights(stage, mode)` → `final_stage1/2/af.json` (fitting agent; currently copies
  of B6gs1 / B6g / B6gaf), else the B6g family, else B4/B5.
- B6 feature groups: `features_extra.site_descriptors(file)` → `x_`-prefixed merge on (chain, resnum) →
  `features_extra.candidate_features` (the same steps as `52.add_extra`); `struct_source` is passed through
  ('xtal' default, 'af' in AF mode, or the table's label) because the pLDDT term is only active for AF-labelled
  structures. Two-stage attenuation: `scale_<kind>` = predict(scale_base) × f (f = flag_lowplddt for 'plddt',
  1 − RSA for 'burial'), then the outer model. Crystal input (mode xtal) never attenuates (test).
- Reproduction: predict() vs the fitting agent's stored predictions (preds_*.parquet, 20:31 issue) on two proteins
  each: B6gs1 Megascale exact (0), B6g / B6gaf ≤ 1.0e-5 (Megascale and Domainome); also checked against an
  independent run of 52.predict_models on freshly merged features (≤ 1.0e-5). Tests in tests/test_mutate.py.
- AF mode (D18, `PhysDDG(mode='af')`, CLI `--mode af [--crop A:10-120] [--no-trim]`): crop = exactly a construct
  file of that range (dihedrals not carried over from the full model; boundaries are real termini) — verified:
  cropping the full v4 model to the Domainome construct range reproduces the construct-file predictions exactly;
  then `afmode.trim_mask` with the MAVISp preset (10-residue window < 50, termini < 70, packed-exempt), trimmed model
  via `afmode.trim_protein` (dihedrals preserved, no invented termini: `afmode.site_scan`), files for the
  file-based feature code via `afmode.write_trimmed_pdb`; sites inside trimmed segments are scored on the untrimmed
  model and flagged `site_trimmed`; stage 2; AF weights (attenuation); flags from `afmode.flags`; `withheld` =
  low_plddt | terminal | exposed_hydrophobic_patch | metal_site. Note: the stored B6gaf predictions were made
  without trimming, so the reproduction test uses `trim=False`.
- Speed (README table): stage-1 300-residue scan 4.8 s (was 2.9 s with B4: the new B6 site descriptors add ≈ 2 s
  of CPU per 270 residues — pure-Python loops in features_extra.site_descriptors, a candidate for vectorisation by
  its owner); Domainome 532k variants 63 s (stage 1) / 203 s (stage 2) / 207 s (AF mode) on 4 GPUs.

## v5 (2026-10-26): CPU-side speed-up
- `features_extra.site_descriptors` vectorised (batched KD-tree pair queries for contacts / H-bond partners /
  packing counts, matrix forms for charges and aromatics, sliding windows for pLDDT, exact nearest-atom shortcut for
  the void fraction so the 10 Å density count runs only on empty grid points); values identical to the loop version
  (`_site_descriptors_v1`, kept as reference; tests/test_features_extra.py on 12 structures: 4 Megascale, 4 FireProt,
  4 AF/bench, max |Δ| ≤ 1e-6, in practice 0). 269 residues: 1.47 → 0.29 s; 597 residues: 4.1 → 0.77 s.
  `site_descriptors(path, prot=...)` reuses the already loaded model (not for trimmed/cropped copies).
- Stage-1 300-residue scan 4.8 → 3.1 s; Domainome stage 1 63 → 56 s (4 GPUs).
- Top CPU hot spots now (269-residue stage-1 scan, cProfile, GPU waits excluded): (1) per-site Python/launch
  overhead in energy.env_energy/_env_terms/_core (~0.6 s tottime over 269 calls; GPU-bound kernels behind it),
  (2) features_extra.site_descriptors 0.31 s, (3) freesasa SASA 0.06 s per annotate × 3 annotates (predict load,
  features_simple.site_table, previously site_descriptors), (4) rotamer queries + NeRF launches in
  mutate.site_candidates 0.32 s (incl. GPU builder), (5) features_simple.site_table + variant_features 0.21 s and
  rotlib.rama_energy 0.08 s (20 per-aa calls). No further easy wins without changing other owners' files
  (features_simple.site_table re-loads the structure; passing the loaded model would save ≈ 0.08 s).
