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
