# AlphaFold mode (AF-mode lead) — `physddg/afmode.py`, scripts 80–85

Status 2026-10-26. Code: `physddg/afmode.py` (treatments as pure functions), `scripts/af_common.py` (shared
loading/metrics), `scripts/80_af_paired.py` (paired xtal/AF), `scripts/81_af_features.py` (engine features under a
treatment), `scripts/82_domainome_af.py` (Domainome characterisation), `scripts/83_af_treatments.py` (treated
prediction tables), `scripts/84_af_gates.py` (fitting + gate verdicts), `scripts/85_af_models_crops.py`
(AF v4/v6, crop vs full-length, global-repack ablation). Tables in `results/af_mode/`.

## 0. Predictors used here
Every treatment is measured on four predictors, because they respond to different things:

| name | what | Megascale-test ρ_prot (nat) | FireProt-test r | Domainome ρ_dom |
|---|---|---|---|---|
| `B3+` | fitted simple descriptors (baseline ladder) | 0.648 | 0.432 | 0.389 |
| `B3` | env potential + burial, no reference energies (ladder) | 0.564 | 0.287 | 0.442 |
| **`B4` / `B5`** | **the fitting agent's production physics models** (stage 1 / stage 2, `results/ladder_physics/models`), re-predicted here on the treated feature tables (reproduces their own predictions at r = 1.000) | 0.677 / 0.687 | 0.450 / 0.456 | 0.405 / 0.411 |
| `P1` | reference physics model fitted here on Megascale-train only (10 stage-1 terms + 20 reference energies, `results/af_mode/models/P1.json`); kept because it was the only physics model available when this study started | 0.597 | 0.465 | 0.293 |
| `s1dF` / `s2dF` | unit-weight stage-1 / stage-2 packing free energy, nothing fitted | 0.535 / 0.557 | 0.09 / 0.16 | 0.323 / 0.303 |

`B3+`/`B3` respond only to treatments that change descriptors (trimming, burial swap, shrinkage); `B4`/`P1`/`s1dF`
respond to all of them. In the tables below `s2` means the stage-2 (shell-repack) feature set, so for the
production models `B4__s2` is in fact B5.

## 1. Q1 — crystal vs AlphaFold on the same variant (`paired_agreement*.csv`, `paired_accuracy.csv`)
3,858 variants / 198 entities / 66 entities with ≥10 variants (FireProt 3,078, S669 627, Ssym 153; the 152 Ssym
rows whose AF model is the mutation's *end* state are excluded). This set is high-confidence by construction
(site pLDDT median ~98, 87% of sites >90), so it measures packing/side-chain differences between confident models
and crystals, not low-confidence behaviour.

**Agreement (AF vs xtal prediction of the same variant).**

| quantity | pooled r | pooled ρ | mean abs diff | per-entity median r | frac entities r ≥ 0.9 |
|---|---|---|---|---|---|
| B3+ | 0.976 | 0.973 | 0.09 kcal/mol | 0.982 | 0.91 |
| B3 / B1 / burial | 0.956–0.958 | 0.962–0.964 | 0.08 | 0.965–0.972 | 0.80–0.83 |
| **P1 (stage 1)** | **0.919** | 0.929 | 0.14 | **0.950** | 0.70 |
| **P1 (stage 2, shell repack)** | **0.927** | 0.932 | 0.14 | 0.938 | 0.64 |
| s1 ΔF (unit weights) | 0.823 | 0.826 | 3.2 | 0.837 | 0.42 |
| s2 ΔF (unit weights) | 0.728 | 0.862 | 2.7 | 0.872 | 0.47 |
| per-term: solv_apolar 0.978, lj_atr 0.980, solv_polar 0.971, ΔTS 0.909, elec 0.854, **lj_rep 0.840**, hb_sc_bb 0.820, **hb_sc_sc 0.721** | | | | | |

The disagreement is concentrated in the terms that depend on exact side-chain placement: **repulsion and
side-chain H-bonds**. A fitted model is far more robust than the raw energy because the fit gives `lj_rep` a weight
of 0.014 and `hb_sc_sc` 0.146 — i.e. the fitting already does most of the "AF-proofing" that treatments are
supposed to do. |AF − xtal| grows with local backbone RMSD (ρ = 0.23–0.27) and, for the physics terms, with
**burial** (ρ(|Δ s1ΔF|, burial) = +0.29, lj_rep +0.34): cores differ, surfaces agree.

**Accuracy against experiment on identical rows** (paired bootstrap over entities, Δ = AF − xtal):

| quantity | subset | ρ(AF) | ρ(xtal) | Δ [95% CI] |
|---|---|---|---|---|
| B3+ | all | 0.546 | 0.543 | +0.003 [−0.006, +0.015] |
| P1 | all | 0.542 | 0.526 | +0.016 [−0.002, +0.030] |
| s1 ΔF | all | 0.436 | 0.387 | **+0.048 [+0.017, +0.088]** |
| P1 | local RMSD 0.6–1.0 Å | 0.559 | 0.492 | **+0.066 [+0.015, +0.125]** |
| s1 ΔF | local RMSD 0.6–1.0 Å | 0.463 | 0.173 | **+0.290 [+0.144, +0.454]** |
| s1 ΔF | local RMSD > 1.0 Å | 0.489 | 0.245 | **+0.244 [+0.105, +0.341]** |
| P1 | site pLDDT 70–90 | 0.567 | 0.495 | +0.071 [−0.008, +0.146] |

**The AF model is never worse than the crystal here, and where the two structures disagree the AF model is
clearly better.** This inverts the usual framing: for a rigid, clash-sensitive energy the risk is the *crystal*
(strain, alternate conformers, removed ligands/waters, resolution), not the prediction. It also means that on
this set there is nothing for an AF-specific treatment to repair — treatments must be justified on Domainome.

## 2. Q2 — Domainome as the low-pLDDT stress test (`dom_*.csv`)
532,501 variants / 518 domains with an AF construct crop; scored against −fitness (per-domain Spearman, ρ_dom).
Site pLDDT: 93.7% ≥ 70, 4.7% in 50–70, 1.3% < 50.

**By site pLDDT band (per-domain median ρ; domains with ≥10 variants in the band):**

| band | n variants | B3 | B3+ | P1 | s1 ΔF | FoldX | −RSA |
|---|---|---|---|---|---|---|---|
| > 90 | 272,209 | 0.466 | 0.429 | — | 0.381 | 0.488 | 0.428 |
| 70–90 | 228,374 | 0.353 | 0.280 | — | 0.192 | 0.321 | 0.272 |
| 50–70 | 25,146 | 0.124 | −0.073 | — | −0.119 | 0.049 | −0.008 |
| < 50 | 6,772 | 0.114 | −0.118 | — | −0.194 | −0.020 | 0.005 |
| all | 532,501 | 0.442 | 0.389 | 0.293 | 0.323 | 0.430 | 0.389 |

By **construct** mean pLDDT the effect is much weaker (<70: B3 0.405, FoldX 0.323; >90: 0.447 / 0.464), i.e. it is
a per-site, not a per-model, effect.

**The low-pLDDT failure is mostly a buried/exposed effect, not a model-error effect** (`dom_plddt_vs_burial.csv`,
pooled within-domain Spearman):

| burial × pLDDT | 50–70 | 70–90 | > 90 |
|---|---|---|---|
| buried (RSA < 0.15): s1ΔF / B3+ / FoldX | 0.19 / 0.19 / 0.27 | 0.29 / 0.24 / 0.38 | 0.29 / 0.30 / 0.39 |
| partial (0.15–0.35) | 0.11 / 0.02 / 0.14 | 0.17 / 0.16 / 0.26 | 0.22 / 0.18 / 0.32 |
| exposed (> 0.35) | −0.12 / −0.03 / 0.08 | 0.01 / 0.12 / 0.22 | 0.03 / 0.11 / 0.25 |

At **buried** sites, dropping from pLDDT > 90 to 50–70 costs only ~0.10 and nothing goes negative. At **exposed**
sites the physics model is near zero *at every confidence level*. Low-pLDDT sites are almost all exposed, so the
raw "pLDDT band" table above mostly re-measures the exposed-surface failure (which the ladder already traced to
Megascale-fitted reference energies not transferring to an abundance assay — `notes/baseline_ladder.md` §4; B3,
the one model with no Δref, is the only one that stays positive below pLDDT 70). **Any pLDDT attenuation fitted
on Domainome is therefore at risk of being a burial correction in disguise.**

**Failure modes** (`dom_failure_modes.csv`, per-domain median ρ, flag vs rest; flags from `afmode.flags`):

| flag | n variants | B3+ | s1 ΔF | FoldX | rest (B3+ / FoldX) |
|---|---|---|---|---|---|
| site pLDDT < 70 | 31,918 | −0.07 | −0.13 | 0.04 | 0.40 / 0.45 |
| site pLDDT < 50 | 6,772 | −0.12 | −0.19 | −0.02 | 0.39 / 0.43 |
| low-pLDDT (<50) residue within 8 Å | 15,353 | −0.02 | −0.12 | 0.05 | 0.39 / 0.44 |
| exposed hydrophobic patch (missing partner) | 5,875 | 0.02 | −0.01 | 0.10 | 0.39 / 0.43 |
| metal site (≥3 Cys/His within 8 Å) | 37,689 | 0.38 | 0.22 | **0.14** | 0.37 / 0.44 |
| within 3 residues of a construct terminus | 52,724 | **0.11** | 0.04 | 0.15 | 0.40 / 0.45 |

Two of these are large: **construct termini** (10% of all rows, ρ collapses to 0.11–0.15 for every predictor) and
**metal sites** (7% of rows; FoldX collapses from 0.44 to 0.14, the physics model from 0.31 to 0.22, while −RSA
*improves* to 0.45 — burial still works, the energy does not). Low pLDDT and missing-partner patches are small in
row count (6% and 1%) but have essentially zero signal.

## 3. Q2 continued — model version and construct (`dom_v4_v6.csv`, `dom_crop_full.csv`)
**AF v4 vs v6** (509,218 variants, 497 constructs with identical sequence): P1 predictions agree at r = 0.979
pooled / 0.988 per domain (97% of domains r ≥ 0.9); accuracy is unchanged (ρ_dom 0.290 both). Agreement falls
only for constructs with mean pLDDT < 70 (r = 0.887, 43% of domains ≥ 0.9) — and even there accuracy is the same
(0.232 vs 0.230). **AF release version is not a decision variable** at our level of accuracy.

**Construct crop vs full-length model** (same sites, 532,501 variants). Cropping to the assayed construct is
not neutral for burial: mean RSA 0.369 (crop) vs 0.326 (full), mean contact number 14.2 vs 15.5 — the full-length
model buries the construct's surface under the rest of the protein, which is *not* what the aPCA assay measures.

| | agreement r (per-domain median) | ρ_dom crop | ρ_dom full |
|---|---|---|---|
| P1 | 0.967 (0.982) | **0.293** | 0.279 |
| s1 ΔF | 0.940 (0.983) | **0.319** | 0.311 |
| RSA | 0.909 (0.926) | 0.389 | 0.356 |
| contact number | 0.910 (0.943) | 0.354 | 0.318 |

At pLDDT < 50 sites the crop/full burial disagreement is extreme (RSA r = 0.596, CN 5.7 vs 7.8). **Crop to the
assayed construct** — it is better in every pLDDT band, as the data-AF report already suspected.

## 4. Q3/Q4 — treatments and gate G5 (`treat_*.csv`, `shrink_control.csv`, `gates.csv`)
AF parameters were chosen on abundance_dms + a random half of the Domainome Pfam clusters (seed 0,
`af_common.split_halves`) and are reported on the other half (280 domains, 279,134 variants). Trimming deletes
some measured sites, so every delta below is computed on **identical rows** (the rows where all treatments of
that predictor are defined); the rows a treatment removes are reported separately as its abstention rate and
scored in §5. Δ = per-domain median Spearman vs no treatment.

**Domainome, report half, identical rows** (`treat_common_rows_report.csv`; fit half in `treat_common_rows_fit.csv`):

| treatment | rows dropped | B3+ | B3 | **B4** | P1 | s1 ΔF |
|---|---|---|---|---|---|---|
| (a) `trim50` windowed pLDDT < 50, packed-exempt | 0.2% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| (a) `trim70` every residue pLDDT < 70 | 2.8% | +0.009 | −0.002 | +0.003 | 0.000 | 0.000 |
| (a) `mavisp` < 50 windows + < 70 termini | 0.9% | +0.001 | −0.001 | +0.001 | 0.000 | +0.001 |
| (a') 50–70 band down-weight w = 0.25/0.5/0.75 | 2.8% | – | – | – | 0.000 | ≤0.001 |
| (b) pLDDT-modulated cap 0.5→3.0 | 0 | – | – | −0.000 | −0.002 | 0.000 |
| (b) softer cap 0.25→1.5 | 0 | – | – | −0.004 | **−0.028** | −0.003 |
| (c) shell repack (stage 2; B4→B5) | 0 | – | – | **+0.003** | −0.014 | −0.013 |
| (d) pLDDT shrinkage (fitted logistic) | 0 | +0.006 | +0.004 | +0.016 | **+0.022** | +0.006 |
| (d-control) **same logistic in burial** | 0 | +0.014 | **+0.022** | +0.019 | **+0.073** | **+0.023** |
| (e) backbone-only burial in 70–90 (`burswap`) | 0 | +0.004 | −0.009 | – | – | – |
| (e) backbone-only burial everywhere (`burcn`) | 0 | −0.009 | −0.035 | – | – | – |

On the fit half the trimming effects are slightly larger and in the same direction (B4: trim70 +0.007,
mavisp +0.007, stage 2 +0.009), so nothing here is a split artefact.

**Megascale-test (natural domains, identical rows, 21,684 variants** — read only as the "does it hurt" side of
G5; no AF parameter was chosen on it**)**: `trim70` +0.016 (B3+) / +0.013 (B4) / +0.023 (s1ΔF),
`mavisp` +0.014 / **+0.017** / +0.022, stage 2 +0.007 (B4) / +0.027 (s1ΔF), `cap_plddt` 0.000,
`cap_soft` −0.004 (B4) / −0.022 (P1). Trimming is the one treatment with a visible positive effect anywhere, and
it is on Megascale — whose structures are AF models with contact-trimmed termini, so the MAVISp rule is removing
residual low-confidence tails from an already-trimmed model.

**Gate G5, part 1 — paired xtal/AF per-site agreement r ≥ 0.9 after uniform repacking.**
Per-entity median r: B3+ 0.982, B3 0.965, **P1 0.950 (stage 1) / 0.938 (stage 2)**, raw s1 ΔF 0.837.
Pooled r: B3+ 0.976, P1 0.919 / 0.927, raw ΔF 0.823. **PASS for every fitted model**, and stage-2 uniform
repacking is what rescues the hard cases (local RMSD > 1 Å: pooled r 0.684 → 0.746, fraction of entities with
r ≥ 0.9 0.375 → 0.625). **FAIL for the unit-weight energy** (0.837 per-entity median, 42% of entities ≥ 0.9):
the gate is in practice a statement about how much weight the fit puts on `lj_rep` and `hb_sc_sc` (P1 gives them
0.014 and 0.146), not about the AF model.

**(c) ablation: global repacking** (`repack_ablation.csv`; `afmode.repack_all` = 2 passes of greedy
lowest-energy library rotamers for every side chain, applied identically to both members of each pair; the 52
entities with ≥ 10 variants, 3,288 identical rows):

| protocol | AF↔xtal pooled r | per-entity median r | frac ≥ 0.9 | ρ(AF, ΔΔG) | ρ(xtal, ΔΔG) |
|---|---|---|---|---|---|
| keep AF/crystal side chains (default) | **0.926** | **0.950** | **0.709** | 0.573 | 0.553 |
| global repack of every side chain | 0.921 | 0.940 | 0.636 | 0.571 | 0.555 |

Global repacking neither improves agreement nor accuracy, and makes the two inputs slightly *less* comparable
(small backbone differences get amplified into different rotamer choices). This reproduces Vangaru &
Bhattacharya 2025 in our own energy: **keep the AF side chains, repack only the site and its shell.**

**Gate G5, part 2 — pLDDT attenuation ships only if Domainome ≥ +0.02 and Megascale-test ≥ −0.005** (`gates.csv`):

| treatment | Domainome Δ (report half) | Megascale-test Δ | verdict |
|---|---|---|---|
| trim50 / mavisp / trim70 | 0.000 … +0.009 | +0.013 … +0.017 | below gate (neutral, never harmful) |
| pLDDT cap | −0.002 … 0.000 | 0.000 | below gate |
| softer cap | −0.003 … −0.028 | −0.004 … −0.022 | harmful |
| shell repack (B5) | +0.003 (B4), −0.013 (unfitted) | +0.007 | below gate, keep for G5 part 1 |
| backbone-only burial (70–90) | +0.004 (B3+), −0.009 (B3) | – | below gate |
| **pLDDT shrinkage** | +0.006 / +0.016 / **+0.022 (P1)** | +0.012 … +0.037 | passes the literal gate for P1 only |
| **burial shrinkage (control)** | +0.014 / +0.019 / **+0.073 (P1)** | – | 3× the pLDDT gain |

Taken literally the fitted shrinkage passes for P1 (+0.022 Domainome, +0.037 Megascale) and fails for every
other predictor (+0.004…+0.016). **But the burial-matched control — the identical two-parameter logistic in
100·(1−RSA) instead of pLDDT, fitted on the same held-in data — gives +0.073 for P1 and +0.014…+0.023 for the
others.** The "pLDDT attenuation" is a weak proxy for a burial-dependent calibration, which belongs in the
energy model (a burial-scaled link, D8), not in the AF mode. **Recommendation: do not ship pLDDT shrinkage;
hand the burial-dependent link to the fitting agent instead.** This is the same confound as §2: on Domainome,
pLDDT is mostly a burial label.

## 5. (f) Abstain / flag rules (`abstain.csv`, `dom_failure_modes.csv`)
Dropping flagged rows from the Domainome report (per-domain median ρ over all 518 domains):

| rule | rows kept | **B4** | B3+ | B3 | P1 | s1 ΔF |
|---|---|---|---|---|---|---|
| none | 100% | 0.405 | 0.389 | 0.442 | 0.293 | 0.319 |
| site pLDDT < 70 | 94.0% | 0.421 | 0.400 | – | 0.309 | 0.333 |
| within 3 residues of a terminus | 90.1% | 0.431 | 0.403 | – | 0.317 | 0.334 |
| metal site (≥3 Cys/His within 8 Å) | 92.9% | 0.389 | 0.366 | 0.417 | 0.291 | 0.308 |
| pLDDT < 70 **or** terminal | 86.7% | 0.437 | 0.407 | 0.451 | 0.323 | 0.342 |
| pLDDT < 70 or terminal or exposed hydrophobic patch | 85.9% | **0.443** | **0.409** | **0.452** | **0.328** | **0.347** |

Flagging 14% of variants buys **+0.038 (B4)**, +0.020 (B3+) and +0.035 (P1/s1ΔF) on what remains — more than
every computation-changing treatment put together. Dropping metal sites *lowers* ρ: the flag fires on 7% of rows
that are still partly predictable from burial (−RSA ρ = 0.45 there), so metal sites should be **reported with a
flag, not abstained on** (and they are the one place where FoldX collapses, 0.44 → 0.14, while the physics
models keep 0.22–0.39).

## 6. Recommended default "AF mode"
```python
from physddg.afmode import AFMode, TrimConfig, FlagConfig
AF_DEFAULT = AFMode(
    name="af_default",
    trim=TrimConfig(below=50.0, window=10, termini_below=70.0, packed_exempt=True, packed_contacts=3.0),
    cap=None,            # pLDDT-modulated repulsion cap: no effect (+0.000); softer caps hurt (-0.026)
    stage=2,             # site + 6 A shell repack, identical for WT and mutant (keeps AF side chains elsewhere)
    shrink_params=None,  # pLDDT shrinkage: fails the burial-matched control, see section 4
    burial_band=None,    # backbone-only burial: +0.002, not worth the extra code path
)
```
plus, outside the energy: **crop to the assayed construct / PAE domain**, **never build mutant structures with
AF**, and **always emit the flags** (`afmode.flags`: site pLDDT, low-confidence neighbourhood, terminal, exposed
hydrophobic patch, metal site), with the recommendation to withhold predictions for pLDDT < 70 **or** terminal
**or** exposed-hydrophobic-patch sites unless the caller asks for them.

Reasoning in one line each:
* **Trim (MAVISp rule)** is the only computation-changing treatment that is positive everywhere it is measured:
  0.000…+0.009 on Domainome (identical rows), +0.014…+0.022 on Megascale-test, at the cost of 0.9% of sites
  (the site itself is trimmed), which the flag system already covers. It is kept because it is free, standard
  practice and never harmful — not because it is proven: every Domainome CI touches zero.
* **Stage 2 (site + shell repack)** is kept because it is what makes the paired xtal/AF agreement hold up in the
  hard cases (local RMSD > 1 Å: fraction of entities with r ≥ 0.9 0.375 → 0.625) and because for the production
  model it is positive on both benchmarks (Domainome +0.003, Megascale-test +0.007). It is negative (−0.013) for
  predictors with unit or self-fitted weights, i.e. it needs weights fitted on the stage-2 features (B5, not B4).
* **No pLDDT cap, no pLDDT shrinkage, no burial swap**: measured and below gate, or beaten by a burial control.
* **Flags**, not corrections, for the two large failure modes (termini, metal sites) and for missing partners.
* **For the fitting agent**: the one real gain this study found is a **burial-dependent link/attenuation**
  (+0.073 ρ_dom for P1, +0.019 for B4, +0.022 for B3 on Domainome, fitted on abundance + half of Domainome).
  It is not an AF treatment — it is a calibration of the model and belongs in D8.

## 7. Caveats
* `B4`/`B5` are read from `results/ladder_physics/models/` (snapshot of 2026-10-26 17:17/17:24, commit 9743124).
  They were refitted once while this study ran; every number above was recomputed against the later fit and
  changed by ≤ 0.001, so the conclusions are not sensitive to that version.
* `afmode.plddt()` reads the CA B-factor. It is only meaningful for predicted models — never apply the pLDDT
  treatments to a crystal structure (the project's xtal tables carry real B-factors there).
* Trimming inside the engine (`trim_protein`) preserves the kept residues' φ/ψ from the full model and keeps the
  original chain-start flags, so deleting a neighbour removes its contacts without inventing a charged terminus.
  The descriptor pipeline is fed trimmed PDB files instead (`afmode.write_trimmed_pdb`), where φ/ψ at a new
  chain end become undefined for the ≤ 2 residues next to a cut — a small inconsistency between the two paths.
* A window-mean off-by-one in `afmode.window_mean` was found and fixed while writing the tests; every windowed
  trim variant (trim50, mavisp) and all tables above were recomputed afterwards.
* The 50–70 "down-weighting" is a linear blend of the features computed with and without those residues; exact
  for additive pair energies, approximate for the Boltzmann free energy. It made no difference for any w.
* `repack_all` is iterated greedy repacking (2 passes, library rotamers, native environment), not a true global
  packer; it is an ablation, not a protocol.
* Domainome has no true ΔΔG, and its exposed-site signal disagrees with Megascale's (see §2). Every Domainome
  delta here is a rank statistic against an abundance proxy.
* Treatment deltas are per-domain Spearman against an abundance proxy on 280 held-out domains; the cluster
  bootstrap CIs in `treat_domainome_boot.csv` are ±0.01–0.02 wide, so differences below ~0.01 are not resolvable
  here. Re-running everything for a new model costs ~5 min (`scripts/83 --stage preds`, then `scripts/84`,
  `scripts/86`); recomputing the engine features for a new treatment costs ~4 min per dataset on 2 GPUs.
