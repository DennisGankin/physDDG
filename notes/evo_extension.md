# Optional evolution-aware mode 'PhysDDG+evo' (2026-10-26)

**Scope.** This is a separate, optional mode. The headline PhysDDG model (B5) stays structure-only. Here we measure
how much non-DL sequence-profile (conservation) information adds when the B5 terms are kept fixed in form. The
profile is a **different information source**: an hhblits search of UniClust30 (2018_08). No learned model and no
embedding is used anywhere, only counting statistics of the alignment plus fitted linear weights (`physddg/fit.py`).

Scripts: `10a_evo_profiles.py` (+ worker `10aw_evo_worker.py`, run in `.venv_ddgun`), `10b_evo_features.py`,
`10c_evo_eval.py` (select / final / eval), `10d_evo_where.py`.
Outputs: `results/evo_extension/` (selection, models, metrics, paired, strata, where_site_within, per_protein_megascale,
msa_coverage*). Feature tables are `data/features/evo_<dataset>.parquet` (git-ignored, regenerable in about 10 s
from the MSA cache). Logs are in `logs/10*_*.log`.

## 1. Profiles (`10a`)
- **Search.** The search is the same as DDGun's: hhblits v3.3.0 against UniClust30 2018_08 with `-n 2 -e 1e-6 -cpu 4`.
  The query is the chain sequence of the benchmark structure, read with DDGun's own parser. Results are cached by
  sha1(sequence) in `data/work/ddgun/hhblits_cache/`. The MSA is query-anchored with insertions removed.
  hhblits filters its result MSA with its defaults (`-id 90 -diff 1000`).
- **Units.** 1,492 (structure, chain) units, which are 1,422 distinct sequences. 551 MSAs came from the DDGun cache
  and 871 were new searches: 12 concurrent jobs, about 42 min wall at machine load 50–85. All units have an MSA.
- **Coverage.** N_eff counts sequences after 80%-identity reweighting and includes the query. Table:
  `results/evo_extension/msa_coverage_summary.csv`.

| set | units | median N_eff | N_eff ≥ 10 | N_eff ≥ 100 | N_eff ≥ 1000 |
|---|---|---|---|---|---|
| Megascale natural | 189 | 795 | 0.89 | 0.71 | 0.39 |
| Megascale designed | 109 | 1.0 | **0.00** | 0 | 0 |
| FireProt | 99 | 1115 | 0.96 | 0.94 | 0.66 |
| S669 / S461 | 94 / 48 | 1072 / 667 | 0.95 / 0.92 | 0.85 / 0.77 | 0.54 / 0.40 |
| Ssym (incl. mutant structures) | 357 | 1035 | 1.00 | 1.00 | 0.71 |
| Domainome constructs | 518 | 968 | 0.98 | 0.81 | 0.42 |
| abundance_dms | 7 | 1928 | 1.00 | 1.00 | 1.00 |

- **Designed domains.** All 109 designed Megascale domains have no hit at all: the query is the only sequence.
- **Shallow natural units (N_eff < 10): 44** (listed in `msa_coverage.csv`).
  - Megascale (21): 2LHC, 2MCH, 2MCK, 2MH8, 5KPH, 5UP5, 5UYO, 7BPM and 6OBK have no homologue. Several of these are
    de novo or engineered mini-proteins that Megascale labels "natural".
  - FireProt / S669 / S461 / S2648: 1AEP, 1SSO, 1AZP, 3MON, 4HE7 and others. Protein G B1 (1PGA) has N_eff 4.7 at
    `-e 1e-6`.
  - Domainome (10): all very short constructs (19–44 aa).
- **Full-length Domainome domains.** Not searched. 98% of constructs already have N_eff ≥ 10, so a longer query
  cannot change much.

## 2. Features (`10b`)
Keys are (protein_id, chain, pdb_resnum, wt, mut) + structure, with 100% row coverage on every benchmark. Sequence weights
use 80% identity, measured over the shorter non-gap length. The profile uses PSI-BLAST pseudocounts:
p = (α f + β g)/(α+β), with g = BLOSUM62-conditional, α = N_eff,site − 1 and β = 10. With no homologues the profile
therefore reduces to BLOSUM62 for the wild type. **Every `evo_` column is oriented so that higher = more tolerated**,
so in a ddG model with positive = destabilising its weight must be ≤ 0. The fit enforces this.

| column | definition |
|---|---|
| `evo_lor` | log p(mut) − log p(wt): site-independent log-ratio, nats |
| `evo_pssm` | PSSM log-odds difference. Equals `evo_lor` + a (wt,mut) constant that the reference energies absorb |
| `evo_entropy`, `evo_naa` | site Shannon entropy of the weighted frequencies; exp(entropy) = effective number of amino acids |
| `evo_nonwt`, `evo_gap` | 1 − f(wt); weighted gap fraction |
| `evo_hyd_lor`, `evo_vol_lor` | class log-ratios: apolar/polar/charged; 5 volume classes |
| `evo_hyd_z`, `evo_vol_z` | −\|KD(mut) − ⟨KD⟩_site\|/sd; same for side-chain volume |
| `evo_blprof` | DDGun-style Σ_a f_a [B62(a,mut) − B62(a,wt)] |
| `evo_blosum` | B62(wt,mut) |
| `evo_neff`, `evo_neff_site`, `evo_nseq`, `evo_gate` | alignment depth; a-priori gate clip(log10 N_eff,site / 2, 0, 1) |

**Sign check** (Megascale-test natural, per-protein median ρ against −ddg, raw features): lor 0.39, entropy 0.45,
blprof 0.41, nonwt 0.37, hyd_lor 0.26, vol_z 0.24. All are positive. On designed domains only the BLOSUM part
carries signal (0.15–0.20).

## 3. Leakage and design
- **UniClust30 is not a held-out source.** UniClust30 2018 contains homologues of every natural benchmark protein,
  often the protein itself. This is inherent to any evolutionary method, including DDGun, GEMME and EVE. It is not
  label leakage, because no ΔΔG enters the profile. But "strict" here means only *no Megascale-train homologue in
  the fit*, not *no evolutionary information about the test protein*.
- **What is fitted.** Only one evo weight (plus a refit of the B5 / B3+ weights) on Megascale-train. Selection used
  train grouped CV and Megascale-val only. Every test set was scored once, after `selection.json` was fixed.
- **Designed domains.** These have no profile. Two treatments were compared:
  - The a-priori N_eff gate.
  - No gate: the pseudocount scheme already gates, because at N_eff = 1 the term is pure BLOSUM62.

  Selection preferred no gate (val+CV score 0.089 vs 0.082). Hard (N_eff ≥ 10) and exponential gates scored below no gate (0.071–0.083).
  On test, the two forms are indistinguishable (B5evo-apriori in `metrics.csv`). Natural and designed domains are
  reported separately below.

**Selection** (`selection.csv`). Every feature set raised B5 a lot on train CV (natural per-protein ρ 0.673 → 0.72,
5/5 folds) and on val (0.733 → 0.765). The sets differed by less than 0.003, so the pre-set rule picked the
1-weight set:
- **B5evo = B5 + `evo_lor`** (ungated).
- Weight −0.113 ± 0.007 kcal/mol per nat (50 cluster bootstraps), i.e. ≈ 0.19 kT per nat of log-ratio.
- Adding entropy, gap, class terms or blprof added at most +0.002 on val and on CV.
- B3+evo uses the same term (weight −0.143). Evo-only = reference energies + `evo_lor` (weight −0.23).

## 4. Results (`metrics.csv`, `paired.csv`; 95% cluster-paired bootstrap, 500 reps; Domainome 200)
Per-protein median Spearman unless stated. B5 and B3+ were refit here and reproduce the published numbers (B5 0.685).

| set | B5 | **B5evo** | Δ vs B5 [95% CI] | B3+ → B3+evo | Evo-only | FoldX | lit. DL |
|---|---|---|---|---|---|---|---|
| Megascale-test all 28: per-protein ρ | 0.685 | **0.726** | +0.040 [−0.005, +0.071] | 0.649 → 0.705 | 0.582 | 0.633 | ThermoMPNN 0.77, SPURS 0.83 |
| Megascale-test all 28: pooled ρ / Pearson | 0.642 / 0.686 | 0.658 / 0.701 | +0.017 [−0.003, +0.033] / +0.014 | 0.603 → 0.627 | 0.502 / 0.532 | 0.569 / 0.447 | ThermoMPNN 0.725 / 0.754 |
| natural 21 | 0.687 | 0.739 | +0.052 [−0.016, +0.082] | 0.648 → 0.735 | 0.602 | 0.629 | |
| natural-strict 14 | 0.697 | 0.691 | −0.006 [−0.037, +0.080]; mean per-protein Δ +0.014, 9/14 up | 0.649 → 0.641 | 0.502 | 0.640 | |
| designed 7 | 0.684 | 0.700 | +0.016 [−0.000, +0.016] | 0.664 → 0.680 | 0.567 | 0.637 | |
| FireProt-test (347): Spearman / Pearson | 0.479 / 0.456 | 0.501 / 0.492 | +0.022 [−0.015, +0.052] / +0.036 [−0.023, +0.068] | 0.455 → 0.490 | 0.389 | 0.449 / 0.411 | |
| S669 strict (470): Spearman / Pearson | 0.320 / 0.315 | 0.357 / 0.337 | **+0.037 [+0.004, +0.075]** / +0.022 | 0.338 → 0.375 | 0.331 | — | |
| S461 strict (285): Spearman / Pearson | 0.560 / 0.556 | 0.565 / 0.566 | +0.005 [−0.037, +0.058] | 0.569 → 0.563 | 0.425 | — | |
| Ssym direct: Spearman | 0.692 | 0.719 | +0.027 [−0.007, +0.064] | 0.575 → 0.617 | 0.422 | — | |
| **Domainome per-domain ρ (518)** | 0.411 | **0.486** | **+0.074 [+0.059, +0.084]** | 0.389 → 0.472 | 0.379 | 0.430 (493) | ThermoMPNN 0.50, SPURS 0.54 |
| Domainome strict (366) | 0.381 | 0.453 | +0.072 [+0.063, +0.092] | 0.363 → 0.451 | 0.373 | 0.385 | |
| Domainome pLDDT ≥ 90 / 70–90 / 50–70 / < 50 | 0.466 / 0.306 / −0.018 / −0.036 | 0.547 / 0.378 / 0.068 / 0.061 | +0.08 / +0.07 / +0.09 / +0.10 (all CIs > 0) | | | 0.488 / 0.321 / 0.049 / −0.020 | |
| abundance_dms (7) | 0.469 | 0.509 | +0.040 [+0.026, +0.065] | 0.446 → 0.502 | 0.358 | — | |

- **Against FoldX.**
  - Megascale-test all: per-protein +0.093 [+0.042, +0.141]; pooled +0.090; Pearson +0.253.
  - Domainome: +0.058 [+0.029, +0.088]. Structure-only B5 is −0.018 n.s. against FoldX there.
  - FireProt: per-protein −0.047 (n.s., FoldX 0.634); pooled +0.053 (n.s.).
- **Against DDGun3D.** DDGun3D uses the same MSAs. B5evo beats it everywhere: Megascale +0.22 per-protein,
  Domainome subset +0.19.

**Where it helps** (`10d`, `where_site_within.csv`; per-protein medians on Megascale-test, B5 → B5evo):

| stratum | site-level ρ | within-site ρ |
|---|---|---|
| all | 0.759 → 0.795 | 0.577 → 0.630 |
| natural | 0.776 → 0.817 | 0.559 → 0.611 |
| designed (no profile) | 0.753 → 0.754 | 0.665 → 0.677 (BLOSUM part only) |
| buried (RSA < 0.25) | 0.680 → 0.703 | 0.683 → 0.719 |
| exposed | 0.590 → 0.616 | 0.501 → 0.554 |
| natural & exposed | 0.595 → 0.652 | 0.491 → 0.542 |
| natural & buried | 0.659 → 0.682 | 0.653 → 0.694 |

- **Both halves gain about equally on natural domains** (+0.04 site, +0.05 within). The largest gain is on exposed
  natural sites, where the structure terms are weakest and conservation carries the information.
- **Domainome.** The gain is uniform over burial and pLDDT. It is largest in relative terms at pLDDT < 70, where
  B5 alone is near zero (−0.02 → +0.06).
- **Designed domains gain nothing beyond BLOSUM62.**
- **Per test protein** (`per_protein_megascale.csv`): 21 of 28 up.
  - The 7 natural test proteins that have a Megascale-train homologue gain +0.062 on average (7/7 up).
  - The 14 strict natural proteins gain +0.014 (9/14 up; median Δ −0.006 because of the n = 14 median).

  A single evo weight cannot memorise families. The difference is either noise or a sign that the well-sampled
  families (SH3, WW-like) also have the most informative profiles. **The natural-strict Megascale number shows no
  significant gain, and it should be quoted alongside the headline.**

## 5. How much of the DL gap closes
- **Megascale-test all, per-protein median.** ThermoMPNN is 0.77 (SPURS paper, same 28 proteins). B5 is 0.685 and
  B5evo 0.726, so the profile closes about 48% of the 0.085 gap.
- **Megascale pooled metrics.** On pooled ρ / Pearson (ThermoMPNN 0.725 / 0.754) it closes only about 20%
  (0.642 → 0.658; 0.686 → 0.701).
- **Domainome.** ThermoMPNN is 0.50 (paper median; zinc fingers included), so B5evo at 0.486 closes about 85% of
  the 0.089 gap. It does not reach SPURS (0.54) or ESM-MSR (0.57).
- **Not comparable.** The S669 / S461 / FireProt literature numbers use other subsets.
- **What the gain is.** Most of it is the information that the DL models also exploit, namely evolutionary
  substitution preferences. It is **not** a gain in structural physics.

## Recommendation
- **Offer 'PhysDDG+evo' as an optional mode.** It is B5 + 1 weight on the site-independent hhblits log-ratio
  (`evo_lor`, PSI-BLAST pseudocounts, no gate needed). It requires hhblits + UniClust30 (93 GB) and 30–60 s per
  new sequence.
- **Report it next to the structure-only headline, never instead of it**, with the leakage statement from §3.
- **Large, significant gains:** Domainome (+0.074) and abundance DMS (+0.040).
- **Positive but not significant on Megascale-test:** per-protein +0.040, which does not survive the strict subset.
- **Small:** FireProt, S461 and Ssym. S669 strict is significant (+0.037).
- **No benefit for de novo designs.** Richer profile features (entropy, gaps, class conservation, DDGun blprof) add
  nothing over the single log-ratio.
