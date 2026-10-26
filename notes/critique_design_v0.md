# Independent critique of DESIGN.md (PhysDDG v0)

Reviewer: external, protein biophysics / stability-prediction methods. 2026-10-26. Scope: PROMPT.md + DESIGN.md only. Citations in §7.

**Overall.** The skeleton is right: identical WT/mutant protocol, decomposed terms, few interpretable weights, staged delivery. Beating FoldX is a low bar and will very likely be cleared. Two things decide whether this approaches DL: (a) whether a knowledge-based, environment-dependent amino-acid preference term is in the model — it is currently absent and is the largest single omission; (b) whether the fitting protocol respects how Megascale ΔG is actually produced (censoring at the assay ceiling; a PSSM unfolded-state correction already baked into the labels). The repacking machinery is the most expensive component and the least likely to pay for itself; it needs an explicit kill gate (§6).

---

## 1. Where the signal lives, and term triage

**Megascale (Tsuboyama 2023).** 776,298 ΔG values, 331 natural + 148 designed domains, 40–72 aa [1]. These are *tiny* domains with high surface:volume ratio — typically only ~25–35% of residues are substantially buried. Consequences:
- Most rows are surface/near-surface mutations with |ΔΔG| < 0.5 kcal/mol, near the noise floor. Per-domain Spearman is dominated by (i) separating buried from exposed sites and (ii) getting the buried-site hydrophobicity/volume response right. **Burial × residue identity is the workhorse.**
- 148/479 domains are de novo designs: idealised, hyperstable, over-packed, few loops. Any packing-sensitive weight fitted on the pooled set is partly fitted to design idealisation. **Report natural-only metrics as the primary number.**
- 121 destabilised-background WTs exist [1]; these are where ΔΔG is measurable in both directions and carry most information about the stabilising tail.

**FireProt / S669.** A few thousand ProTherm-derived rows, heterogeneous pH/T/buffer, heavy Ala/Gly bias, strong mutual overlap. The ceiling is set by experimental and curation noise. Orientation: FoldX Pearson ≈ 0.22 on S669 direct, Rosetta and RaSP ≈ 0.39, best published 0.5–0.6 [2,3]. A structure-independent "average ΔΔG for this WT→mutant residue pair" null already reaches r = 0.42 / MAE 1.1 kcal/mol on one 344-mutation benchmark [4]. **If PhysDDG cannot clearly beat residue-pair-lookup + RSA here, nothing else matters.**

**Domainome.** ~500k variants over 522 human domains, yeast abundance via growth [5]. Not ΔΔG: a sigmoidal, per-domain-saturating abundance readout on *isolated* domains. Signal is almost entirely burial plus aggregation/degron effects, compressed at both ends. Use for per-domain Spearman and AF-mode validation, **not** for fitting energy weights.

### Term triage

| Term | Verdict |
|---|---|
| 1 vdW attraction, 3 apolar solvation/burial | **Core.** The hydrophobic-burial axis; most of the Megascale signal. |
| 4 polar desolvation, 5/6 H-bonds | **Core but entangled.** Burying an unsatisfied donor/acceptor is the second axis. Expect 4 and 5/6 to be strongly collinear — fit jointly, report the correlation, do not interpret individual weights. |
| 2 vdW repulsion (capped) | **Core.** The cap is the single most important fitted parameter (R3). |
| 8 side-chain entropy | **Delete as a separate term** — get it free from Boltzmann rotamer averaging (§5.2). |
| 9 backbone torsion (φ/ψ, pre-Pro) | **Keep**, but it cannot capture Pro's loss of the backbone N–H. Needs a companion indicator (R4). |
| 10 helix propensity | **Likely redundant** with 9 + Δref. Ablate; drop unless worth ≥0.01 Spearman. |
| 11 cavity / packing defect | **Redundant** with 1/3, as the design admits. Drop in v1. |
| 12 special cases (SS, cis-Pro, metal, Gly φ>0) | **Keep — high value per parameter.** Few rows, huge errors (a broken disulfide is 2–5 kcal/mol). But Megascale *excluded* >4-Cys and metal/ligand structures [1], so these can only be fitted on FireProt: expect poor determination, shrink hard. |
| 13 helix dipole / capping | **Over-engineering.** 0.1–0.5 kcal/mol effects, few informative rows, collinear with 7. Defer. |
| 7 electrostatics | **Keep but constrain** (R5, R6). Risk of being actively harmful at the surface. |
| **Missing: environment-dependent statistical potential** | **Largest omission** (§5.1). |

**Is local repacking worth its cost?** Prior: low-to-moderate. Rosetta cartesian_ddg does constrained relaxation and full repacking, yet needs a 1/2.47 rescale to reach kcal/mol and only matches RaSP on S669 [2,4]; FoldX does far less, is better calibrated in magnitude (1/1.04) and not much worse in rank [4]. Megascale-trained shallow models do well on burial + identity alone. My expectation: repacking buys ~0.01–0.03 Spearman on Megascale, concentrated in buried small→large and large→large substitutions (~10–15% of rows), and ~nothing on FireProt. Real, but not where week one should go. **Build Stage-1 lite with the statistical potential first; treat repacking as an ablation with a pre-registered gate (§6).**

---

## 2. Unfolded state, the link, and fitting

### 2.1 Δref is confounded with the assay — most serious technical issue in the design

Tsuboyama ΔG is not measured directly: K50 is measured, and ΔG is inferred using an **unfolded-state model K50,U that is itself a PSSM fitted to 64,238 scrambled sequences** [1]. That is a per-amino-acid sequence correction — the same functional form as the proposed 20-value Δref. Therefore:
- Any systematic error in Tsuboyama's K50,U is absorbed, residue by residue, into your fitted Δref. Those 20 numbers become assay artefacts wearing a physical label, and will not transfer to FireProt/S669 (thermal/chemical denaturation, a different unfolded-state reference entirely).
- Proline explicitly inhibits cleavage by both proteases [1]. X→Pro and Pro→X carry an extra assay-specific channel; treat Pro separately or exclude it from the Δref fit.

**Proposal.** `r_a = r_a^prior + δ_a^shared + δ_a^(assay)`: EEF1 prior fixed; `δ^shared` under strong L2 shrinkage (σ ≈ 0.3 kcal/mol); a **Megascale-only** nuisance vector `δ^(Megascale)` excluded at inference. If `δ^(Megascale)` comes out large (>0.4 kcal/mol for several residues), that is itself a publishable diagnostic: anyone training on Megascale ΔG is partly fitting the PSSM.

### 2.2 A soft-clip on ΔΔG is the wrong censoring model

Dynamic range is ~5 kcal/mol, and ΔG becomes unreliable as K50 approaches K50,F or K50,U [1]. Crucially **the censoring bound lives in ΔG space and depends on ΔG_wt of that domain**: in a hyperstable domain (ΔG_wt ≈ 4.5) stabilising mutations are censored; in a destabilised background (ΔG_wt ≈ 1) destabilising ones beyond ~1.5 kcal/mol are censored. A single monotone `g` applied to ΔΔG cannot represent this — it compresses domains that are not compressed and biases fitted weights downward on stable domains.

**Proposal.** Predict `ΔG_mut = ΔG_wt_obs − ΔΔG_pred` and use an **interval-censored (Tobit) likelihood in ΔG space**, with per-row bounds from the reported K50 limits and 95% CI. Rows with CI > 0.5 kcal/mol (the paper's own low-confidence flag [1]) get interval likelihoods, not point targets. Keep a nonlinearity in `g` only if ablation justifies it *after* censoring is handled — I expect it then buys almost nothing, and you save parameters.

For Domainome the link is genuinely sigmoidal and per-domain: `fitness ≈ σ((ΔG_wt − ΔΔG_pred − m_d)/s_d)`, with `m_d, s_d` as nuisance parameters fitted by profile likelihood against synonymous/stop-codon anchors while the energy weights are held fixed.

### 2.3 Fitting protocol

1. **Freeze the hidden degrees of freedom.** "≈15 weights + 20 refs" badly understates complexity: radii, well depths, EEF1 λ and ΔG_free, H-bond geometry, cutoffs, rotamer-library resolution, repulsion cap, neighbour radius. Take them from published CHARMM19/EEF1 values and declare them fixed in a `params_frozen.yaml` with provenance. Anything touched after seeing a test number is a fitted parameter and must be counted and disclosed.
2. **Objective.** Per-domain-centred Huber on ΔG with inverse-variance weights (use the reported 95% CI), plus a differentiable rank surrogate (soft-Spearman, or pairwise logistic over within-domain pairs with |ΔΔG| > 0.5) at weight ~0.3. Pure pooled MSE will chase the designed hyperstable tail.
3. **Multi-dataset weighting.** Do not pool by row count (300k vs 3k). Weight so effective contributions are comparable, e.g. `w_d ∝ 1/sqrt(n_clusters_d)`. Fit energy weights on Megascale-train + FireProt-trainval with per-assay scale `a_d` and offset `b_d`. Resist per-protein offsets on Megascale (they absorb real ΔG_wt signal); per-domain nuisance is appropriate only for Domainome.
4. **CV.** Grouped by mmseqs2 cluster at 25–30% identity, designed proteins in their own group. Report mean ± s.e.m. per-domain Spearman over clusters, not over rows.
5. **Antisymmetry as a constraint, not a metric.** Add `λ·|f(wt→mut) + f(mut→wt)|²`, computing the reverse by re-running the pipeline with the modelled mutant as the new "WT". Roughly doubles cost, worth it: FoldX's mean antisymmetry violation on S669 is ⟨δ⟩ ≈ 0.34 with r_dir−rev ≈ −0.2 [2], so this is a cheap way to look good on S669/Ssym.
6. **Free validation signal.** Megascale contains selected double mutants [1]; thermodynamic-cycle additivity residuals are an unlabelled consistency check — use as regulariser and held-out diagnostic.

---

## 3. Specific risks

**R1 — "include current" rotamer breaks WT/mutant symmetry.** Step 1 adds the native rotamer as a candidate, so WT gets crystallographic χ while the mutant only gets library rotamers. This systematically lowers E(WT) and inflates *every* ΔΔG positive. "Identical protocol" does not save you, because the candidate *sets* differ. Fix: drop native rotamers for both, or seed the mutant with WT χ up to the last common χ (standard preserve-χ trick). Verify with a null test — a WT→WT "mutation" must return exactly 0.000.

**R2 — Neighbour set must be identical for WT and mutant.** Define the flexible set as the *union* of WT and mutant neighbourhoods, computed once; use CB-plus-rotamer-cloud distance, not CB–CB at 8 Å. For Trp/Arg/Lys a CB cutoff silently misses contacts, again biasing small→large.

**R3 — Fixed backbone over-penalising small→large.** One global repulsion cap is too blunt; real backbone give is site-dependent. Make it `cap(site) = c0 + c1·flex(site)` with flex from normalised contact number (B-factor for xtal, pLDDT for AF). Two parameters, probably the highest-value two in the model.

**R4 — Pro and Gly.** φ/ψ log-odds are necessary, not sufficient. X→Pro additionally (i) deletes the backbone N–H, large when that N–H donates an H-bond (helix interiors, β-strands), and (ii) restricts φ at i, ψ at i−1. Add an explicit "WT backbone N–H is H-bonded" × X→Pro indicator. For Gly: X→Gly adds unfolded-state backbone entropy (~1 kcal/mol scale; cf. Pace–Scholtz where Gly ≈ 1.0 and Pro ≈ 3.2 relative to Ala [6]); Gly→X is catastrophic at positive φ — keep that indicator.

**R5 — His and buried ionisables.** Do not assign His a formal charge; model it neutral with both tautomers, choose by H-bond satisfaction. FireProt rows span a wide pH range (the table is literally `4_fireprotDB_bestpH.csv`), so a fixed protonation state is wrong for some rows — record pH per row and check residuals against it for His/Asp/Glu. For buried Asp/Glu/Lys/Arg, formal charges with ε = 4r give absurd energies; apply burial-dependent charge attenuation (one parameter) or Debye–Hückel screening at ~0.15 M.

**R6 — Surface electrostatics.** Surface salt bridges contribute ~0.1–0.5 kcal/mol experimentally; unscreened Coulomb at ε = 4r over-predicts them by an order of magnitude and injects noise into the majority of rows, which are surface rows. Use screened Coulomb and *check the fitted weight is small*. If ablating term 7 does not hurt, report that rather than keeping it for appearances.

**R7 — Crystal-structure pathologies in FireProt/S669.** Megascale is clean (ligand-, metal- and crystal-contact-stabilised structures excluded by construction [1]); FireProt is not — heme proteins, Ca-binding nucleases, obligate oligomers. A mutation at a cofactor site looks like cavity creation if HETATMs are stripped and like a buried site if they are not. Document (a) biological assembly vs asymmetric unit, (b) HETATM policy. Flag every row within 5 Å of a HETATM or a crystal-contact-only neighbour and **report metrics with and without them**.

**R8 — Structure hygiene.** Missing loop residues near a site silently change burial; altlocs and partial occupancies double-count atoms. Enforce altloc A only, occupancy ≥ 0.5, flag/reject rows with a chain break within 10 Å of the site, reject rows whose WT site has missing side-chain atoms.

**R9 — Position numbering.** The most common silent killer in this field (author numbering vs SEQRES vs UniProt, insertion codes, negative indices, 0- vs 1-based tables). Mandate a hard assertion: for 100% of rows the WT residue at the mapped position must match the table, else drop and count. Print the drop count in every run log. Never apply a global offset "fix" without a per-protein check.

**R10 — Homology leakage.** FireProt, S669, Ssym and ThermoMutDB all descend from ProTherm and overlap each other; Megascale's small domains include homologues of classic test proteins. Run mmseqs2 *and* foldseek all-vs-all between every training cluster and every test set, and report both the full test metric and the strict non-homologous subset. Ssym in particular relies on experimentally determined mutant structures for a handful of proteins — treat it as an antisymmetry diagnostic only, never as a performance claim.

---

## 4. AF mode: what will actually help (ranked by evidence)

1. **Repack all side chains with the same protocol before scoring (point c) — keep, highest value.** AF side chains are predicted, not observed; repacking both xtal and AF inputs with one procedure removes most of the domain shift. This is the thing that makes AF mode work, and it costs nothing extra.
2. **Construct cropping for Domainome (point d) — keep, high value.** The assay measures an isolated Pfam domain; scoring the full-chain model gives wrong burial at domain boundaries. Crop to the assayed construct, treat cut termini as exposed.
3. **pLDDT as a flexibility modulator, not an output shrink.** Evidence that AF confidence correlates with experimental ΔΔG is weak [7]; what pLDDT really says is that local geometry — and hence the repulsion/packing terms — is unreliable. Feed it into the R3 cap and into an abstention flag rather than multiplying the prediction. Per-site attenuation *does* change within-domain ranking, so it is not free: test as an explicit ablation (G5). Expect it to help AUC-style Domainome metrics and hurt calibrated RMSE.
4. **pLDDT < 50 handling (point a).** Fine, but "treat as solvent-exposed coil" is a strong assumption; safer to exclude those residues from the environment and mark the row low-confidence.
5. **Quality expectation to state and test.** Homology models at ≥40% template identity give crystal-like ΔΔG accuracy for cartesian_ddg, while FoldX and ddg_monomer degrade to near the null model below ~45% [4]. High-pLDDT AF regions generally beat a 40%-identity homology model, so expect AF mode to be fine where pLDDT > 80 and near-useless where pLDDT < 70.
6. **Missing partners — the under-appreciated AF failure.** An AF monomer of an obligate complex member exposes what is really an interface. Detect it (large exposed hydrophobic patch, low-pLDDT flanks) and flag. Note that for Domainome (isolated domains) the monomer model may legitimately agree with the assay — so this is a reporting issue as much as a modelling one.
7. **Over-compacted / idealised cores.** Less of a problem than folklore suggests and largely absorbed by item 1 plus a separate fitted repulsion cap for AF inputs (1 parameter). Do not build machinery before measuring it.

---

## 5. Missing ideas, ranked by expected gain / effort

**5.1 Environment-dependent statistical potential (gain high, effort low) — do this first.** A PDB-derived log-odds table P(aa | burial bin × secondary structure × optionally φ/ψ basin), entering with **one fitted weight**. This is the non-DL version of what ProteinMPNN-style models learn; it is derivable from the same /rsyncPDB non-redundant set you already need for the rotamer library; and it captures context effects (β-branched in strands, Pro in loops, charge at surface) that no 15-weight pairwise force field reproduces. The ~180 table entries are *estimated from structural statistics, not fitted to ΔΔG*, so the honest parameter count rises by 1. I expect this single term to contribute more to Megascale per-domain Spearman than terms 5, 6, 7, 10, 11 and 13 combined.

**5.2 Boltzmann-average instead of minimise (gain high, effort very low).** Replace `min_rotamers E` with `F = −kT·log Σ exp(−E/kT)`. This (i) yields side-chain conformational entropy for free, letting you delete term 8, (ii) is smooth and differentiable, making fitting far better behaved, (iii) is more physical. **Algorithmic consequence: DEE prunes for the global minimum and is the wrong algorithm once you want a partition function.** Use mean-field / belief propagation or a short batched Gibbs sweep — dense tensor operations that GPU-batch perfectly, unlike DEE's branchy logic.

**5.3 Censored likelihood + per-domain Domainome link (gain high, effort low).** §2.2.

**5.4 Explicit antisymmetric training (gain high on S669/Ssym, effort medium).** §2.3 item 5.

**5.5 Per-protein stability context (gain medium-high, effort low).** The same ΔΔG means different things in a ΔG_wt = 1 and a ΔG_wt = 4.5 domain. Expose ΔG_wt (predicted, or observed where available) as an input to `g` for abundance-type readouts. This is the principled way to use Domainome without corrupting the energy function.

**5.6 Contact-number burial instead of SASA (gain medium, effort very low).** A smooth sigmoid-weighted CB neighbour count within 10 Å is differentiable, ~50× cheaper than SASA, and correlates ~0.9 with RSA. Use it for the cap, entropy weighting and electrostatic screening.

**5.7 Backbone ensemble (gain medium, effort medium).** 10–20 perturbed backbones (low-frequency normal modes or small Cartesian noise), Boltzmann-averaged ΔΔG. This is the physically correct fix for R3, where the fitted cap is the cheap surrogate. **Build only if the cap's residuals still show a small→large bias** (G3), else it is 10–20× cost for a parameter you already fitted.

**5.8 Additivity regularisation on Megascale double mutants (gain medium, effort low).** Free labels, already in the dataset [1].

**5.9 Continuous χ refinement (gain low, effort medium).** Design step 5. With a dense backbone-dependent library plus Boltzmann averaging, sub-rotamer relaxation is largely redundant. Gate hard (G4).

**5.10 Deprioritise:** helix dipole/capping, separate cavity term, helix propensity as a distinct term. Each spends a parameter on a hypothesis not yet shown to be needed.

---

## 6. Minimal ablation plan and decision gates

**Baseline ladder** (identical rows, grouped CV, natural-only as primary metric):

| ID | Model |
|---|---|
| B0 | Residue-pair average ΔΔG lookup, no structure — expect r ≈ 0.42 on low-throughput sets [4] |
| B1 | RSA only |
| B2 | hydrophobicity × burial + Δref |
| B3 | **statistical potential (§5.1) + burial — 2 weights** |
| B4 | Stage-1 lite: all energy terms, mutant rotamer Boltzmann-averaged in a fixed WT environment |
| B5 | B4 + neighbour repacking |
| B6 | B5 + backbone ensemble |
| B7 | B6 + continuous χ refinement |
| X | FoldX 5.1 native, identical rows |

**Pre-register these gates before looking at any test number.**

- **G1 (does the physics earn its keep?)** B4 beats X by ≥0.08 mean per-domain Spearman on Megascale-test natural-only and ≥0.05 Pearson on FireProt-test; *and* B4 − B3 ≥ 0.03. **If B4 − B3 < 0.02, the force field is decoration on a statistical potential — say so, ship B3-plus, and reallocate effort.**
- **G2 (repacking).** B5 − B4 ≥ 0.02 mean per-domain Spearman on Megascale-test, **and** ≥0.03 on the buried small→large subset, **and** present in ≥70% of CV folds, **and** runtime still within the <1 min / 300-residue target. Otherwise **drop neighbour repacking** and keep the Boltzmann-averaged single-side-chain model.
- **G3 (backbone ensemble).** Build only if B5 residuals still show significant ΔΔG bias vs. volume change at buried sites (|slope| > 0.02 kcal/mol/Å³, p < 0.01); then require B6 − B5 ≥ 0.015.
- **G4 (continuous refinement).** B7 − B6 ≥ 0.01 at ≤2× runtime, else drop.
- **G5 (AF mode).** After identical repacking, paired xtal/AF per-site ΔΔG agreement r ≥ 0.9. pLDDT attenuation ships only if it improves Domainome mean per-domain Spearman by ≥0.02 *and* degrades Megascale-test by ≤0.005.
- **G6 (transfer — the gate that backs the PROMPT's "the physics should transfer" claim).** Fit weights on Megascale only and on FireProt only, separately. **No term weight may change sign**, and none may change by more than 3×. Any term that fails is fitting assay idiosyncrasy: shrink to zero or delete. Publish this table — it is the strongest available evidence that the model is physics and not a Megascale regression.
- **G7 (leakage).** Every headline number reported twice: full test set and strict non-homologous subset (<25% id to anything in training). If the gap exceeds 0.05 Spearman, the full number is not the headline.

**Order of work:** B3 → B4 → fitting protocol (§2) → G1 → G6 → AF mode → *only then* B5. Stage 2 (GPU packer) should not begin until G1 and G6 have passed.

**On the speed target.** <1 min for 300-residue saturation mutagenesis is achievable, but the binding constraint is pair-table memory, not FLOPs. Shared WT–WT neighbour pair tables are ~L × 30 × R² entries (a few million — trivial); mutation-specific tables dominate. Use bf16 storage with fp32 accumulation and a 6 Å heavy-atom cutoff on the pair term. And once §5.2 is adopted, drop DEE from the plan entirely.

---

## 7. References

1. Tsuboyama K. et al., *Nature* 620:434–444 (2023), doi:10.1038/s41586-023-06328-6 — 776,298 ΔG; 331 natural + 148 designed domains, 40–72 aa; dynamic range ~5 kcal/mol; K50,U from a PSSM fitted to 64,238 scrambled sequences; Pro inhibits both proteases; >4-Cys, metal/ligand and crystal-contact-stabilised structures excluded; replicate K50 R = 0.97/0.99; trypsin-vs-chymotrypsin ΔG R = 0.94; per-scan Pearson 0.94 ± 0.04 over 478 domains; CI > 0.5 kcal/mol flagged low-confidence.
2. Pancotti C. et al., *Brief. Bioinform.* 23:bbab555 (2022) (S669) — direct-variant Pearson 0.21–0.5 across tools, systematic compression toward zero; FoldX total Pearson ≈ 0.31, RMSE 2.39, antisymmetry ⟨δ⟩ ≈ 0.34, r_dir−rev ≈ −0.2.
3. Blaabjerg L.M. et al., *eLife* 12:e82593 (2023) (RaSP) — S669 direct Pearson: RaSP 0.39, Rosetta 0.39, FoldX 0.22.
4. Valanciute A. et al., *Comput. Struct. Biotechnol. J.* 21:66–73 (2023) — residue-pair-average null r = 0.42, MAE 1.1 kcal/mol over 344 mutations; homology models match crystal accuracy at ≥40% template identity; cartesian_ddg robust, FoldX/ddg_monomer near-null below ~45%; kcal/mol scale factors FoldX 1/1.039, cartesian_ddg 1/2.469.
5. Beltran A., Jiang X., Shen Y., Lehner B., *Nature* 637:885–894 (2025), doi:10.1038/s41586-024-08370-4 — >500k variants, 522 human domains, growth-coupled abundance readout, domains assayed in isolation.
6. Pace C.N., Scholtz J.M., *Biophys. J.* 75:422–427 (1998) — helix propensity: Ala 0.0, Gly ≈ 1.0, Pro ≈ 3.2 kcal/mol.
7. AF2 best-practice reviews (e.g. arXiv:2403.12668 and refs therein) — weak-to-absent correlation between AlphaFold confidence metrics and experimental ΔΔG/stability; pLDDT > 90 well-predicted, < 50 unreliable. Use pLDDT as a geometry-reliability signal, not a ΔΔG proxy.
8. Dieckhaus H. et al., ThermoMPNN, *PNAS* 121:e2314853121 (2024) — Megascale-test Pearson ≈ 0.72–0.75, Spearman ≈ 0.72. This is the realistic DL reference point that "closing the gap" is measured against.
