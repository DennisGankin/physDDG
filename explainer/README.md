# PhysDDG explained

**PhysDDG takes a protein structure (X-ray or AlphaFold) and a list of single amino-acid substitutions and predicts,
for each, the change in folding stability, ΔΔG in kcal/mol (positive = destabilising), with every physical
contribution shown, at about 1 ms per mutation on one GPU.**

> The predictor was built in an exercise testing whether AI agents can run a bioinformatics project: a PI agent
> directed 29 subagents on 26–27 October 2026. This folder explains the method and how it was reached. Claims the
> project's own audits withdrew are not repeated.

![PhysDDG pipeline](figures/pipeline.svg)

**In three sentences.** For each mutated position, PhysDDG builds the original and the new side chain the same way, in
many possible shapes, and scores every shape with classical force-field energies. It adds statistics learned from known
structures (how buried the position is, which amino acids normally sit there, which backbone angles they tolerate).
A weighted sum with 41 numbers fitted to 216,000 measured stability changes turns this into one ΔΔG, and on held-out
proteins it ranks mutations better than FoldX (per-protein Spearman ρ 0.69 vs 0.63) but below deep-learning
predictors (≈0.77, published).

## Versus the standard approaches

FoldX and Rosetta build one best mutant model starting from the experimental wild-type side chain. Statistical tools
(PoPMuSiC, KORPM, DDGun) look up environment tables. Deep learning learns a representation from millions of
parameters. PhysDDG (1) rebuilds the wild type exactly like the mutant so modelling errors cancel, (2) averages over
many side-chain shapes (Boltzmann average) rather than trusting one, (3) mixes physics with structure statistics and
fits the mix on a large assay, (4) evaluates all 20 amino acids at all positions as one GPU computation: a full scan
of a 269-residue protein takes 3.6 s (fast mode) or 8 s (default), against 7–14 CPU-hours for FoldX.

## Six steps

| # | step | why it is needed | code |
|---|---|---|---|
| 1 | Describe each site: angles, secondary structure, burial (AlphaFold: crop, trim) | everything depends on context; burial alone ranks at ρ ≈ 0.53 | `physddg/structure.py`, `physddg/afmode.py` |
| 2 | Build library side-chain shapes for wild type **and** all 19 mutants alike | otherwise every mutation looks destabilising from modelling error | `physddg/mutate.py:154`, `rotlib.py`, `builder.py` |
| 3 | Score shapes with 9 force-field terms; Boltzmann-average; stage 2 lets ≤ 16 neighbours adapt | one wrong shape can give a 50 kcal/mol clash; neighbours make room for larger side chains | `mutate.py:223`, `:383`, `energy.py` |
| 4 | Add PDB statistics: environment potential, Ramachandran, cavity, N-cap, helix dipole | position sensitivity carries most of the signal | `mutate.py:556`, `features_extra.py` |
| 5 | Weighted sum with 41 fitted numbers (robust, censoring-aware, homology-weighted) | raw energies are not ΔΔG units; unfolded-state reference needed | `fit.py`, `predict.py:122`, `:481` |
| 6 | Report per-term contributions and flags; AlphaFold: withhold pLDDT < 70 etc. | no method works below pLDDT 70; withholding 14% gives +0.03 ρ on the rest | `predict.py:530`, `afmode.py:344` |

## Evidence in brief

On 21 natural held-out Megascale domains (22,159 variants), PhysDDG ρ = 0.692, FoldX 0.629: +0.063 (project CI
[+0.006, +0.120]), better on 19 of 21 domains; Pearson 0.70 vs 0.60, RMSE 0.78 vs 1.85 kcal/mol. A 10-parameter
statistics-only model reaches 0.648, so the physics adds +0.044. On other assays (FireProt, S669, S461, Domainome)
there is no clear win over FoldX or the simple model. **Not claimed:** that the physics generalises across assays, that
it closes the gap to deep learning. Known weak spots: forward/reverse
consistency (r ≈ 0.24), stabilising mutations, buried small→large, low-confidence AlphaFold regions.

## Contents

| file | what |
|---|---|
| [METHOD.md](METHOD.md) | Deliverable A: full method, worked example (real 1PSE run), pseudocode, results, limitations, credits |
| [JOURNEY.md](JOURNEY.md) | Deliverable B: eight turning points, dead ends, the research process |
| [INSIGHTS.md](INSIGHTS.md) | Deliverable C: scientific, methodological and engineering insights, ranked |
| [DISCREPANCIES.md](DISCREPANCIES.md) | where records disagree with code or results, open questions |
| `physddg_explainer.html` | the same content as one interactive page (clickable pipeline, expandable timeline, stepwise example) |
| `worked_example/` | `trace.py` (reruns the example), `trace.json`, `trace_stdout.txt` |
| `figures/` | SVG figures and `make_figures.py`; `build_html.py` builds the page |

Cross-links: each step in METHOD.md links to the turning points that produced it; each turning point links back to
the steps it shaped; each insight links to where it arose.

## Reproducing the example

```bash
cd stability_hackathon/explainer/worked_example
CUDA_VISIBLE_DEVICES= PYTHONPATH=.. ../../.venv/bin/python trace.py   # CPU, about 4 minutes
```
(`trace.py` writes `trace.json` into the current directory; paths inside it refer to the project root.)
