# Negative results

Ideas that were tried and did not help, or did not ship. Each line points to the place with the numbers. The full
record is the research tree ([RESEARCH_TREE.md](../RESEARCH_TREE.md), items marked `[-]`). A longer supplement is
in [`../manuscript/`](../manuscript/).

| idea | outcome | details |
|---|---|---|
| Refitting FoldX's 22 energy terms with free weights | worse than stock FoldX under grouped CV (pooled ρ 0.541 vs 0.569) | [LITERATURE.md](../LITERATURE.md), `scripts/lit_foldx_term_analysis.py` |
| tmol (Rosetta-style energy, GPU) as the engine | too slow and dominated by clashes; refit ≈ FoldX level; kept as a baseline | [notes/tmol_eval.md](../notes/tmol_eval.md) |
| Antisymmetry from one structure (symmetrised ΔF) | r(dir, −rev) 0.15, bias +3.5 kcal/mol | [notes/ladder_physics.md](../notes/ladder_physics.md) (Antisymmetry) |
| Backbone ensembles (gate G3) | no gain, 9× the cost of stage 1 | [notes/ladder_physics.md](../notes/ladder_physics.md), `scripts/58_backbone_ensemble.py` |
| Repulsion-cap variants | differences within noise | [notes/ladder_physics.md](../notes/ladder_physics.md) (Repulsion cap selection) |
| Extra substitution terms (helix/β propensity, cation–π, salt bridges, hydrophobicity × burial, per-SS scaling, …) | no gain on validation | [notes/residual_analysis.md](../notes/residual_analysis.md) |
| `site_scsc` and buried H-bond partner terms | hurt crystal benchmarks or failed the transfer gate (G6) | [notes/ladder_physics.md](../notes/ladder_physics.md) (B6) |
| End-to-end fit of 32 physical parameters | no ranking gain; small calibration gain only | [notes/diff_params.md](../notes/diff_params.md) |
| Pairwise terms from double mutants | little coupling left once assay saturation is removed | [notes/ladder_physics.md](../notes/ladder_physics.md) (Doubles) |
| Floor-aware loss | no change in fit | [notes/ladder_physics.md](../notes/ladder_physics.md) (Floor-aware loss) |
| Elastic-network flexibility as dynamics | adds nothing over static contact number | [notes/enm_flexibility.md](../notes/enm_flexibility.md) |
| pLDDT attenuation as an AlphaFold correction | gain matches a burial-matched control; ships only inside `final_af` with a CI that touches 0 | [notes/af_mode.md](../notes/af_mode.md) |
| pLDDT-modulated repulsion cap; per-protein pLDDT scaling | no gain | [notes/af_mode.md](../notes/af_mode.md) |
| Evolution term (sequence profile) | gains on Megascale and Domainome, but a different information source and not homology-clean; optional only, not shipped | [notes/evo_extension.md](../notes/evo_extension.md) |
| GPU tricks (torch.compile, multi-site batching, fp16/bf16, CUDA graphs) | no useful speed-up | [RESEARCH_TREE.md](../RESEARCH_TREE.md) §6 |

Claims withdrawn after the audits are listed in [AUDITS/audit_1.md](../notes/audit_1.md),
[AUDITS/audit_2.md](../notes/audit_2.md) and in the errata at the top of [MODEL_CARD.md](../notes/ladder_physics.md).
