# Fitted PhysDDG models (JSON) — schema and feature contract

Produced by `scripts/52_ladder_physics.py --stage fit`; loaded by `physddg/predict.py` and `physddg.fit.load_model`.
All weights are in **kcal/mol per unit of the feature** and all models follow D1 (positive = destabilising).

## Which file to use (D22 release — the shipped models are the JOINT fits)
| file | = | stage | inputs | use |
|---|---|---|---|---|
| **`final_stage2.json`** | B6gj | 2 (mean-field, flexible neighbours) | crystal or AF | **default** |
| **`final_stage1.json`** | B6gjs1 | 1 (Boltzmann site, fixed env) | crystal or AF | **default fast mode** (~3.5x faster) |
| **`final_af.json`** | B6gjaf | 2 | **AF models only** (B-factor = pLDDT) | AF mode; adds pLDDT attenuation |
| `ref_stage2.json` / `ref_stage1.json` / `ref_af.json` | B6g / B6gs1 / B6gaf | 2 / 1 / 2 | any | **D19 Megascale-only** reference fits, kept for comparison |
| `B6x.json` | | 2 | any | + `site_scsc`: Megascale/Domainome-tuned, worse on crystal sets. Transparency only |
| `B6g-refzero.json`, `B6g-noref.json`, `B6g-floor.json` | | 2 | | Δref and loss diagnostics, not for production |
| `B5.json`, `B4.json`, `B4j.json`, `B5j.json` | | 1 / 2 | any | earlier rungs |
| `*+C.json` | | | | **test-informed** CXXC disulfide exception — do not use for reporting |
| `B4-capC*`, `B4-noenv/phys/dF/odd*`, `B5-*` | | | | ablations |

**D22: the shipped models are fitted jointly on Megascale-train + FireProt-train** (per-assay scale/offset,
cluster-balanced weights, Megascale δ_ref as a nuisance). They beat the Megascale-only `ref_*` models on every
assay except Megascale itself, where they are equal. **The nuisance block must never be applied at inference** —
see the prediction rule below. Model form is otherwise identical to D19: B6g = B5 + `ncap_pref` + `helix_dipole`.

## Prediction rule
```
ddg = intercept + Σ_k weights[k] · x_k + ref[mut] − ref[wt]          (WT→WT is exactly 0 by definition)
```
`ref` is the 20 reference energies: for a joint model this is the **assay-agnostic** prior + shared part, which
is exactly what inference should use. The `nuisance` block (FireProt `scale`/`offset`, Megascale `delta_ref`) is
fitted only to absorb assay idiosyncrasies and **is dropped at inference** (D7). `physddg.fit.predict` ignores it
by construction; a loader that re-implements the sum must ignore it too.

**Attenuated models** (`B6af*`, any model with a `scale_kind` key) are two-stage:
```
core = predict(model["scale_base"], x)                       # same rule, the embedded core model
x["scale_plddt"] = core · (1 − pLDDT/100)                    # 0 where B-factors are not pLDDT
ddg = predict(model, x)                                      # includes the negative weight on scale_plddt
```
`model["scale_base"]` is a complete nested model dict with the same schema. `physddg/predict.py` must branch on
`scale_kind`; a loader that ignores it silently drops the attenuation and raises a KeyError on `scale_plddt`.

## Keys
`weights` (feature → kcal/mol), `intercept`, `ref` (aa → kcal/mol, may be `{}`), `stage` (1 or 2),
`derive_opts` (the feature-transform options — pass to `derive()`), `provenance.caps` (percentile clips, usually
null), `se` (cluster-bootstrap standard errors), `inactive_features` (weights pinned to 0 for lack of variance),
`config` (the full FitConfig), `train` (per-dataset row/cluster counts), `git_commit`, `created`.
`sym_pair` is `{}` except in the antisymmetry ablations. Unknown keys must be ignored by loaders.

## Feature order and provenance
Order is irrelevant — `weights` is a dict and prediction is a sum — but **every key must be present** in the
feature frame. Groups, in the order they appear in `B6.json`:

| group | features | produced by |
|---|---|---|
| engine energies | `e_lj_atr`, `e_lj_rep`, `e_solv_apolar`, `e_solv_polar`, `e_hb_sc_bb`, `e_hb_sc_sc`, `e_hb_bb_bb`, `e_elec_dh` | `derive()` from `s{1,2}_dE_*` (`physddg/mutate.py`); `e_lj_rep` is signed-log1p-transformed when `derive_opts["transform"] == "slog_rep"` |
| entropy / prior | `dTS`, `dprior_avg` | `s{1,2}_dTS`, `s{1,2}_dprior_avg` |
| statistics | `d_env_cn_ss_bb`, `d_rama`, `d_rama_prev` | `physddg/rotlib.py` (PDB statistical potentials) |
| B3+ remainder | `bur`, `pro_nhhb`, `gly_posphi`, `occ_dvol_loss`, `ss_broken` | `physddg/features_simple.py` |
| **B6g additions** | `ncap_pref`, `helix_dipole` (B6x adds `site_scsc`) | `physddg/features_extra.py` (`site_table` + `candidate_features`) |
| unfolded-state prior | `dref_apolar`, `dref_polar` | `s{1,2}_dref_*`; present only when `derive_opts["drefp"]` is true |
| attenuation | `scale_plddt` | computed at inference, see above |

B6g feature definitions (from the WT structure, no side-chain modelling; both are state-function differences, D15):
- `ncap_pref` = E_ncap(mut) − E_ncap(wt) at N-cap positions only (0 elsewhere), in kT, from
  `physddg/data/ncap_pref.json` (PDB-derived, `scripts/53_ncap_stats.py`). Weight ≈ 0.29 kcal/mol per kT.
- `helix_dipole` = (q(mut) − q(wt)) × (+1 at helix N1–N3, −1 at C1–C3), q = formal side-chain charge, His neutral.
  Weight ≈ 0.09 kcal/mol per charge. (Measurably inert in B6g: dropping it changes CV by −0.001; kept per D19.)
B6x only: `site_scsc` = min(side-chain heavy atoms of other residues within 4.5 Å of the WT side chain, 30)/30 for
every substitution — a WT-site main effect (not a state-function difference).

**The attenuated model's `scale_base` is itself a joint model**, so the nuisance rule above applies to it too.

**pLDDT for `final_af.json`:** `flag_lowplddt = 1 − pLDDT/100` only when the structure is an AF model whose
per-structure mean B-factor is in [30, 100]; 0 for crystal structures and for design models with B = 0.

## Reproducing
`PYTHONPATH=. .venv/bin/python scripts/52_ladder_physics.py --stage fit` (selection: `--stage select|caps|b6`,
all on Megascale-train/val only). Numbers and gate verdicts: `notes/ladder_physics.md`.
