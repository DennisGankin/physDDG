# Fitted PhysDDG models (JSON) — schema and feature contract

Produced by `scripts/52_ladder_physics.py --stage fit`; loaded by `physddg/predict.py` and `physddg.fit.load_model`.
All weights are in **kcal/mol per unit of the feature** and all models follow D1 (positive = destabilising).

## Which file to use
| file | stage | inputs | use |
|---|---|---|---|
| `B6.json` | 2 (mean-field, flexible neighbours) | crystal or AF | **default / best accuracy** |
| `B6s1.json` | 1 (Boltzmann site, fixed env) | crystal or AF | **default fast mode** (~3.5x faster) |
| `B6af.json`, `B6af_s1.json` | 2 / 1 | **AF models only** (B-factor = pLDDT) | AF mode; adds pLDDT attenuation |
| `B5.json`, `B4.json` | 2 / 1 | any | previous rung, no B6 site terms |
| `B6j.json`, `B4j.json`, `B5j.json` | 2 / 1 | any | joint Megascale+FireProt fit (larger magnitudes) |
| `B6d15.json` | 2 | any | D15 state-function-only variant of B6 (see below) |
| `*+C.json` | | | **test-informed** CXXC disulfide exception — do not use for reporting |
| `B4-capC*.json`, `B4-noenv/phys/dF/odd*` | 1 / 2 | | ablations, not for production |

## Prediction rule
```
ddg = intercept + Σ_k weights[k] · x_k + ref[mut] − ref[wt]          (WT→WT is exactly 0 by definition)
```
`ref` is the 20 reference energies (absent/empty when the model was fitted without them). `nuisance` holds
per-dataset scale/offset/σ and the Megascale δ_ref; **all of it is dropped at inference** (D7) — do not apply it.

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
| **B6 additions** | `site_scsc`, `site_hbsc_bur`, `ncap_pref`, `helix_dipole` | `physddg/features_extra.py` (`site_table` + `candidate_features`) |
| unfolded-state prior | `dref_apolar`, `dref_polar` | `s{1,2}_dref_*`; present only when `derive_opts["drefp"]` is true |
| attenuation | `scale_plddt` | computed at inference, see above |

B6 feature definitions (all from the WT structure, no side-chain modelling):
- `site_scsc` = min(side-chain heavy atoms of other residues within 4.5 Å of the WT side chain, 30)/30, for every
  substitution (0 when wt == mut). A **site main effect**, not a state-function difference.
- `site_hbsc_bur` = min(polar partner atoms within 3.5 Å of the WT side chain, 4) × (1 − RSA). Also site-only.
- `ncap_pref` = E_ncap(mut) − E_ncap(wt) at N-cap positions only, in kT, from `physddg/data/ncap_pref.json`
  (PDB-derived, `scripts/53_ncap_stats.py`). A state-function difference.
- `helix_dipole` = (q(mut) − q(wt)) × (+1 at helix N1–N3, −1 at C1–C3), q = formal side-chain charge, His neutral.
`B6d15.json` replaces the two site-only terms with `scsc_dvol` and `hbsc_dpolar`, which are state-function
differences (D15-compliant) and cost ~0.014 per-protein ρ.

## Reproducing
`PYTHONPATH=. .venv/bin/python scripts/52_ladder_physics.py --stage fit` (selection: `--stage select|caps|b6`,
all on Megascale-train/val only). Numbers and gate verdicts: `notes/ladder_physics.md`.
