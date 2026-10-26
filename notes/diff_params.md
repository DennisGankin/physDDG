# Fitting the physical parameters end to end through the Stage-1 engine (E3, 2026-10-26)

Question: so far only the ~20 term weights + 20 reference energies are fitted; every physical constant
(EEF1 ΔG_free/ΔG_ref/λ/volumes, CHARMM19 radii and well depths, H-bond and electrostatic forms, the rotamer-ensemble
kT, rep_scale, the repulsion cap) is a frozen literature value. Rosetta-style refits of the parameters themselves
usually buy accuracy. Does a **modest, interpretable** refit (32 parameters) of the physics itself — fitted through
the differentiable Stage-1 scan, Boltzmann average included — generalise better than the literature values?

Code: `physddg/diffparams.py` (wrapper), `scripts/111_diffparams_fit.py` (fit θ), `scripts/112_diffparams_features.py`
(features at a θ for whole tables), `scripts/113_diffparams_eval.py` (standard-pipeline evaluation).
Outputs `results/diff_params/`, logs `logs/111_*`, `logs/113_*`. Tests `tests/test_diffparams.py`.

## 1. What is differentiable, and the one hook in E1/E2 code

`physddg/diffparams.py` wraps E1's `SiteScan` (it does not change it):
- **Fixed per θ:** the rotamer candidate sets, their coordinates and library probabilities, the backbone, the
  environment coordinates. They are built once per site (`SiteSet`), so a forward pass is energy evaluation +
  `logsumexp` only.
- **Differentiable:** the per-rotamer energies, the Boltzmann weights, ⟨E_k⟩, the site entropy, ⟨prior⟩, F, and the
  EEF1 reference energies — i.e. the whole Stage-1 feature vector, gradients flowing through the Boltzmann average.
- Per-rotamer neighbour lists are precomputed once (`SiteSet._neighbours`), all sites of a chunk are evaluated in one
  kernel call and each chunk is gradient-checkpointed: **0.03 s forward / 0.08 s forward+backward per protein**
  (≈1.2 GB), ~10× faster than looping `env_energy` per site as `SiteScan` does.

**Hook (the only change to E1/E2 files):** `energy.features` now reads an optional per-type parameter table from
`EnergyParams.extra["type_table"]` (2 lines, `physddg/energy.py`). Absent → the frozen `params.type_table()`, same
code path. `tests/test_diffparams.py::test_hook_absent_is_identical` checks that passing the frozen table explicitly
gives **bitwise identical** energies, and `test_default_stage1_bitwise_unchanged` that `SiteScan.stage1` still
reproduces the committed feature file `data/features/phys_s1_megascale.parquet` **exactly**. Nothing else was
touched (`rep_scale`, `debye` and `Blocks.cap` already accepted arbitrary values).

θ = 0 through the new code reproduces the shipped Stage-1 features to float32 rounding (max |Δ| ≤ 7e-4 kcal/mol on
`s1_dF` over all FireProt rows) and the shipped B6gs1 model fitted on those features reproduces its published
numbers exactly (val natural ρ_prot 0.7262, grouped-CV natural 0.6758).

## 2. Parameterisation (32 parameters, all multiplicative around the literature value)

θ_k = ln(value / literature value), bounded to |θ| ≤ 1 (scale within ×/÷ 2.7), penalised λ‖θ‖².

| group | n | what |
|---|---|---|
| `solv_<type>` | 14 | EEF1 solvation strength per atom-type class (scales ΔG_free **and** ΔG_ref together): N_bb, O_bb, CH1E(CA+CH1), CH2, CH3, C_aro, N_amide, N_ring(His/Trp), N_lys, N_arg, O_amide, O_carboxyl, O_hydroxyl, S |
| `eef1_lambda` | 1 | EEF1 correlation length λ (all types) |
| `rad_*` | 4 | vdW radius (and the EEF1 R_i, as in EEF1) per class: apolar C, aromatic C, polar N/O, S |
| `eps_*` | 4 | LJ well depth per the same 4 classes |
| `U_hb_bb_bb`, `U_hb_sc_bb`, `U_hb_sc_sc`, `U_elec_dh` | 4 | weight of these terms **inside the packing energy U** that sets the Boltzmann weights (the linear model rescales the features anyway, so only the ensemble effect is fitted) |
| `debye`, `kT`, `U_prior`, `rep_scale`, `rep_cap` | 5 | Debye length; rotamer-ensemble temperature; weight of −kT ln p(rot) in U; repulsive-radius scale (D14); repulsion cap (D9 cap0) |

## 3. Objective and protocol

`min_θ [ min_w L(w; X(θ)) + λ‖θ‖² ]` — variable projection: for every θ the **full B6gs1 linear model** (20 weights
+ 20 shrunk reference energies + intercept, sign constraints, Huber + interval/censored loss, D7/D8 conventions of
`physddg/fit.py`, exactly `scripts/52` `b6g` stage 1) is solved to optimality by L-BFGS-B, and the θ-gradient is the
envelope-theorem gradient (∂L/∂X · ∂X/∂θ at the optimal w). Verified against finite differences
(agreement < 0.5 % at h = 2e-4; per-parameter FD checks in the test file).

Training data: **site-balanced subsample of Megascale-train** — 20 random sites per domain × 3 random variants per
site over all 239 domains = 14,340 rows (4,780 sites), censored rows included as intervals. Outer optimiser L-BFGS-B
over θ, bounds ±1.
Selection: λ ∈ {0.001, 0.01, 0.1} and the stopping iteration chosen on **grouped 5-fold CV over homology clusters**
(the folds of `fit.cross_validate`, seed 0) plus Megascale-val. No test set was touched before the single final look.
Cost: ≈ 11 s per value+gradient on 2 GPUs; a converged fit ≈ 30 min; one θ CV fold ≈ 18 min.

(Results sections below.)

## 4. Does it help? Selection evidence (Megascale-train CV + val only, no test data)

Fitted θ at λ = 0.001 (converged, 30 L-BFGS iterations) lowers the **training** loss by 1.9 % (0.2694 → 0.2644)
and moves parameters by up to ×1.6. Generalisation (`results/diff_params/confirm.csv`; CV = grouped 5-fold over
homology clusters on the **full** Megascale-train rows, fold j scored with the θ fitted without fold j):

| model | val nat ρ_prot | val all ρ_prot | val Pearson | CV nat ρ_prot | CV all ρ_prot | CV Pearson | folds up (nat / all) | FireProt-train ρ / r |
|---|---|---|---|---|---|---|---|---|
| B6gs1 literature parameters (θ = 0) | **0.7262** | 0.6991 | 0.7045 | 0.6758 | 0.6788 | 0.6480 | – | 0.6318 / 0.6491 |
| θ λ=0.001 (converged) | 0.7260 | **0.7005** | **0.7096** | **0.6813** | 0.6832 | **0.6519** | 3/5 / 5/5 | **0.6439 / 0.6639** |
| θ λ=0.001, early stop (it 9) | 0.7202 | 0.6955 | 0.7084 | **0.6813** | **0.6838** | 0.6512 | 3/5 / 5/5 | 0.6405 / 0.6588 |
| θ λ=0.01 (converged) | 0.7253 | 0.6976 | 0.7058 | 0.6786 | 0.6819 | 0.6495 | 3/5 / 5/5 | 0.6402 / 0.6569 |

FireProt-train is **never** used for fitting here, so it is an honest assay-transfer check made before the test look.
The λ-and-iteration sweep on the held-out subsample (`cv_select.csv`, 3 λ × 5 folds × 30 iterations) is flat within
±0.005 and picked λ = 0.001; λ = 0.1 pins θ (max scale 1.05) and gains nothing.

**Pre-set bar** (val per-protein ρ + 0.01 **and** CV up in ≥ 4/5 folds **and** no FireProt/S669-strict loss > 0.01):
**not met.** Val per-protein ρ is flat (−0.000), CV natural per-protein ρ gains +0.0055 but in only 3/5 folds. The
one consistent signal is **calibration**: CV Pearson +0.004 (5/5 folds on all domains), val Pearson +0.005,
FireProt-train Pearson +0.015.

## 5. Single test look (one candidate, chosen by CV before any test number was computed)

`results/diff_params/test_metrics.csv`, `test_paired.csv`; identical rows; 95 % cluster-paired bootstrap vs the
shipped B6gs1 (the stage-1 model this replaces) and vs FoldX-clip.

| set (identical rows) | B6gs1 | B6gs1 + fitted θ | Δ [95 % CI] |
|---|---|---|---|
| Megascale-test natural, ρ_prot | 0.6784 | 0.6830 | +0.005 [−0.007, +0.016] |
| Megascale-test natural, pooled ρ | 0.6591 | 0.6620 | +0.003 [−0.001, +0.008] |
| Megascale-test natural, **Pearson** | 0.6967 | 0.7021 | **+0.005 [+0.002, +0.009]** |
| Megascale-test natural, RMSE | 0.780 | 0.774 | −0.006 |
| Megascale-test natural-strict, ρ_prot / Pearson | 0.6900 / 0.7035 | 0.6896 / 0.7086 | −0.001 / **+0.005 [+0.000, +0.009]** |
| Megascale-test all 28, ρ_prot / Pearson | 0.6843 / 0.6814 | 0.6840 / 0.6858 | −0.000 / **+0.004 [+0.001, +0.007]** |
| FireProt-test (347), r / ρ / ρ_prot | 0.444 / 0.466 / 0.574 | 0.455 / 0.472 / 0.599 | +0.011 [−0.037, +0.048] / +0.006 / +0.025 |
| FireProt-test strict (333), r | 0.449 | 0.458 | +0.009 [−0.039, +0.041] |
| S669-strict (470), r / ρ | 0.328 / 0.340 | 0.343 / 0.354 | +0.014 [−0.009, +0.033] / +0.014 |
| S461-strict (285), r | 0.569 | 0.577 | +0.007 [−0.015, +0.034] |
| Domainome all (532k), ρ_dom / Pearson | 0.4135 / 0.4650 | 0.4126 / 0.4672 | −0.001 [−0.006, +0.006] / **+0.002 [+0.001, +0.004]** |
| Domainome flag-kept, ρ_dom | 0.4479 | 0.4429 | −0.005 [−0.006, +0.005] |

Reading: **the refit never hurts beyond noise and never helps the ranking.** Every rank metric moves by |Δ| ≤ 0.005
with a CI containing 0; the Pearson/RMSE gains are small, positive on every set, and significant only because
Megascale-test and Domainome have tens of thousands of rows. The direction is the same on the two crystal assays
the model was not fitted on (FireProt +0.011, S669 +0.014 Pearson), i.e. the gain is a genuine, assay-transferable
calibration improvement — it is simply ~0.005, not the "substantial accuracy" a Rosetta-style refit promises.
Against FoldX-clip the margin is unchanged (Megascale natural ρ_prot +0.054, Pearson +0.106).

## 6. What the fitted physics looks like (`results/diff_params/params.csv`)

λ = 0.001 Megascale fit; SE and the 2.5/97.5 % range over **12 cluster bootstraps** (warm-started refits);
last two columns are two independent fits at matched λ = 0.01 (the transfer test of the physics itself).

| parameter | literature | fitted (λ=0.001) | scale ± SE | MS λ=0.01 | **FireProt-only λ=0.01** |
|---|---|---|---|---|---|
| `rad_NO` polar N/O radius | 1.60 Å | 1.24 Å | 0.77 ± 0.05 | 0.80 | **0.76** |
| `rep_scale` repulsive-radius scale (D14) | 0.90 | 0.76 | 0.84 ± 0.02 | 0.86 | **0.82** |
| `eps_NO` polar N/O well depth | 0.238 | 0.176 | 0.74 ± 0.02 | 0.94 | 0.86 |
| `solv_N_bb` backbone-N desolvation | −8.90 | −5.56 | 0.62 ± 0.04 | 0.90 | 0.89 |
| `solv_N_ring` His/Trp ring N | −4.00 | −5.23 | 1.31 ± 0.07 | 1.08 | **1.23** |
| `solv_O_carboxyl` Asp/Glu O | −10.0 | −12.3 | 1.23 ± 0.09 | 1.09 | **1.30** |
| `solv_O_bb` backbone O | −5.85 | −7.02 | 1.20 ± 0.12 | 1.00 | **1.18** |
| `solv_N_amide` Asn/Gln N | −7.80 | −9.22 | 1.18 ± 0.07 | 1.06 | 1.06 |
| `eef1_lambda` EEF1 correlation length | 3.5 Å | 4.93 Å | 1.41 ± 0.10 | 1.13 | 1.16 |
| `rad_S`, `rad_Capolar`, `rad_Caro` | 1.89 / 2.365 / 2.10 Å | 2.71 / 2.56 / 2.27 | 1.44 / 1.08 / 1.08 | 1.02 / 1.06 / 1.01 | 1.04 / 1.12 / 1.19 |
| `kT` of the rotamer ensemble | 0.593 | 0.736 | 1.24 ± 0.09 | 1.01 | 1.05 |
| `U_hb_sc_sc`, `U_hb_sc_bb`, `U_elec_dh` in U | 1 | 0.87 / 0.95 / 0.96 | ±0.03 / 0.02 / 0.01 | ~0.99 | 1.00 / 0.96 / 1.00 |
| `rep_cap` (D9 cap0) | 3.0 | 2.65 | 0.88 ± 0.02 | 1.01 | 0.99 |
| `U_hb_bb_bb` | 1 | **1.000 exactly** | 0 | 1.000 | 1.000 |
| the other 10 `solv_*` / `debye` / `U_prior` | | | within 0.88–1.22 | ~1.0 | ~1.0 |

- **Signs and orders are physical.** Every parameter stays within ×/÷ 1.6 of its literature value; none hits the
  ±1 bound (the FireProt λ=0.001 fit does, see below). The consistent story is *polar groups are too big and too
  sticky, their burial is too cheap*: polar radii ×0.77, polar well depth ×0.74, backbone-N desolvation ×0.62, while
  the charged/polar side-chain desolvation of His/Trp N, Asp/Glu O and backbone O goes **up** 1.2–1.3×, and the
  sc–sc H-bond weight inside the rotamer ensemble goes **down** 0.87. That is the classic polar-desolvation ↔
  H-bond balance; the engine validation (`notes/engine_energy.md` (c)) had already flagged `solv_polar` as the one
  term that prefers decoys over native chis, and the refit moves exactly that balance.
- **`rep_scale` 0.90 → 0.76–0.86 in all three independent fits** — ΔΔG data wants the repulsive radii even softer
  than D14's structural calibration, consistent with the known fixed-backbone over-penalty on small→large.
- **`U_hb_bb_bb` is exactly 1 with zero gradient**, as it must be: backbone–backbone H-bonds do not depend on the
  side-chain rotamer, so scaling them inside U cannot change the Boltzmann weights. A free self-check that the
  gradient path is doing what it claims.
- **G6-style transfer of the physics:** θ fitted on FireProt-train alone (2,669 rows, 45 clusters) correlates with
  the Megascale-only θ at **r = 0.80** over the 32 parameters at matched λ = 0.01, and agrees on the sign of every
  parameter whose |scale − 1| > 0.1 (`rad_NO`, `rep_scale`, `eps_NO`, `solv_N_bb`, `solv_N_ring`, `solv_O_carboxyl`,
  `solv_O_bb`, `eef1_lambda`, `rad_Capolar`). **Two independent assays, fitted through the same differentiable
  engine, agree on how the literature force field is wrong** — a stronger statement than the accuracy numbers.
  At λ = 0.001 the FireProt fit runs into the |θ| ≤ 1 bound on 4 parameters (2.7k rows cannot determine 32
  parameters) and the correlation drops to 0.57: the FireProt-only fit needs the stronger prior.

## 7. Verdict

**Negative on the pre-set bar; a small, well-transferring calibration gain.** Fitting 32 interpretable physical
parameters end to end through the Stage-1 engine (Boltzmann average included) buys
+0.004 to +0.005 Pearson and −0.006 kcal/mol RMSE on Megascale-test, +0.01 Pearson on FireProt-test and +0.014 on
S669-strict, with **no** per-protein ranking gain anywhere (|Δρ| ≤ 0.005, CI contains 0) and no loss. The bar
(val ρ_prot +0.01 and CV up in ≥ 4/5 folds) is not met, so **nothing is shipped**: `predict.py` is untouched, no
`physddg/data/params_fitted.json` is written, and the default engine keeps the literature values. The code path
stays in the repository (`diffparams.py`, scripts 111–113) with a test that the default behaviour is bit-for-bit
unchanged.

Why so small, given a 1.9 % training-loss drop? Three reasons visible in the data:
1. **The linear layer already absorbs most of what θ can do.** 20 fitted weights multiply the 8 energy terms; a
   per-type parameter change that acts like a term rescaling is degenerate with them. Only the *within-term,
   per-atom-type* reshaping and the *ensemble reweighting* are new degrees of freedom — and the ensemble part is
   small because the rotamer Boltzmann weights are dominated by repulsion.
2. **The residual is not a parameter problem.** `notes/residual_analysis.md` localises 43 % of the residual variance
   at the site level, unreachable from WT-structure descriptors (GBM R² ≤ 0.25), and the within-site part in
   interactions between substitution type and the existing terms. Re-scaling radii and solvation strengths cannot
   create a missing physical effect (backbone give, real tolerance at SH3-type "core" positions, Pro at φ > 0).
3. **The usual Rosetta-style gain comes with far more parameters and the structural tasks** (rotamer recovery,
   sequence recovery, decoy discrimination) in the objective. With 32 parameters and a single censored ΔΔG
   objective, the likelihood is nearly flat: the loss is already within 0.5 % of its optimum after ~10 iterations
   and the remaining 20 iterations buy nothing out of sample.

### If someone wants to revisit it
- The most promising single knob is **`rep_scale` (0.76–0.86 in three independent fits)**: it is one parameter, it
  transfers, and it points at the known buried-small→large failure. It could be selected on train/val alone
  (one 1-D sweep of the existing Stage-1 features: `scripts/112` with θ = only that entry) and would be a cheap,
  honest, single-parameter change — unlike the 32-parameter refit.
- Fitting θ **jointly with the structural objectives** (rotamer recovery of `scripts/56`, native-vs-decoy chi
  discrimination of `scripts/45c`) would use the data the parameters were originally derived from, and is the
  standard Rosetta recipe we did not follow here.
- Stage 2 was not attempted: with no Stage-1 gain there is nothing to carry through the mean-field packer, and the
  gradient would have to flow through 30 mean-field sweeps.

### Reproduce
```bash
PYTHONPATH=. .venv/bin/python scripts/111_diffparams_fit.py --data megascale --lams 0.001 --folds none --gpus 0,1
PYTHONPATH=. .venv/bin/python scripts/111_diffparams_fit.py --data megascale --lams 0.001,0.01,0.1 --folds 0,1,2,3,4 --gpus 0
PYTHONPATH=. .venv/bin/python scripts/111_diffparams_fit.py --data fireprot  --lams 0.001,0.01 --folds none --gpus 1
PYTHONPATH=. .venv/bin/python scripts/111_diffparams_fit.py --data megascale --lams 0.001 --boot 12 \
    --theta0 results/diff_params/fits/megascale_lam0.001_full.json --maxiter 12 --gpus 0,1
PYTHONPATH=. .venv/bin/python scripts/113_diffparams_eval.py --stage select
PYTHONPATH=. .venv/bin/python scripts/113_diffparams_eval.py --stage confirm --cand zero=zero th0.001=megascale_lam0.001_full
PYTHONPATH=. .venv/bin/python scripts/113_diffparams_eval.py --stage params  --cand th0.001=megascale_lam0.001_full ...
PYTHONPATH=. .venv/bin/python scripts/113_diffparams_eval.py --stage test    --cand th0.001=megascale_lam0.001_full
```
Every configuration tried is logged: 3 λ × (1 full + 5 CV folds) on Megascale, 2 λ on FireProt, 12 bootstraps,
4 evaluated candidates (θ = 0, λ = 0.001 converged, λ = 0.001 early-stopped at iteration 9, λ = 0.01) — the fit
trajectories (objective, held-out metrics and θ at every iteration) are in `results/diff_params/fits/*.json`,
the CV traces in `cv_traces.csv` / `cv_select.csv`, the selection table in `confirm.csv`.

### Caveats
- θ is fitted on a **site-balanced subsample** (14,340 of 227,094 train rows) for speed; the linear weights are
  always refitted on the **full** train rows for every evaluation. A full-row θ fit was not affordable (≈ 16× cost)
  and is the one thing that could still change the verdict, though the training loss is already nearly converged.
- Stage 1 only (the shipped default is Stage 2, B6g); the comparison baseline is B6gs1 on identical rows.
- The 12 bootstrap refits are warm-started with 12 L-BFGS iterations each, so the SEs are a lower bound on the
  true parameter uncertainty.
- `eef1_lambda` and `rad_*` also change the EEF1 Gaussian centre R_i (as in EEF1, where R_i = Rmin/2), so those
  two groups are not independent of the solvation group.
