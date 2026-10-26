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
