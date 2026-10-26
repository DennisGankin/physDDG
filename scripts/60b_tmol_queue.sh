#!/bin/bash
# Runs the tmol ddG evaluations used in notes/tmol_eval.md (separate venv, GPU 3).
# usage: scripts/60b_tmol_queue.sh pack|min   (run both modes in parallel)
set -u
cd "$(dirname "$0")/.."
MODE=$1; EXTRA=""; [ "$MODE" = min ] && EXTRA="--min"
PY=.venv_tmol/bin/python; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
TRAIN="EHEE_rd2_0487 2O2W 2RRU 5OAO 5AHT HHH_rd1_0606 2EXD 3V1A HHH_rd4_0676 2QFF 2JZ2 1F0M 1R69 1JIC 2RJV HHH_rd4_0870 3L1X HEEH_rd3_0055 HEEH_KT_rd6_3632 2D1U HHH_rd1_0335 2LHC 2MI6 2L9R 1PV0 1O6X 2KVT 6YSE HHH_rd1_0473 2MA4 2LHR 2KZJ HHH_rd4_0816 HEEH_KT_rd6_0280 2OCH 1OPS 2JVG 2L2D 1UFM 1K1V"
$PY scripts/60_tmol_ddg.py --dataset megascale --split test --tag test_$MODE $EXTRA
$PY scripts/60_tmol_ddg.py --dataset fireprot --split test --tag test_$MODE $EXTRA
# Megascale-train subset for the per-term refit: one protein per homology family, 40 families (seed 0), 37,753 rows
$PY scripts/60_tmol_ddg.py --dataset megascale --split train --proteins $TRAIN --tag train40_$MODE $EXTRA
