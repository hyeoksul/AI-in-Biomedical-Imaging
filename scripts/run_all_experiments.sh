#!/usr/bin/env bash
# Reproduces the full experiment matrix (assignment tasks 1-1 through 2-2, plus
# the OOD extra credit). Single GPU -> runs sequentially, expect ~6 hours.
# Run from the repo root: bash scripts/run_all_experiments.sh
set -uo pipefail

PY="/d/Graduate Study/AI in biomed imaging/Bio_image/MicroDegree/holograph_env/Scripts/python.exe"
SRC="src"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

run() {
  local name="$1"; shift
  echo "=== $(date '+%H:%M:%S')  START $name ==="
  "$PY" "$@" 2>&1 | tee "$LOG_DIR/$name.log"
  echo "=== $(date '+%H:%M:%S')  DONE  $name ==="
}

# ---- 1-1: architecture comparison (baseline / wide / deep, L1, 20 epochs) ----
run sup_baseline_e20_l1 "$SRC/train_supervised.py" --run_name sup_baseline_e20_l1 --arch baseline --loss l1 --epochs 20
run sup_wide_e20_l1     "$SRC/train_supervised.py" --run_name sup_wide_e20_l1     --arch wide     --loss l1 --epochs 20
run sup_deep_e20_l1     "$SRC/train_supervised.py" --run_name sup_deep_e20_l1     --arch deep     --loss l1 --epochs 20

# ---- 1-2: epoch comparison (deep, L1, 10 / 50 epochs -- 20 already covered above) ----
run sup_deep_e10_l1 "$SRC/train_supervised.py" --run_name sup_deep_e10_l1 --arch deep --loss l1 --epochs 10
run sup_deep_e50_l1 "$SRC/train_supervised.py" --run_name sup_deep_e50_l1 --arch deep --loss l1 --epochs 50

# ---- 1-3: loss comparison (deep, 20 epochs, L2 / L1+SSIM -- L1 already covered above) ----
run sup_deep_e20_l2     "$SRC/train_supervised.py" --run_name sup_deep_e20_l2     --arch deep --loss l2      --epochs 20
run sup_deep_e20_l1ssim "$SRC/train_supervised.py" --run_name sup_deep_e20_l1ssim --arch deep --loss l1+ssim --epochs 20

# ---- 1-4 eval: generalization of the 3 loss-variant deep models across all test distances ----
for loss_tag in l1 l2 l1ssim; do
  for dist in 5 10 15 20; do
    run "eval_${loss_tag}_${dist}mm" "$SRC/evaluate.py" \
      --checkpoint "results/supervised/sup_deep_e20_${loss_tag}/best.pth" --arch deep \
      --test_split "test/Distance${dist}mm" --run_name "sup_deep_e20_${loss_tag}"
  done
done

# ---- also eval the 1-1 architecture models + 1-2 epoch models @ 10mm (their native distance) ----
for name in sup_baseline_e20_l1 sup_wide_e20_l1 sup_deep_e10_l1 sup_deep_e50_l1; do
  arch=deep
  [[ "$name" == sup_baseline* ]] && arch=baseline
  [[ "$name" == sup_wide* ]] && arch=wide
  run "eval_${name}_10mm" "$SRC/evaluate.py" \
    --checkpoint "results/supervised/${name}/best.pth" --arch "$arch" \
    --test_split test/Distance10mm --run_name "$name"
done

# ---- 2-1: self-supervised, various distance configs (baseline arch, 30 epochs) ----
run ss_5_5   "$SRC/train_selfsupervised.py" --run_name ss_5_5   --dist_min 5  --dist_max 5  --epochs 30
run ss_10_10 "$SRC/train_selfsupervised.py" --run_name ss_10_10 --dist_min 10 --dist_max 10 --epochs 30
run ss_15_15 "$SRC/train_selfsupervised.py" --run_name ss_15_15 --dist_min 15 --dist_max 15 --epochs 30
run ss_20_20 "$SRC/train_selfsupervised.py" --run_name ss_20_20 --dist_min 20 --dist_max 20 --epochs 30
run ss_5_20  "$SRC/train_selfsupervised.py" --run_name ss_5_20  --dist_min 5  --dist_max 20 --epochs 30

# ---- 2-1 eval: each self-supervised model across all 4 test distances ----
for ss in ss_5_5 ss_10_10 ss_15_15 ss_20_20 ss_5_20; do
  for dist in 5 10 15 20; do
    run "eval_${ss}_${dist}mm" "$SRC/evaluate.py" \
      --checkpoint "results/selfsupervised/${ss}/best.pth" --arch baseline \
      --test_split "test/Distance${dist}mm" --run_name "$ss"
  done
done

# ---- 2-2: distance regression / auto-focus ----
run depth_5_20 "$SRC/train_depth_regressor.py" --run_name depth_5_20 --dist_min 5 --dist_max 20 --epochs 30

# ---- Extra credit: OOD (2mm / 25mm), supervised vs self-supervised ----
run ood_comparison "$SRC/evaluate_ood.py" \
  --supervised_ckpt results/supervised/sup_deep_e20_l1ssim/best.pth --supervised_arch deep \
  --selfsupervised_ckpt results/selfsupervised/ss_5_20/best.pth --selfsupervised_arch baseline \
  --ood_distances 2 25

echo "=== ALL EXPERIMENTS DONE: $(date) ==="
