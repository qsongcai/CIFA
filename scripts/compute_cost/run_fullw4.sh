#!/bin/bash
# Re-run only the 4 missing full-w4 jobs (CIFA + SWAD + optimized loader),
# after the filename bug fix (filenames now include the variant).
set -u

PY=/home/miniconda3/envs/lf_ssq/bin/python
CODE=/home/qsong/cifa_loader_test/cifa
OUT=/home/qsong/cifa_loader_test/results_loader
mkdir -p "$OUT"

run () { # GPU DOMAIN SEED
  local gpu=$1 dom=$2 seed=$3
  echo "[GPU$gpu] $dom seed$seed full valw=4"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$CODE/train.py" \
    --dataset pacs --target_domain "$dom" --seed "$seed" \
    --variant full --val_workers 4 \
    --workers 8 --output_dir "$OUT"
}

queue0 () {
  run 0 sketch 0
  run 0 art_painting 1
}

queue1 () {
  run 1 sketch 1
  run 1 art_painting 0
}

echo "start $(date)"
queue0 > "$OUT/rerun_q0.log" 2>&1 &
Q0=$!
queue1 > "$OUT/rerun_q1.log" 2>&1 &
Q1=$!
wait $Q0 $Q1
echo "all done $(date)"
