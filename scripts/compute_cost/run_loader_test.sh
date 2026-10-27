#!/bin/bash
# R3-12d loader-optimization timing test on PACS (2 domains x 2 seeds).
# GPU0 and GPU1 each run a serial queue of 7 jobs; the two queues run in parallel.
set -u

PY=/home/miniconda3/envs/lf_ssq/bin/python
CODE=/home/qsong/cifa_loader_test/cifa
OUT=/home/qsong/cifa_loader_test/results_loader
mkdir -p "$OUT"

run () { # GPU DOMAIN SEED VARIANT VALW EXTRA...
  local gpu=$1 dom=$2 seed=$3 variant=$4 valw=$5; shift 5
  echo "[GPU$gpu] $dom seed$seed $variant valw=$valw $@"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$CODE/train.py" \
    --dataset pacs --target_domain "$dom" --seed "$seed" \
    --variant "$variant" --val_workers "$valw" \
    --workers 8 --output_dir "$OUT" "$@"
}

queue0 () {
  # sketch seed0: full w0, full w4, noswad, erm
  run 0 sketch 0 full 0
  run 0 sketch 0 full 4
  run 0 sketch 0 noswad 4
  run 0 sketch 0 full 0 --no_cifa --no_swad
  # art seed1: full w0, full w4, noswad
  run 0 art_painting 1 full 0
  run 0 art_painting 1 full 4
  run 0 art_painting 1 noswad 4
}

queue1 () {
  # sketch seed1: full w0, full w4, noswad
  run 1 sketch 1 full 0
  run 1 sketch 1 full 4
  run 1 sketch 1 noswad 4
  # art seed0: full w0, full w4, noswad, erm
  run 1 art_painting 0 full 0
  run 1 art_painting 0 full 4
  run 1 art_painting 0 noswad 4
  run 1 art_painting 0 full 0 --no_cifa --no_swad
}

echo "start $(date)"
queue0 > "$OUT/queue0.log" 2>&1 &
Q0=$!
queue1 > "$OUT/queue1.log" 2>&1 &
Q1=$!
wait $Q0 $Q1
echo "all done $(date)"
