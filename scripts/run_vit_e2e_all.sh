#!/bin/bash
# End-to-end ViT-B/16 CIFA experiments on PACS — dual GPU parallel
# Main: 4 domains x 3 methods x 5 seeds = 60 runs (block 6)
# Ablation: sketch x 3 seeds x 3 blocks (3,6,9) = 9 runs
# Total: 69 runs, 2 GPUs, ~2 runs at a time

set -e
cd ~
PYTHON=/home/miniconda3/bin/python
SCRIPT=~/vit_cifa_e2e.py
OUTDIR=~/vit_e2e_results
DATA_DIR=/mnt/sdd1/datasets_qsong/PACS/Homework3-PACS-master/PACS

DOMAINS="art_painting cartoon photo sketch"
METHODS="erm swad cifa"
SEEDS="0 1 2 3 4"
BLOCK=6

mkdir -p $OUTDIR
LOGFILE=~/vit_e2e_run_$(date +%Y%m%d_%H%M%S).log
echo "Starting ViT E2E experiments at $(date)" | tee -a $LOGFILE
echo "Log: $LOGFILE" | tee -a $LOGFILE

# Build task list
TASKS=()
for dom in $DOMAINS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      TASKS+=("$method $dom $seed $BLOCK")
    done
  done
done
# ablation: sketch, cifa, seeds 0-2, blocks 3/6/9
for blk in 3 6 9; do
  for seed in 0 1 2; do
    TASKS+=("cifa sketch $seed $blk")
  done
done

TOTAL=${#TASKS[@]}
echo "Total tasks: $TOTAL" | tee -a $LOGFILE

run_one() {
  local gpu=$1
  local method=$2
  local dom=$3
  local seed=$4
  local blk=$5
  local outfile=$OUTDIR/${method}_${dom}_seed${seed}_block${blk}.json
  if [ -f "$outfile" ]; then
    echo "[SKIP] GPU$gpu $method $dom seed$seed block$blk" | tee -a $LOGFILE
    return 0
  fi
  echo "[RUN] GPU$gpu $method target=$dom seed=$seed block=$blk" | tee -a $LOGFILE
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON $SCRIPT \
    --data_dir $DATA_DIR --output_dir $OUTDIR \
    --target_domain $dom --method $method --seed $seed \
    --vit_block $blk --steps 3000 --batch_size 32 --lr 5e-5 \
    >> $LOGFILE 2>&1
  if [ $? -eq 0 ]; then
    echo "[OK] GPU$gpu $method $dom seed$seed block$blk" | tee -a $LOGFILE
  else
    echo "[FAIL] GPU$gpu $method $dom seed$seed block$blk" | tee -a $LOGFILE
  fi
}

# Process tasks in pairs (one per GPU)
i=0
while [ $i -lt $TOTAL ]; do
  # GPU 0
  read -r m0 d0 s0 b0 <<< "${TASKS[$i]}"
  run_one 0 "$m0" "$d0" "$s0" "$b0" &
  PID0=$!
  i=$((i + 1))

  # GPU 1 (if tasks remain)
  PID1=""
  if [ $i -lt $TOTAL ]; then
    read -r m1 d1 s1 b1 <<< "${TASKS[$i]}"
    run_one 1 "$m1" "$d1" "$s1" "$b1" &
    PID1=$!
    i=$((i + 1))
  fi

  # Wait for both
  wait $PID0
  [ -n "$PID1" ] && wait $PID1
  echo "--- completed $i/$TOTAL at $(date) ---" | tee -a $LOGFILE
done

echo "All done: $TOTAL tasks at $(date)" | tee -a $LOGFILE
echo "=== RESULTS SUMMARY ===" | tee -a $LOGFILE
for method in $METHODS; do
  for dom in $DOMAINS; do
    accs=""
    for seed in $SEEDS; do
      f=$OUTDIR/${method}_${dom}_seed${seed}_block${BLOCK}.json
      if [ -f "$f" ]; then
        a=$($PYTHON -c "import json; print(json.load(open('$f'))['target_acc']*100)")
        accs="$accs $a"
      fi
    done
    if [ -n "$accs" ]; then
      avg=$($PYTHON -c "import numpy as np; print(f'{np.mean([$accs]):.2f}')")
      echo "$method $dom: $avg%" | tee -a $LOGFILE
    fi
  done
done
