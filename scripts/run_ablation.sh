#!/usr/bin/env bash
# PACS component ablation (pre-registered):
#   5 variants (full, fullmoment, noguard, uniform, noswad)
#   x 4 target domains x seeds 0..4 = 100 runs.
# Hparams are frozen; no target-domain data is used. Re-runs go to
# reproduce_output/ablation/<variant>/ and never touch results/.
#
# Environment: DATA_DIR (PACS root), SEEDS, NGPU, WORKERS, PYTHON.
set -e

SEEDS=${SEEDS:-"0 1 2 3 4"}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CIFA=$ROOT/cifa
PYTHON=${PYTHON:-python}
NGPU=${NGPU:-1}
WORKERS=${WORKERS:-4}
PDATA=/mnt/sdd1/datasets_qsong/PACS/Homework3-PACS-master/PACS

VARIANTS=(full fullmoment noguard uniform noswad)
IFS=' ' read -ra DOMS <<< "art_painting cartoon photo sketch"

if [ -n "$DATA_DIR" ]; then PDATA=$DATA_DIR; fi

cd "$CIFA"
i=0
for seed in $SEEDS; do
    for dom in "${DOMS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            gpu=$((i % NGPU))
            OUTDIR=$ROOT/reproduce_output/ablation/$variant
            mkdir -p "$OUTDIR"
            echo ">>> GPU$gpu variant=$variant target='$dom' seed=$seed"
            CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" train.py \
                --dataset pacs --data_dir "$PDATA" --target_domain "$dom" \
                --seed "$seed" --workers "$WORKERS" \
                --config "$ROOT/configs/ablation/$variant.yaml" \
                --variant "$variant" --output_dir "$OUTDIR" &
            i=$((i + 1))
            while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do wait -n; done
        done
    done
done
wait
echo "=== Ablation runs finished. Build the table with:"
echo "    python scripts/collect_tables.py --root reproduce_output --ablation"
