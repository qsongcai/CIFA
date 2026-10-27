#!/usr/bin/env bash
# A3 alignment probe: test Assumption 3 (trajectory--augmentation alignment).
#
# The probe trains CIFA under LODO and compares three ensembles on the target:
#   TE  trajectory checkpoints inside the SWAD window
#   AE  M random augmentation views of the final averaged model
#   hbar the dense weight average (Theorem 2 sanity: hbar ~ TE)
#
# Usage:
#   scripts/run_a3.sh [DATASET] [SEEDS] [M]
#     DATASET : pacs | vlcs | officehome | terra   (default pacs)
#     SEEDS   : space-separated seed list          (default "0 1 2 3 4 5 6 7 8 9")
#     M       : number of random augmentation views (default 32)
#
# Environment overrides (same as run_sweep.sh):
#   DATA_DIR  dataset root on this machine (recommended)
#   NGPU      GPUs to round-robin (default 1)
#   WORKERS   dataloader workers per job (default 4)
#   PYTHON    interpreter (default "python")
#
# Outputs go to reproduce_output/a3/DATASET (never results/, which holds the
# paper's reported numbers). The averaging window is chosen from source holdout
# loss only; the repeated target evaluations are post-hoc diagnostics.
set -e

DATASET=${1:-pacs}
SEEDS=${2:-"0 1 2 3 4 5 6 7 8 9"}
M=${3:-32}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CIFA=$ROOT/cifa
PYTHON=${PYTHON:-python}
NGPU=${NGPU:-1}
WORKERS=${WORKERS:-4}
CONFIG=$ROOT/configs/$DATASET.yaml

case "$DATASET" in
    pacs)       IFS=' ' read -ra DOMS <<< "art_painting cartoon photo sketch" ;;
    vlcs)       IFS=' ' read -ra DOMS <<< "CALTECH LABELME PASCAL SUN" ;;
    officehome) IFS=',' read -ra DOMS <<< "Art,Clipart,Product,Real World" ;;
    terra)      IFS=',' read -ra DOMS <<< "location_38,location_43,location_46,location_100" ;;
    *) echo "unknown DATASET=$DATASET"; exit 1 ;;
esac

OUTDIR=$ROOT/reproduce_output/a3/$DATASET
mkdir -p "$OUTDIR"
if [ -n "${DATA_DIR:-}" ]; then
    DATA_FLAG=(--data_dir "$DATA_DIR")
else
    DATA_FLAG=()
    echo "WARNING: DATA_DIR not set; falling back to config/registry path."
fi

echo ">>> A3 dataset=$DATASET seeds='$SEEDS' M=$M ngpu=$NGPU out=$OUTDIR"
cd "$CIFA"
i=0
for dom in "${DOMS[@]}"; do
    for seed in $SEEDS; do
        gpu=$((i % NGPU))
        echo ">>> GPU$gpu target='$dom' seed=$seed"
        CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" a3_probe.py \
            --dataset "$DATASET" --target_domain "$dom" --seed "$seed" \
            --workers "$WORKERS" --output_dir "$OUTDIR" --a3_m "$M" \
            --config "$CONFIG" "${DATA_FLAG[@]}" &
        i=$((i + 1))
        while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do wait -n; done
    done
done
wait

cd "$ROOT"
"$PYTHON" scripts/a3_analyze.py --a3_dir "$OUTDIR" --dataset "$DATASET"
