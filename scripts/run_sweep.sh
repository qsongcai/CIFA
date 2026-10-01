#!/usr/bin/env bash
# Generic leave-one-domain-out sweep for one dataset and one method.
#
# Usage:
#   scripts/run_sweep.sh DATASET MODE [SEEDS]
#     DATASET : pacs | vlcs | officehome | terra
#     MODE    : cifa (CIFA + SWAD) | swad (SWAD control) | erm (ERM control)
#     SEEDS   : space-separated seed list (default "0 1 2 3 4")
#
# Environment overrides:
#   DATA_DIR  root of the dataset on this machine (recommended; otherwise the
#             data_dir inside the YAML config / registry default is used)
#   NGPU      number of GPUs to round-robin (default 1)
#   WORKERS   dataloader workers per job (default 4)
#   PYTHON    python interpreter (default "python")
#
# Re-runs are written to reproduce_output/ (never over results/, which holds
# the paper's reported numbers).
set -e

DATASET=${1:?usage: run_sweep.sh DATASET MODE [SEEDS]}
MODE=${2:?usage: run_sweep.sh DATASET MODE [SEEDS]}
SEEDS=${3:-"0 1 2 3 4"}

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

case "$MODE" in
    cifa) FLAGS=(--config "$CONFIG");                  OUT=cifa ;;
    swad) FLAGS=(--config "$CONFIG" --no_cifa);        OUT=swad ;;
    erm)  FLAGS=(--config "$CONFIG" --no_cifa --no_swad); OUT=erm ;;
    *) echo "unknown MODE=$MODE (expected cifa|swad|erm)"; exit 1 ;;
esac

OUTDIR=$ROOT/reproduce_output/$DATASET/$OUT
mkdir -p "$OUTDIR"
if [ -n "$DATA_DIR" ]; then
    DATA_FLAG=(--data_dir "$DATA_DIR")
else
    DATA_FLAG=()
    echo "WARNING: DATA_DIR not set; falling back to the config/registry path."
fi

echo ">>> dataset=$DATASET mode=$MODE seeds='$SEEDS' ngpu=$NGPU out=$OUTDIR"
cd "$CIFA"
i=0
for dom in "${DOMS[@]}"; do
    for seed in $SEEDS; do
        gpu=$((i % NGPU))
        echo ">>> GPU$gpu target='$dom' seed=$seed"
        CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" train.py \
            --dataset "$DATASET" --target_domain "$dom" --seed "$seed" \
            --workers "$WORKERS" --output_dir "$OUTDIR" \
            "${DATA_FLAG[@]}" "${FLAGS[@]}" &
        i=$((i + 1))
        while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do wait -n; done
    done
done
wait
"$PYTHON" aggregate.py --output_dir "$OUTDIR"
