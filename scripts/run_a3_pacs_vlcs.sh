#!/usr/bin/env bash
# A3 alignment probe on TWO datasets, run back-to-back on the same GPUs:
#   PACS : positive case  (method works; Assumption 3 expected to hold)
#   VLCS : boundary case  (no mixable style structure; expected NOT to align)
# Each dataset is 4 targets x 10 seeds = 40 runs (80 total), then a3_analyze
# runs automatically per dataset (run_a3.sh calls it on completion).
#
# Usage:
#   scripts/run_a3_pacs_vlcs.sh [SEEDS] [M]
#     SEEDS : space-separated seed list (default "0 1 2 3 4 5 6 7 8 9")
#     M     : random augmentation views  (default 32)
#
# Honour the same environment overrides as run_a3.sh:
#   NGPU (default 2), WORKERS, PYTHON, DATA_DIR.
set -e

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEEDS=${1:-"0 1 2 3 4 5 6 7 8 9"}
M=${2:-32}
export NGPU=${NGPU:-2}

cd "$ROOT"
echo "=== A3 [1/2] PACS (positive) seeds='$SEEDS' M=$M NGPU=$NGPU ==="
bash scripts/run_a3.sh pacs "$SEEDS" "$M"

echo "=== A3 [2/2] VLCS (boundary) seeds='$SEEDS' M=$M NGPU=$NGPU ==="
bash scripts/run_a3.sh vlcs "$SEEDS" "$M"

echo "=== A3 PACS + VLCS ALL 80 RUNS DONE $(date '+%F %T') ==="
