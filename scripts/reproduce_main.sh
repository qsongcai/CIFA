#!/usr/bin/env bash
# One-command reproduction of the main comparison on all four benchmarks.
# For each benchmark it runs the SWAD control and then CIFA, leave-one-domain-out,
# over seeds 0..4 (override with SEEDS). Results land in reproduce_output/.
#
#   DATA_DIR must be set per dataset if the four datasets live in different
#   roots; see the per-dataset DATA_DIR_* overrides below.
#
# Example (single GPU, datasets under /data):
#   NGPU=1 \
#   DATA_DIR_pacs=/data/PACS DATA_DIR_vlcs=/data/VLCS \
#   DATA_DIR_officehome=/data/OfficeHome DATA_DIR_terra=/data/terra_imagefolder \
#   bash scripts/reproduce_main.sh
set -e

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEEDS=${SEEDS:-"0 1 2 3 4"}

for ds in pacs vlcs officehome terra; do
    var="DATA_DIR_$ds"
    if [ -n "${!var}" ]; then
        export DATA_DIR=${!var}
    fi
    echo "########## $ds : SWAD control ##########"
    "$ROOT/scripts/run_sweep.sh" "$ds" swad "$SEEDS"
    echo "########## $ds : CIFA ##########"
    "$ROOT/scripts/run_sweep.sh" "$ds" cifa "$SEEDS"
done

echo
echo "=== Main sweeps finished. Build the comparison table with:"
echo "    python scripts/collect_tables.py --root reproduce_output"
