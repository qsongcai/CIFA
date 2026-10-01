# Results

This directory holds the paper's **reported** runs — the authoritative numbers
the tables are built from. They are provided so a reviewer can audit the claims
without re-running anything.

## Layout

```
results/
├── pacs/{cifa,swad}/        # 20 JSON each (4 domains x 5 seeds)
├── vlcs/{cifa,swad}/
├── officehome/{cifa,swad}/
├── terra/{cifa,swad}/
├── erm/<dataset>/           # ERM controls, 20 JSON each
├── ablation/<variant>/      # 5 variants x 20 JSON (PACS)
└── tables/
    ├── main_results.md          # regenerated from the run files
    ├── ablation_results.md
    ├── summary_*.json           # machine-readable per-dataset summaries
    └── three_arm_recomputed.json
```

Total: 340 per-run JSON files (160 main CIFA/SWAD + 80 ERM + 100 ablation).

## Regenerate the tables

From the repository root:

```bash
python scripts/collect_tables.py --root results
```

The Markdown tables and every mean/std/Δ in them are recomputed from the
individual JSON; nothing is hand-entered. The machine-readable
`summary_*.json` are kept for cross-checking but are not consumed when
rebuilding the tables.

## Your own runs

Re-runs go to `reproduce_output/` (created by the scripts, git-ignored), never
into this directory, so the reported numbers stay unchanged. Compare with

```bash
python scripts/collect_tables.py --root results \
    --compare-root reproduce_output
```
