"""Aggregate result JSON files into DomainBed-style tables.

Prints one table per (dataset, method): per-target-domain mean over seeds
and the overall average, for target accuracy and worst source-domain
hold-out accuracy. The hold-out column is computed on source domains only
(the target domain is never used for selection).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from datasets import DATASETS


def domain_order(dataset, domains):
    """Registry order for known datasets; alphabetical fallback."""
    if dataset in DATASETS:
        order = DATASETS[dataset]["domains"]
        return sorted(domains, key=lambda d: order.index(d))
    return sorted(domains)


def print_table(dataset, method, by_domain):
    header = (f"[{dataset}/{method}] {'target domain':<16}{'seeds':>6}"
              f"{'acc (mean±std)':>18}{'worst src-holdout':>20}")
    print(header)
    print("-" * len(header))
    all_acc, all_wh = [], []
    for dom in domain_order(dataset, set(by_domain)):
        runs = by_domain[dom]
        accs = [r["target_acc"] * 100 for r in runs]
        whs = [r["worst_holdout_acc"] * 100 for r in runs]
        mean = sum(accs) / len(accs)
        std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        whm = sum(whs) / len(whs)
        print(f"{dom:<16}{len(accs):>6}{mean:>13.2f}±{std:<3.2f}{whm:>19.2f}")
        all_acc += accs
        all_wh += whs
    if all_acc:
        print("-" * len(header))
        print(f"{'OVERALL':<16}{len(all_acc):>6}"
              f"{sum(all_acc)/len(all_acc):>18.2f}"
              f"{sum(all_wh)/len(all_wh):>20.2f}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="results")
    args = p.parse_args()

    # tables[(dataset, method)][domain] -> list of run records
    tables = defaultdict(lambda: defaultdict(list))
    for fn in sorted(glob.glob(os.path.join(args.output_dir, "*.json"))):
        with open(fn) as f:
            r = json.load(f)
        # method from the record, falling back to the filename prefix
        method = r.get("method", os.path.basename(fn).split("_")[0])
        # dataset: prefer the record field; older records fall back to 'pacs'
        dataset = r.get("dataset", "pacs")
        tables[(dataset, method)][r["target_domain"]].append(r)

    for key in sorted(tables):
        print_table(key[0], key[1], tables[key])

    print("Note: 'worst src-holdout' is the worst accuracy over the "
          "source-domain 20% hold-out splits (model selection only); "
          "it is not an out-of-domain number.")


if __name__ == "__main__":
    main()
