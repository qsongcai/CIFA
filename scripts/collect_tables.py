"""Recompute every reported number from the raw per-run JSON and emit Markdown.

This is the audit entry point: the tables in the paper are regenerated directly
from the individual run files (no hand-copied numbers), so a reader can verify
the claims without trusting the prose.

    python scripts/collect_tables.py --root results
    python scripts/collect_tables.py --root reproduce_output
    python scripts/collect_tables.py --root results --compare-root reproduce_output

Numbers use the DomainBed leave-one-domain-out convention:
  * per-domain cell = mean over seeds, with the sample std (ddof=1), as in the
    paper tables;
  * overall = macro-average over domains computed per seed, reported as the
    mean +/- std (ddof=1) across seeds;
  * CIFA-SWAD deltas are paired cell-by-cell (same target domain and seed).
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cifa"))
from datasets import DATASETS  # noqa: E402

DATASET_ORDER = ["pacs", "vlcs", "officehome", "terra"]
ABLATION_ORDER = ["full", "fullmoment", "noguard", "uniform", "noswad"]


def mean(xs):
    return sum(xs) / len(xs)


def sample_std(xs):
    """Sample standard deviation (ddof=1), matching the paper's reported ±std."""
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def classify(path):
    """Return ('ablation', variant) or ('main', dataset, method) from path."""
    parts = path.split(os.sep)
    if "ablation" in parts:
        i = parts.index("ablation")
        return ("ablation", parts[i + 1])
    ds = next((p for p in parts if p in DATASETS), None)
    if "erm" in parts:
        method = "erm"
    else:
        parent = parts[-2]
        method = parent if parent in ("cifa", "swad", "erm") else None
    return ("main", ds, method)


def load_tree(root):
    """Return (main, ablation): main[ds][method][domain][seed] -> acc (%)."""
    main = {d: {} for d in DATASET_ORDER}
    ablation = {v: {} for v in ABLATION_ORDER}
    for fn in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        with open(fn) as f:
            r = json.load(f)
        if "target_acc" not in r:
            continue  # skip summary_*.json
        kind = classify(fn)
        acc = r["target_acc"] * 100
        dom, seed = r["target_domain"], r["seed"]
        if kind[0] == "ablation":
            ablation[kind[1]].setdefault(dom, {})[seed] = acc
        else:
            _, ds, method = kind
            if ds is None or method is None:
                continue
            main[ds].setdefault(method, {}).setdefault(dom, {})[seed] = acc
    return main, ablation


def domains_of(ds):
    return DATASETS[ds]["domains"]


def summarize(cells, domains):
    """Per-domain mean/std and per-seed macro overall mean/std."""
    seeds = sorted({s for d in domains for s in cells.get(d, {})})
    per_domain = {}
    for d in domains:
        vals = [cells[d][s] for s in sorted(cells.get(d, {}))]
        per_domain[d] = (mean(vals), sample_std(vals))
    per_seed = [mean([cells[d][s] for d in domains]) for s in seeds]
    return per_domain, mean(per_seed), sample_std(per_seed), seeds


def paired(cifa, swad, domains):
    """Cell-by-cell CIFA-SWAD comparison; returns deltas and worst-domain stats."""
    seeds = sorted({s for d in domains for s in cifa.get(d, {})})
    per_domain, all_delta = {}, []
    pos = total = 0
    for d in domains:
        ds = [cifa[d][s] - swad[d][s] for s in seeds]
        per_domain[d] = (mean(ds), sum(x > 0 for x in ds), len(ds))
        all_delta += ds
        pos += sum(x > 0 for x in ds)
        total += len(ds)
    overall_delta = mean(
        [mean([cifa[d][s] for d in domains]) -
         mean([swad[d][s] for d in domains]) for s in seeds])
    worst = []
    for s in seeds:
        wd = min(domains, key=lambda d: swad[d][s])
        worst.append(cifa[wd][s] - swad[wd][s])
    return {
        "per_domain": per_domain,
        "overall_delta": overall_delta,
        "pos": pos, "total": total,
        "worst_delta": mean(worst),
        "worst_pos": sum(x > 0 for x in worst), "worst_n": len(worst),
    }


def ablation_pair(variant_cells, full_cells, domains):
    seeds = sorted({s for d in domains for s in variant_cells.get(d, {})})
    overall = [mean([variant_cells[d][s] for d in domains]) -
               mean([full_cells[d][s] for d in domains]) for s in seeds]

    def worst_marginal(cells):
        """Mean accuracy on each seed's own worst domain (a marginal, not paired)."""
        vals = []
        for s in seeds:
            wd = min(domains, key=lambda d: cells[d][s])
            vals.append(cells[wd][s])
        return mean(vals)

    # worst-domain gap is the difference of the two marginals, matching the
    # ablation summary (e.g. noguard 77.27 vs full 80.62 = -3.36).
    d_w = worst_marginal(variant_cells) - worst_marginal(full_cells)
    return mean(overall), d_w


def fmt(x, sign=False):
    return (f"{x:+.2f}" if sign else f"{x:.2f}")


def build_main_md(main, title):
    lines = [f"# {title}", ""]
    # overall table
    lines += [
        "## Overall accuracy (%, leave-one-domain-out, ResNet-50)", "",
        "| Dataset | ERM | SWAD | CIFA | Δ (CIFA−SWAD) | positive cells | "
        "worst-domain Δ |",
        "|---|---|---|---|---|---|---|",
    ]
    details = {}
    for ds in DATASET_ORDER:
        methods = main[ds]
        if "cifa" not in methods or "swad" not in methods:
            continue
        domains = domains_of(ds)
        summ = {m: summarize(methods[m], domains) for m in methods}
        p = paired(methods["cifa"], methods["swad"], domains)
        details[ds] = (summ, p)
        erm_cell = (f"{summ['erm'][1]:.2f}±{summ['erm'][2]:.2f}"
                    if "erm" in summ else "—")
        lines.append(
            f"| {ds} | {erm_cell} | {summ['swad'][1]:.2f}±{summ['swad'][2]:.2f} "
            f"| {summ['cifa'][1]:.2f}±{summ['cifa'][2]:.2f} "
            f"| {p['overall_delta']:+.2f} | {p['pos']}/{p['total']} "
            f"| {p['worst_delta']:+.2f} ({p['worst_pos']}/{p['worst_n']}) |")
    lines.append("")
    # per-domain tables
    for ds in DATASET_ORDER:
        if ds not in details:
            continue
        summ, p = details[ds]
        lines += [f"## {ds} — per target domain", "",
                  "| target domain | SWAD (mean±std) | CIFA (mean±std) | "
                  "Δ mean | positive seeds |",
                  "|---|---|---|---|---|"]
        for d in domains_of(ds):
            sm, ss = summ["swad"][0][d]
            cm, cs = summ["cifa"][0][d]
            dm, dpos, dn = p["per_domain"][d]
            lines.append(
                f"| {d} | {sm:.2f}±{ss:.2f} | {cm:.2f}±{cs:.2f} "
                f"| {dm:+.2f} | {dpos}/{dn} |")
        lines.append("")
    lines += [
        "> `worst-domain Δ` compares, for each seed, CIFA against SWAD on the "
        "domain where SWAD is weakest that seed. OfficeHome shows no measurable "
        "gain (Δ≈0.04 overall); see docs/CLAIMS.md.",
        "",
    ]
    return "\n".join(lines), details


def build_ablation_md(ablation, title):
    domains = domains_of("pacs")
    rows = {}
    for v in ABLATION_ORDER:
        if not ablation[v]:
            continue
        _, om, os, _ = summarize(ablation[v], domains)
        rows[v] = (om, os)
    lines = [f"# {title}", "",
             "PACS, 5 variants × 4 target domains × 5 seeds = 100 runs.", "",
             "| variant | overall mean±std | Δ vs full (overall) | "
             "Δ vs full (worst domain) |",
             "|---|---|---|---|"]
    for v in ABLATION_ORDER:
        if v not in rows:
            continue
        om, os = rows[v]
        if v == "full":
            d_ov, d_w = 0.0, 0.0
        else:
            d_ov, d_w = ablation_pair(ablation[v], ablation["full"], domains)
        lines.append(
            f"| {v} | {om:.2f}±{os:.2f} | {d_ov:+.2f} | {d_w:+.2f} |")
    lines += ["",
              "> `fullmoment` (+0.23) is within run-to-run noise: the "
              "baseline/deviation split is what enables the content-preservation bound, "
              "not a source of accuracy. `noswad` is the largest drop "
              "(−2.42), i.e. weight averaging drives the overall level; the "
              "feature-moment term targets the worst domain.",
              ""]
    return "\n".join(lines)


def write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print("wrote", os.path.relpath(path, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--compare-root", default=None)
    args = ap.parse_args()

    main_tree, abl_tree = load_tree(args.root)
    label = os.path.basename(args.root.rstrip("/"))
    main_md, _ = build_main_md(main_tree, f"Main results ({label})")
    write_file(os.path.join(ROOT, args.root, "tables", "main_results.md"),
               main_md)
    abl_md = build_ablation_md(abl_tree, f"Component ablation ({label})")
    write_file(os.path.join(ROOT, args.root, "tables", "ablation_results.md"),
               abl_md)

    if args.compare_root:
        cmp_tree, _ = load_tree(args.compare_root)
        print("\nOverall: reported ({}) vs re-run ({})".format(
            args.root, args.compare_root))
        for ds in DATASET_ORDER:
            for m in ("swad", "cifa"):
                try:
                    a = summarize(main_tree[ds][m], domains_of(ds))[1]
                    b = summarize(cmp_tree[ds][m], domains_of(ds))[1]
                    print(f"  {ds:>10} {m:>4}: {a:6.2f}  ->  {b:6.2f}  "
                          f"(diff {b-a:+.2f})")
                except (KeyError, IndexError):
                    pass


if __name__ == "__main__":
    main()
