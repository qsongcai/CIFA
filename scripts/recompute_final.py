#!/usr/bin/env python3
# ============================================================================
# Final n=10 recomputation for the CIFA paper (post external-review experiments)
#
#   Part 1  Four benchmarks, n=10 : CIFA - SWAD (overall / own-worst / per-domain)
#   Part 2  PACS, n=10            : CIFA - MixStyle/DSU (+/-SWAD), four arms
#   Part 3  ERM reference         : every method overall - ERM
#   Part 4  Hyperparameter sensitivity (3 seeds) : each results_sens_* vs base
#   Part 5  Backbone / budget (5 seeds) : ResNet-18 and 10,000-step CIFA-SWAD
#
# Independent unit = seed. Every paired endpoint reports:
#   - paired t mean/std, t 95% CI, t p; Wilcoxon signed-rank; sign test; pos seeds
#   - nonparametric paired-bootstrap 95% CI (B=20,000, np.random.RandomState(20261002))
# Holm step-down adjusted p for:
#   (i)  preregistered primary family : PACS overall + PACS worst (m=2)
#   (ii) full CIFA-SWAD family        : 4 benchmarks x overall+worst (m=8)
#   (iii) PACS baseline family        : 4 baselines x overall (m=4)
#
# All numbers are recomputed from the individual JSON; nothing is hand-entered.
# ============================================================================
import json, glob, os, re
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
BENCH = ["pacs", "vlcs", "officehome", "terra"]
BLABEL = {"pacs": "PACS", "vlcs": "VLCS",
          "officehome": "OfficeHome", "terra": "TerraIncognita"}
BASELINES = ["mixstyle", "dsu", "mixstyle_swad", "dsu_swad"]
B = 20000
BOOT_SEED = 20261002


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_dir(path):
    """Index {(domain, seed): target_acc in percent} from a results directory."""
    idx = {}
    for f in glob.glob(os.path.join(path, "*.json")):
        j = json.load(open(f))
        if "target_acc" not in j:
            continue
        idx[(j["target_domain"], int(j["seed"]))] = j["target_acc"] * 100.0
    return idx


# --------------------------------------------------------------------------
# Per-endpoint reporting
# --------------------------------------------------------------------------
def report(d):
    d = np.asarray(d, float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (m - tcrit * se, m + tcrit * se)
    t_p = stats.ttest_1samp(d, 0.0).pvalue
    nz = d[d != 0]
    w_p = stats.wilcoxon(nz).pvalue if len(nz) else 1.0
    pos = int((d > 0).sum())
    s_p = stats.binomtest(max(pos, n - pos), n, 0.5,
                          alternative="two-sided").pvalue
    boot = bootstrap_ci(d)
    return dict(n=n, mean=float(m), sd=float(sd),
                t_ci=[float(ci[0]), float(ci[1])], t_p=float(t_p),
                w_p=float(w_p), pos=pos, sign_p=float(s_p),
                boot_ci=[float(boot[0]), float(boot[1])])


def bootstrap_ci(d, B=B):
    """Paired nonparametric bootstrap, unit = seed; percentile 95% CI.

    A dedicated RandomState(BOOT_SEED) is used per endpoint so each CI is
    independently reproducible and order-independent.
    """
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.RandomState(BOOT_SEED)
    means = np.empty(B)
    for b in range(B):
        pick = rng.randint(0, n, size=n)
        means[b] = d[pick].mean()
    return np.percentile(means, [2.5, 97.5])


def paired(idxA, idxB):
    doms = sorted({k[0] for k in idxA} & {k[0] for k in idxB})
    seeds = sorted({k[1] for k in idxA} & {k[1] for k in idxB})
    ov, wo = [], []
    perdom = {dm: [] for dm in doms}
    for s in seeds:
        a = np.array([idxA[(dm, s)] for dm in doms])
        b = np.array([idxB[(dm, s)] for dm in doms])
        ov.append(a.mean() - b.mean())
        wo.append(a.min() - b.min())
        for dm in doms:
            perdom[dm].append(idxA[(dm, s)] - idxB[(dm, s)])
    return dict(doms=doms, seeds=seeds, overall=report(ov),
                worst=report(wo),
                perdom={dm: report(v) for dm, v in perdom.items()},
                ov_diffs=ov, wo_diffs=wo)


def fmt(tag, r):
    return (f"{tag:<26} {r['mean']:+6.2f} +/- {r['sd']:5.2f}  "
            f"tCI[{r['t_ci'][0]:+5.2f},{r['t_ci'][1]:+5.2f}]  "
            f"bootCI[{r['boot_ci'][0]:+5.2f},{r['boot_ci'][1]:+5.2f}]  "
            f"t={r['t_p']:.4f} W={r['w_p']:.4f} sign={r['sign_p']:.4f}  "
            f"{r['pos']}/{r['n']}")


# --------------------------------------------------------------------------
# Holm step-down
# --------------------------------------------------------------------------
def holm(items):
    """items: list of (name, p). Returns dict name -> adjusted p."""
    items = sorted(items, key=lambda x: x[1])
    m = len(items)
    adj = {}
    running = 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, p * (m - k)))
        adj[name] = running
    return adj


out = {}

# ==========================================================================
# PART 1  Four benchmarks n=10 : CIFA - SWAD
# ==========================================================================
print("=" * 90)
print("PART 1  FOUR BENCHMARKS n=10 : CIFA - SWAD (unit = seed)")
print("=" * 90)
p1 = {}
for b in BENCH:
    ci = load_dir(os.path.join(RES, b, "cifa"))
    sw = load_dir(os.path.join(RES, b, "swad"))
    r = paired(ci, sw)
    p1[b] = {k: r[k] for k in
             ["doms", "seeds", "overall", "worst", "perdom"]}
    print(f"\n--- {BLABEL[b]} (n={len(r['seeds'])}) domains={r['doms']}")
    print(fmt("overall", r["overall"]))
    print(fmt("own-worst", r["worst"]))
    for dm in r["doms"]:
        print(fmt(f"  {dm}", r["perdom"][dm]))
out["cifa_vs_swad"] = p1

# ==========================================================================
# PART 2  PACS n=10 : CIFA - each MixStyle/DSU baseline
# ==========================================================================
print("\n" + "=" * 90)
print("PART 2  PACS n=10 : CIFA - each MixStyle/DSU baseline (unit = seed)")
print("=" * 90)
p2 = {}
ci_pacs = load_dir(os.path.join(RES, "pacs", "cifa"))
for m in BASELINES:
    bi = load_dir(os.path.join(RES, "pacs", m))
    r = paired(ci_pacs, bi)
    p2[m] = {k: r[k] for k in
             ["doms", "seeds", "overall", "worst", "perdom"]}
    print(f"\n--- CIFA - {m} (n={len(r['seeds'])})")
    print(fmt("overall", r["overall"]))
    print(fmt("own-worst", r["worst"]))
    for dm in r["doms"]:
        print(fmt(f"  {dm}", r["perdom"][dm]))
out["cifa_vs_baselines"] = p2

# ==========================================================================
# PART 3  ERM reference : every method overall - ERM
# ==========================================================================
print("\n" + "=" * 90)
print("PART 3  ERM REFERENCE : each method overall - ERM (n=10, pp)")
print("=" * 90)
p3 = {}
for b in BENCH:
    er = load_dir(os.path.join(RES, "erm", b))
    rows = {}
    for sub in ["cifa", "swad"]:
        idx = load_dir(os.path.join(RES, b, sub))
        doms = sorted({k[0] for k in idx} & {k[0] for k in er})
        seeds = sorted({k[1] for k in idx} & {k[1] for k in er})
        delta = [np.mean([idx[(dm, s)] for dm in doms]) -
                 np.mean([er[(dm, s)] for dm in doms]) for s in seeds]
        rows[sub] = report(delta)
    if b == "pacs":
        for m in BASELINES:
            idx = load_dir(os.path.join(RES, "pacs", m))
            doms = sorted({k[0] for k in idx} & {k[0] for k in er})
            seeds = sorted({k[1] for k in idx} & {k[1] for k in er})
            delta = [np.mean([idx[(dm, s)] for dm in doms]) -
                     np.mean([er[(dm, s)] for dm in doms]) for s in seeds]
            rows[m] = report(delta)
    p3[b] = rows
    print(f"\n--- {BLABEL[b]} overall over ERM")
    for tag, r in rows.items():
        print(fmt(tag, r))
out["over_erm"] = p3

# ==========================================================================
# PART 4  Hyperparameter sensitivity (3 seeds) vs base
# ==========================================================================
print("\n" + "=" * 90)
print("PART 4  HYPERPARAMETER SENSITIVITY (3 seeds) : overall vs base")
print("=" * 90)
GROUPS = [
    ("base", ["base"]),
    ("p (cifa_prob)", ["p01", "p025", "p10"]),
    ("delta (consistency_delta)", ["d05", "d10", "d80"]),
    ("beta (soft_eta)", ["b00", "b025", "b10"]),
    ("layer (cifa_layers)", ["L1", "L2", "L123"]),
    ("mixing range (lambda)", ["rNarrow", "rMid", "rFull"]),
]


def overall_by_seed(path):
    idx = load_dir(path)
    doms = sorted({k[0] for k in idx})
    seeds = sorted({k[1] for k in idx})
    per_seed = {s: float(np.mean([idx[(dm, s)] for dm in doms])) for s in seeds}
    return per_seed


base_seed = overall_by_seed(os.path.join(RES, "sens", "results_sens_base"))
base_mean = float(np.mean(list(base_seed.values())))
p4 = {"base_overall_mean": base_mean,
      "base_overall_by_seed": base_seed, "groups": {}}
print(f"\nbase overall mean = {base_mean:.2f}  by seed = "
      + ", ".join(f"{base_seed[s]:.2f}" for s in sorted(base_seed)))
for gname, tags in GROUPS:
    p4["groups"][gname] = {}
    print(f"\n[{gname}]")
    for tag in tags:
        ps = overall_by_seed(os.path.join(RES, "sens", f"results_sens_{tag}"))
        seeds = sorted(ps)
        mean = float(np.mean([ps[s] for s in seeds]))
        # paired per-seed delta vs base
        dseeds = sorted(set(ps) & set(base_seed))
        dd = [ps[s] - base_seed[s] for s in dseeds]
        entry = dict(overall_mean=mean, overall_by_seed=ps,
                     delta_vs_base=float(mean - base_mean),
                     delta_per_seed=dd,
                     delta_min=float(min(dd)), delta_max=float(max(dd)))
        p4["groups"][gname][tag] = entry
        print(f"  {tag:<8} overall={mean:6.2f}  dvsBase={mean-base_mean:+5.2f}"
              f"  per-seed d=[{', '.join(f'{x:+.2f}' for x in dd)}]")
out["sensitivity"] = p4

# ==========================================================================
# PART 5  Backbone / budget (5 seeds)
# ==========================================================================
print("\n" + "=" * 90)
print("PART 5  BACKBONE / BUDGET (5 seeds) : CIFA - SWAD")
print("=" * 90)
p5 = {}
conditions = [
    ("r18", "results_r18_cifa", "results_r18_swad"),
    ("s10k", "results_s10k_cifa", "results_s10k_swad"),
]
# default R50 / 5000-step, restricted to seeds 0..4 for a matched context row
r50_ci = load_dir(os.path.join(RES, "pacs", "cifa"))
r50_sw = load_dir(os.path.join(RES, "pacs", "swad"))
r50_ci = {k: v for k, v in r50_ci.items() if k[1] <= 4}
r50_sw = {k: v for k, v in r50_sw.items() if k[1] <= 4}
r_def = paired(r50_ci, r50_sw)
p5["r50_5k_default"] = {k: r_def[k] for k in
                        ["seeds", "overall", "worst"]}
print("\n--- default ResNet-50 / 5,000 (n=5, context row)")
print(fmt("overall", r_def["overall"]))
print(fmt("own-worst", r_def["worst"]))
for cname, cdir, sdir in conditions:
    ci = load_dir(os.path.join(RES, "robustness", cdir))
    sw = load_dir(os.path.join(RES, "robustness", sdir))
    r = paired(ci, sw)
    p5[cname] = {k: r[k] for k in ["seeds", "overall", "worst", "perdom"]}
    label = "ResNet-18" if cname == "r18" else "ResNet-50 / 10,000"
    print(f"\n--- {label} (n={len(r['seeds'])})")
    print(fmt("overall", r["overall"]))
    print(fmt("own-worst", r["worst"]))
out["backbone_budget"] = p5

# ==========================================================================
# HOLM CORRECTIONS
# ==========================================================================
print("\n" + "=" * 90)
print("HOLM STEP-DOWN CORRECTIONS")
print("=" * 90)
# (i) preregistered primary family: PACS overall + worst
primary_items = [
    ("PACS overall", p1["pacs"]["overall"]["t_p"]),
    ("PACS worst", p1["pacs"]["worst"]["t_p"]),
]
holm_primary = holm(primary_items)
# (ii) full CIFA-SWAD family: 4 benchmarks x overall+worst
full_items = []
for b in BENCH:
    full_items.append((f"{BLABEL[b]} overall", p1[b]["overall"]["t_p"]))
    full_items.append((f"{BLABEL[b]} worst", p1[b]["worst"]["t_p"]))
holm_full = holm(full_items)
# (iii) PACS baseline family: 4 baselines x overall
base_items = [(f"CIFA-{m} overall", p2[m]["overall"]["t_p"])
              for m in BASELINES]
holm_base = holm(base_items)

holm_out = {"preregistered_primary_m2": holm_primary,
            "full_cifa_swad_m8": holm_full,
            "pacs_baselines_m4": holm_base}
out["holm"] = holm_out

print("\n[preregistered primary family, m=2]")
for k, v in holm_primary.items():
    print(f"  {k:<16} adj p = {v:.4f}")
print("\n[full CIFA-SWAD family, m=8]")
for k, v in sorted(holm_full.items(), key=lambda x: x[1]):
    print(f"  {k:<22} adj p = {v:.4f}")
print("\n[PACS baseline family, m=4]")
for k, v in sorted(holm_base.items(), key=lambda x: x[1]):
    print(f"  {k:<26} adj p = {v:.4f}")

# ==========================================================================
# SAVE
# ==========================================================================
tables = os.path.join(RES, "tables")
os.makedirs(tables, exist_ok=True)


def default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (list, tuple)):
        return list(o)
    raise TypeError(f"not serializable: {type(o)}")


json.dump({k: out[k] for k in
           ["cifa_vs_swad", "cifa_vs_baselines", "over_erm", "holm"]},
          open(os.path.join(tables, "paired_stats.json"), "w"),
          indent=2, default=default)
json.dump(out["sensitivity"],
          open(os.path.join(tables, "sensitivity_summary.json"), "w"),
          indent=2, default=default)
json.dump(out["backbone_budget"],
          open(os.path.join(tables, "budget_robustness_summary.json"), "w"),
          indent=2, default=default)
print("\nSaved tables/paired_stats.json, sensitivity_summary.json, "
      "budget_robustness_summary.json")
