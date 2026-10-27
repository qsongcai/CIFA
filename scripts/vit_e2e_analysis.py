#!/usr/bin/env python3
"""ViT E2E experiment statistical analysis."""
import json, glob, os, itertools
import numpy as np
from scipy import stats

RESULTS_DIR = "/home/user/Doubao/chats/1449328766471938/vit_e2e_results"
METHODS = ["erm", "swad", "cifa"]
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
SEEDS = [0, 1, 2, 3, 4]
BLOCKS_ABLATION = [3, 6, 9]
SEEDS_ABLATION = [0, 1, 2]
MARGIN = 1.0  # percentage points for TOST

def load(method, domain, seed, block=6):
    path = os.path.join(RESULTS_DIR, f"{method}_{domain}_seed{seed}_block{block}.json")
    with open(path) as f:
        return json.load(f)

def pct(x):
    return x * 100.0

# 1. Main experiment
print("=" * 80)
print("1. MAIN EXPERIMENT (block6): Target Domain Accuracy (%)")
print("=" * 80)
main_results = {}
for method in METHODS:
    main_results[method] = {}
    for domain in DOMAINS:
        accs = [pct(load(method, domain, s)["target_acc"]) for s in SEEDS]
        main_results[method][domain] = accs
        print(f"  {method:6s} {domain:14s}: {np.mean(accs):6.2f} +/- {np.std(accs, ddof=1):5.2f}  (seeds: {[f'{a:.2f}' for a in accs]})")

print("\n  --- Average across 4 domains ---")
for method in METHODS:
    per_domain_means = [np.mean(main_results[method][d]) for d in DOMAINS]
    flat = [a for sub in [main_results[method][d] for d in DOMAINS] for a in sub]
    print(f"  {method:6s}: mean-of-means={np.mean(per_domain_means):6.2f},  pooled={np.mean(flat):6.2f} +/- {np.std(flat, ddof=1):5.2f}")

# 2. Paired t-tests
print("\n" + "=" * 80)
print("2. PAIRED T-TESTS on target_acc (n=20 domain-seed pairs)")
print("=" * 80)
pairs = list(itertools.product(DOMAINS, SEEDS))
def get_pair_accs(method):
    return np.array([pct(load(method, d, s)["target_acc"]) for d, s in pairs])

for m1, m2 in [("cifa", "swad"), ("cifa", "erm"), ("swad", "erm")]:
    a1, a2 = get_pair_accs(m1), get_pair_accs(m2)
    diff = a1 - a2
    t, p = stats.ttest_rel(a1, a2)
    print(f"  {m1} vs {m2}: mean diff={np.mean(diff):+.3f}pp, sd={np.std(diff, ddof=1):.3f}, t({len(diff)-1})={t:+.3f}, p={p:.4f}, wins={np.sum(diff>0)}/{len(diff)}")

# 3. TOST
print("\n" + "=" * 80)
print(f"3. TOST EQUIVALENCE: CIFA vs SWAD (margin={MARGIN}pp, n=20)")
print("=" * 80)
a_cifa, a_swad = get_pair_accs("cifa"), get_pair_accs("swad")
diff = a_cifa - a_swad
mean_diff = np.mean(diff)
se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
df = len(diff) - 1
t_upper = (mean_diff - MARGIN) / se_diff   # tests H0: mean >= +MARGIN (upper bound)
t_lower = (mean_diff + MARGIN) / se_diff   # tests H0: mean <= -MARGIN (lower bound)
p_upper = stats.t.cdf(t_upper, df)         # upper-bound test is a LEFT-tail test
p_lower = stats.t.sf(t_lower, df)          # lower-bound test is a RIGHT-tail test
p_tost = max(p_upper, p_lower)
ci_low = mean_diff - stats.t.ppf(0.95, df) * se_diff
ci_high = mean_diff + stats.t.ppf(0.95, df) * se_diff
print(f"  mean diff={mean_diff:+.4f}pp, SE={se_diff:.4f}, df={df}")
print(f"  90% CI=[{ci_low:+.4f}, {ci_high:+.4f}]pp")
print(f"  upper t={t_upper:+.3f} p={p_upper:.6f}; lower t={t_lower:+.3f} p={p_lower:.6f}")
print(f"  TOST p={p_tost:.6f} -> Equivalent? {'YES' if p_tost < 0.05 else 'NO'}")
print(f"  90% CI within [-{MARGIN},+{MARGIN}]? {'YES' if (ci_low > -MARGIN and ci_high < MARGIN) else 'NO'}")

# 4. Own-worst-domain
# CALIBER: the reported own-worst-domain is, for each seed, the arm's accuracy on
# the target domain that arm itself finds hardest = min over the four target_acc.
# The JSON field `worst_holdout_acc` is the worst *source* holdout (a SWAD
# selection monitor near 96.6%) and is NOT the reported worst-domain metric.
print("\n" + "=" * 80)
print("4. OWN-WORST-DOMAIN ANALYSIS (per-seed min over target_acc)")
print("=" * 80)


def own_worst(method, s):
    accs = {d: pct(load(method, d, s)["target_acc"]) for d in DOMAINS}
    wd = min(accs, key=accs.get)
    return accs[wd], wd


for method in METHODS:
    vals = []
    dist = {d: 0 for d in DOMAINS}
    for s in SEEDS:
        v, wd = own_worst(method, s)
        vals.append(v)
        dist[wd] += 1
    print(f"  {method:6s}: own-worst={np.mean(vals):6.2f} +/- {np.std(vals, ddof=1):5.2f} "
          f"per-seed={[f'{v:.2f}' for v in vals]} worst-domain dist={dist}")

print("\n  --- paired CIFA-SWAD own-worst delta per seed ---")
deltas = []
for s in SEEDS:
    cv, cd = own_worst("cifa", s)
    sv, sd = own_worst("swad", s)
    dlt = cv - sv
    deltas.append(dlt)
    print(f"  seed{s}: CIFA {cv:6.2f}({cd[:4]}) - SWAD {sv:6.2f}({sd[:4]}) = {dlt:+.2f}")
deltas = np.array(deltas)
t, p = stats.ttest_1samp(deltas, 0.0)
print(f"  mean own-worst Δ={deltas.mean():+.2f} SD={deltas.std(ddof=1):.2f} "
      f"paired p={p:.4f} wins={int((deltas>0).sum())}/{len(deltas)}")

print("\n  --- [diagnostic, NOT reported] per-target worst SOURCE holdout ---")
for method in METHODS:
    accs = [pct(load(method, d, s)["worst_holdout_acc"])
            for d in DOMAINS for s in SEEDS]
    print(f"  {method:6s}: worst_holdout={np.mean(accs):6.2f} +/- {np.std(accs, ddof=1):5.2f}")

# 5. Layer ablation
print("\n" + "=" * 80)
print("5. LAYER ABLATION (sketch, CIFA, seeds 0-2)")
print("=" * 80)
block_data = {}
for block in BLOCKS_ABLATION:
    accs = [pct(load("cifa", "sketch", s, block)["target_acc"]) for s in SEEDS_ABLATION]
    block_data[block] = accs
    print(f"  block {block}: {np.mean(accs):6.2f} +/- {np.std(accs, ddof=1):5.2f}  (seeds: {[f'{a:.2f}' for a in accs]})")
f_stat, p_anova = stats.f_oneway(*[block_data[b] for b in BLOCKS_ABLATION])
print(f"\n  ANOVA: F(2,6)={f_stat:.3f}, p={p_anova:.4f}")

# 6. Ceiling check
print("\n" + "=" * 80)
print("6. CEILING EFFECT CHECK")
print("=" * 80)
for method in METHODS:
    all_target = [pct(load(method, d, s)["target_acc"]) for d in DOMAINS for s in SEEDS]
    all_holdout = []
    for d in DOMAINS:
        for s in SEEDS:
            all_holdout.extend([pct(v) for v in load(method, d, s)["holdout_acc"].values()])
    print(f"  {method:6s}: target [{min(all_target):.2f},{max(all_target):.2f}] >95%:{sum(a>95 for a in all_target)}/{len(all_target)}; holdout [{min(all_holdout):.2f},{max(all_holdout):.2f}] >95%:{sum(a>95 for a in all_holdout)}/{len(all_holdout)}")

# 7. CIFA diagnostics
print("\n" + "=" * 80)
print("7. CIFA AUGMENTATION DIAGNOSTICS (block6)")
print("=" * 80)
tf, gt, kl = [], [], []
for domain in DOMAINS:
    for s in SEEDS:
        d = load("cifa", domain, s)
        tf.append(d["aug_trigger_fraction"]); gt.append(d["aug_mean_gate"]); kl.append(d["aug_mean_consistency_kl"])
print(f"  trigger frac: {np.mean(tf):.4f} +/- {np.std(tf, ddof=1):.4f}")
print(f"  mean gate:    {np.mean(gt):.4f} +/- {np.std(gt, ddof=1):.4f}")
print(f"  mean KL:      {np.mean(kl):.6f} +/- {np.std(kl, ddof=1):.6f}")
