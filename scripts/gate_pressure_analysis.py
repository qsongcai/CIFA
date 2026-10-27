#!/usr/bin/env python3
"""Gate pressure experiment analysis."""
import json, os, itertools
import numpy as np
from scipy import stats

BASE = "/home/user/Doubao/chats/1449328766471938/gate_pressure_results"
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
SEEDS = [0, 1, 2]
DELTAS = ["2.0", "0.05", "0.025", "0.01", "0.005", "0.001"]

def load(delta, domain, seed, prefix="d"):
    path = os.path.join(BASE, f"{prefix}{delta}", f"cifa_{domain}_seed{seed}.json")
    with open(path) as f:
        return json.load(f)

def pct(x): return x * 100.0

# ============================================================
# 1. Delta sweep: gate activation & performance
# ============================================================
print("=" * 90)
print("1. DELTA SWEEP (default augmentation): Gate Activation & Performance")
print("=" * 90)
print(f"{'delta':>8s} {'mean_gate':>10s} {'mean_KL':>10s} {'trigger%':>9s} {'target%':>9s} {'worst%':>9s} {'target_sd':>10s}")
print("-" * 90)

delta_data = {}
for delta in DELTAS:
    gates, kls, triggers, targets, worsts = [], [], [], [], []
    for dom in DOMAINS:
        for s in SEEDS:
            d = load(delta, dom, s)
            gates.append(d["aug_mean_gate"])
            kls.append(d["aug_mean_consistency_kl"])
            triggers.append(d["aug_trigger_fraction"])
            targets.append(pct(d["target_acc"]))
            worsts.append(pct(d["worst_holdout_acc"]))
    delta_data[delta] = {"gate": gates, "kl": kls, "target": targets, "worst": worsts}
    print(f"{delta:>8s} {np.mean(gates):>10.4f} {np.mean(kls):>10.4f} {np.mean(triggers)*100:>8.1f}% {np.mean(targets):>9.2f} {np.mean(worsts):>9.2f} {np.std(targets, ddof=1):>10.2f}")

# Per-domain target accuracy for each delta
print("\n  --- Per-domain target accuracy (%) ---")
print(f"{'delta':>8s} {'art':>8s} {'cartoon':>8s} {'photo':>8s} {'sketch':>8s} {'avg':>8s}")
for delta in DELTAS:
    row = []
    for dom in DOMAINS:
        accs = [pct(load(delta, dom, s)["target_acc"]) for s in SEEDS]
        row.append(np.mean(accs))
    print(f"{delta:>8s} {row[0]:>8.2f} {row[1]:>8.2f} {row[2]:>8.2f} {row[3]:>8.2f} {np.mean(row):>8.2f}")

# Gate activation detail: how many runs have gate < 1.0
print("\n  --- Gate activation detail (runs with gate < 1.0) ---")
for delta in DELTAS:
    gates = delta_data[delta]["gate"]
    n_active = sum(1 for g in gates if g < 0.999)
    print(f"  delta={delta:>6s}: mean_gate={np.mean(gates):.4f}, min={min(gates):.4f}, runs with gate<0.999: {n_active}/{len(gates)}")

# ============================================================
# 2. Strong augmentation: with gate vs without gate
# ============================================================
print("\n" + "=" * 90)
print("2. STRONG AUGMENTATION: With Gate (d=0.01) vs Without Gate")
print("   (strong: lambda=[0,0.5], layers=1,2,3, prob=1.0)")
print("=" * 90)

def load_strong(name, domain, seed):
    path = os.path.join(BASE, name, f"cifa_{domain}_seed{seed}.json")
    with open(path) as f:
        return json.load(f)

print(f"{'setting':>20s} {'mean_gate':>10s} {'mean_KL':>10s} {'trigger%':>9s} {'target%':>9s} {'worst%':>9s}")
print("-" * 90)
for name in ["strong_d0.01", "strong_nogate"]:
    gates, kls, triggers, targets, worsts = [], [], [], [], []
    for dom in DOMAINS:
        for s in SEEDS:
            d = load_strong(name, dom, s)
            gates.append(d["aug_mean_gate"])
            kls.append(d["aug_mean_consistency_kl"])
            triggers.append(d["aug_trigger_fraction"])
            targets.append(pct(d["target_acc"]))
            worsts.append(pct(d["worst_holdout_acc"]))
    print(f"{name:>20s} {np.mean(gates):>10.4f} {np.mean(kls):>10.4f} {np.mean(triggers)*100:>8.1f}% {np.mean(targets):>9.2f} {np.mean(worsts):>9.2f}")

# Per-domain strong comparison
print("\n  --- Per-domain target accuracy (%) ---")
print(f"{'domain':>14s} {'with_gate':>10s} {'no_gate':>10s} {'delta':>8s}")
for dom in DOMAINS:
    g = [pct(load_strong("strong_d0.01", dom, s)["target_acc"]) for s in SEEDS]
    ng = [pct(load_strong("strong_nogate", dom, s)["target_acc"]) for s in SEEDS]
    print(f"{dom:>14s} {np.mean(g):>10.2f} {np.mean(ng):>10.2f} {np.mean(g)-np.mean(ng):>+8.2f}")

# Paired test: strong with gate vs no gate
print("\n  --- Paired test (12 domain-seed pairs) ---")
g_targets = [pct(load_strong("strong_d0.01", dom, s)["target_acc"]) for dom in DOMAINS for s in SEEDS]
ng_targets = [pct(load_strong("strong_nogate", dom, s)["target_acc"]) for dom in DOMAINS for s in SEEDS]
g_worsts = [pct(load_strong("strong_d0.01", dom, s)["worst_holdout_acc"]) for dom in DOMAINS for s in SEEDS]
ng_worsts = [pct(load_strong("strong_nogate", dom, s)["worst_holdout_acc"]) for dom in DOMAINS for s in SEEDS]

diff_t = np.array(g_targets) - np.array(ng_targets)
diff_w = np.array(g_worsts) - np.array(ng_worsts)
t_t, p_t = stats.ttest_rel(g_targets, ng_targets)
t_w, p_w = stats.ttest_rel(g_worsts, ng_worsts)
print(f"  target:  mean diff={np.mean(diff_t):+.3f}pp, t(11)={t_t:+.3f}, p={p_t:.4f}, positive={sum(diff_t>0)}/{len(diff_t)}")
print(f"  worst:   mean diff={np.mean(diff_w):+.3f}pp, t(11)={t_w:+.3f}, p={p_w:.4f}, positive={sum(diff_w>0)}/{len(diff_w)}")

# ============================================================
# 3. Default augmentation: delta=0.001 (max pressure) vs delta=2.0 (baseline)
# ============================================================
print("\n" + "=" * 90)
print("3. DEFAULT AUG: Max Pressure (d=0.001) vs Baseline (d=2.0)")
print("=" * 90)
for metric_name, metric_key in [("target_acc", "target_acc"), ("worst_holdout", "worst_holdout_acc")]:
    base = [pct(load("2.0", dom, s)[metric_key]) for dom in DOMAINS for s in SEEDS]
    press = [pct(load("0.001", dom, s)[metric_key]) for dom in DOMAINS for s in SEEDS]
    diff = np.array(press) - np.array(base)
    t, p = stats.ttest_rel(press, base)
    print(f"  {metric_name:>15s}: baseline={np.mean(base):.2f}, pressure={np.mean(press):.2f}, diff={np.mean(diff):+.3f}pp, p={p:.4f}")

# ============================================================
# 4. Summary: does gate provide protection?
# ============================================================
print("\n" + "=" * 90)
print("4. SUMMARY: Does the gate provide protection under pressure?")
print("=" * 90)

# Strong aug is the key test: does gate help when augmentation is strong?
print("  Strong augmentation (the critical test):")
print(f"    With gate (d=0.01):    target={np.mean(g_targets):.2f}%, worst={np.mean(g_worsts):.2f}%")
print(f"    Without gate (d=1e6):  target={np.mean(ng_targets):.2f}%, worst={np.mean(ng_worsts):.2f}%")
print(f"    Gate benefit:          target={np.mean(diff_t):+.2f}pp (p={p_t:.4f}), worst={np.mean(diff_w):+.2f}pp (p={p_w:.4f})")
print(f"    Gate activation:       mean_gate={np.mean([load_strong('strong_d0.01', dom, s)['aug_mean_gate'] for dom in DOMAINS for s in SEEDS]):.4f}")

# Default aug delta sweep: does shrinking delta hurt performance?
print("\n  Default augmentation delta sweep (gate shrinks as delta decreases):")
print(f"    d=2.0   (gate=1.000):  target={np.mean(delta_data['2.0']['target']):.2f}%")
print(f"    d=0.001 (gate<1.0):    target={np.mean(delta_data['0.001']['target']):.2f}%")
print(f"    Difference:            {np.mean(delta_data['0.001']['target'])-np.mean(delta_data['2.0']['target']):+.2f}pp")

print("\n  Interpretation:")
gate_helps_strong = np.mean(diff_t) > 0 and p_t < 0.1
print(f"    Gate helps under strong augmentation? {'YES' if gate_helps_strong else 'NO/UNCLEAR'} (target diff={np.mean(diff_t):+.2f}pp, p={p_t:.4f})")
