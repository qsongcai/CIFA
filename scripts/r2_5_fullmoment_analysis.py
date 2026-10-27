"""
R2-5 独立重算：fullmoment（混合完整矩，cifa_decompose=false）vs full（只混 style deviation，decompose=true）
数据：CIFA-Repro/results/ablation/{variant}/{variant}_{domain}_seed{0..4}.json
PACS 四域 × 5 seed = 20/变体；唯一受控变量 cifa_decompose。
口径：target_acc（目标域留一域）；macro=每 seed 四域等权；own-worst=每 seed 跨四域 target_acc 最小值。
配对（同 seed，控初始化/划分）+ Welch 未配对（稳健性）并列；n=5 功效低，重效应量与 CI。
"""
import json, glob, os
import numpy as np
from scipy import stats

ROOT = "CIFA-Repro/results/ablation"
DOMS = ["art_painting", "cartoon", "photo", "sketch"]
VAR = ["full", "fullmoment", "noguard", "uniform", "noswad"]

def load(variant):
    """返回 acc[domain][seed]，并按 seed 对齐。"""
    acc = {d: np.full(5, np.nan) for d in DOMS}
    for f in glob.glob(os.path.join(ROOT, variant, "*.json")):
        d = json.load(open(f))
        acc[d["target_domain"]][d["seed"]] = d["target_acc"]
    return acc

data = {v: load(v) for v in VAR}

def macro(v):
    A = np.array([data[v][d] for d in DOMS])      # 4 x 5
    return A.mean(0)                               # 每 seed 四域等权

def ownworst(v):
    A = np.array([data[v][d] for d in DOMS])
    return A.min(0)                                # 每 seed 跨四域 min

def paired_tests(diff, label):
    """diff = fullmoment - full（同 seed 配对差，长度 n）。"""
    n = len(diff); m = diff.mean(); sd = diff.std(ddof=1)
    tcrit = stats.t.ppf(0.975, n-1)
    ci = (m - tcrit*sd/np.sqrt(n), m + tcrit*sd/np.sqrt(n))
    t, p = stats.ttest_1samp(diff, 0)
    try:
        wp = stats.wilcoxon(diff).pvalue
    except ValueError:
        wp = np.nan
    dz = m/sd if sd > 0 else np.nan
    print(f"  {label:28s} Δ={m*100:+.2f}pp  95%CI[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]  "
          f"配对t={t:+.2f},p={p:.3f}  Wilcoxon p={wp:.3f}  dz={dz:+.2f}")
    return m, ci, p

def welch(x, y, label):
    t, p = stats.ttest_ind(x, y, equal_var=False)
    u, pu = stats.mannwhitneyu(x, y, alternative='two-sided')
    print(f"  {label:28s} 未配对 Welch p={p:.3f}  Mann-Whitney p={pu:.3f}")
    return p

print("="*92)
print("一、各变体逐域 / macro / own-worst（均值±SD，百分点；n=5 seeds）")
print("="*92)
print(f"{'variant':12s} {'art':>10s} {'cartoon':>10s} {'photo':>10s} {'sketch':>10s} {'macro':>10s} {'own-worst':>10s}")
summary = {}
for v in VAR:
    cells = []
    for d in DOMS:
        a = data[v][d]
        cells.append(f"{a.mean()*100:.2f}±{a.std(ddof=1)*100:.2f}")
    ma = macro(v); ow = ownworst(v)
    summary[v] = dict(macro=ma, ownworst=ow)
    print(f"{v:12s} " + " ".join(f"{c:>10s}" for c in cells) +
          f" {ma.mean()*100:8.2f}±{ma.std(ddof=1)*100:.2f} {ow.mean()*100:7.2f}±{ow.std(ddof=1)*100:.2f}")

print()
print("="*92)
print("二、核心对比 fullmoment − full（同 seed 配对，正=fullmoment 更好）")
print("="*92)
for d in DOMS:
    diff = data["fullmoment"][d] - data["full"][d]
    paired_tests(diff, f"域 {d}")
print("  " + "-"*86)
m_macro, ci_macro, p_macro = paired_tests(macro("fullmoment")-macro("full"), "overall macro")
m_ow, ci_ow, p_ow = paired_tests(ownworst("fullmoment")-ownworst("full"), "own-worst-domain")
print()
print("  未配对稳健性（同结论则稳）：")
welch(macro("fullmoment"), macro("full"), "overall macro")
welch(ownworst("fullmoment"), ownworst("full"), "own-worst-domain")

print()
print("="*92)
print("三、fullmoment 列出逐 seed 的 macro 与 own-worst（看离散度/方向）")
print("="*92)
print("  seed :   full_macro  fm_macro   Δmacro | full_ow  fm_ow   Δow")
for s in range(5):
    print(f"   {s}    :  {macro('full')[s]*100:7.2f}  {macro('fullmoment')[s]*100:7.2f}  "
          f"{(macro('fullmoment')[s]-macro('full')[s])*100:+6.2f}  | "
          f"{ownworst('full')[s]*100:6.2f} {ownworst('fullmoment')[s]*100:6.2f}  "
          f"{(ownworst('fullmoment')[s]-ownworst('full')[s])*100:+6.2f}")

print()
print("="*92)
print("四、全貌：各变体相对 full 的 macro 差（服务 R2-6 noswad / R2-8 guard / uniform）")
print("="*92)
for v in VAR:
    if v == "full": continue
    Δm = (macro(v)-macro("full")).mean()*100
    Δo = (ownworst(v)-ownworst("full")).mean()*100
    print(f"  {v:12s} Δmacro={Δm:+.2f}pp   Δown-worst={Δo:+.2f}pp")
