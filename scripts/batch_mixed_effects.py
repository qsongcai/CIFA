#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R3-(5): Re-analysis of the two PACS batches (reference vs unified code).

The two batches share the same seed labels and the same deterministic data
splits (datasets.py is identical), but they are two independent training
realizations: no run is bit-identical (0/40 CIFA), because the codebases
differ (train.py/networks.py), the unified build enables persistent_workers
(altering the worker RNG stream for random image augmentation), and cuDNN
determinism is not pinned.

Structure: a paired CIFA-SWAD contrast y_{b,s} (or y_{b,d,s}), observed at the
same split design (seed; domain,seed) in two batches b in {ref, uni}.
Because the split is shared, a seed/block random effect u may correlate the two
batches; the naive n=20 treatment ignores it. We therefore fit

    macro  : y_{b,s}   = mu + alpha_b + u_s + eps
    pooled : y_{b,d,s} = mu + alpha_b + beta_d + u_{d,s} + eps

with seed (or domain x seed block) as a random intercept and batch as a fixed
effect. We report ANOVA variance components (exact for the balanced design),
cross-check with statsmodels MixedLM, and re-run the TOST using the *correct*
SE for the pooled grand mean.

Delta = CIFA - SWAD (percentage points); positive favors CIFA.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
SEEDS = list(range(10))


def acc(batch, method, domain, seed, field):
    base = ("_preview_n10/results/pacs" if batch == "ref"
            else "CIFA-Repro/results/pacs")
    p = os.path.join(ROOT, base, method, f"{method}_{domain}_seed{seed}.json")
    return 100.0 * json.load(open(p))[field]


def paired_delta(batch, domain, seed, field="target_acc"):
    return (acc(batch, "cifa", domain, seed, field)
            - acc(batch, "swad", domain, seed, field))


# ---------------- macro: per-seed cross-domain mean (n=10 per batch) --------
rows = []
for b in ["ref", "uni"]:
    for s in SEEDS:
        ov = np.mean([paired_delta(b, d, s, "target_acc") for d in DOMAINS])
        ws = np.mean([paired_delta(b, d, s, "worst_holdout_acc") for d in DOMAINS])
        rows.append(dict(batch=b, seed=s, block=f"s{s}", overall=ov, worst=ws))
macro = pd.DataFrame(rows)

# ---------------- pooled: domain x seed (n=40 per batch) --------------------
prows = []
for b in ["ref", "uni"]:
    for d in DOMAINS:
        for s in SEEDS:
            prows.append(dict(batch=b, domain=d, seed=s, block=f"{d}__s{s}",
                              overall=paired_delta(b, d, s, "target_acc"),
                              worst=paired_delta(b, d, s, "worst_holdout_acc")))
pooled = pd.DataFrame(prows)


def naive(col, df):
    y = df[col].values
    m = y.mean(); se = y.std(ddof=1) / np.sqrt(len(y))
    return m, se


def anova_macro(df, col):
    """Balanced two-way: batch fixed (B=2), seed random (S=10)."""
    y = df.pivot(index="seed", columns="batch", values=col)
    B, S = y.shape[1], y.shape[0]
    grand = y.values.mean()
    mean_b = y.mean(axis=0)
    mean_s = y.mean(axis=1)
    SS_b = S * ((mean_b - grand) ** 2).sum()
    SS_s = B * ((mean_s - grand) ** 2).sum()
    SS_e = sum((y.loc[s, b] - mean_b[b] - mean_s[s] + grand) ** 2
               for b in y.columns for s in y.index)
    df_b, df_s, df_e = B - 1, S - 1, (B - 1) * (S - 1)
    MS_b, MS_s, MS_e = SS_b / df_b, SS_s / df_s, SS_e / df_e
    sig_u2 = max(0.0, (MS_s - MS_e) / B)
    sig_e2 = MS_e
    # pooled grand mean variance
    a = sig_u2 / S; c = sig_e2 / (B * S)
    var_mu = a + c
    df_mu = (a + c) ** 2 / (a ** 2 / df_s + c ** 2 / df_e) if (a > 0 and c > 0) else df_e
    # batch effect uni - ref (paired across seeds)
    diff = y["uni"] - y["ref"]
    t_batch, p_batch = st.ttest_rel(y["uni"], y["ref"])
    # within-seed cross-batch correlation
    r_corr, p_corr = st.pearsonr(y["ref"], y["uni"])
    return dict(B=B, S=S, grand=grand, mean_ref=mean_b["ref"], mean_uni=mean_b["uni"],
                batch_diff=diff.mean(), t_batch=t_batch, p_batch=p_batch,
                sig_u2=sig_u2, sig_e2=sig_e2, ICC=sig_u2 / (sig_u2 + sig_e2),
                r_seed=r_corr, p_seed=p_corr,
                se_mu_correct=np.sqrt(var_mu), df_mu=df_mu,
                se_mu_naive=naive(col, df)[1],
                se_batch_paired=np.sqrt(2 * sig_e2 / S))


def mixed_fit(df, col, with_domain):
    cols = [np.ones(len(df)), (df["batch"].values == "uni").astype(float)]
    names = ["Intercept(ref level)", "batch[uni-ref]"]
    if with_domain:
        dm = pd.get_dummies(df["domain"], drop_first=True)
        for c in dm.columns:
            cols.append(dm[c].values.astype(float)); names.append(f"dom[{c}]")
    X = np.column_stack(cols)
    md = MixedLM(df[col].values.astype(float), X,
                 groups=df["block"].values, exog_re=np.ones((len(df), 1)))
    fit = md.fit(reml=True)
    return fit, names


def tost_ci(mu, se, df, margin):
    tcrit = st.t.ppf(0.95, df)
    lo, hi = mu - tcrit * se, mu + tcrit * se
    pL = st.t.sf((mu + margin) / se, df)
    pU = st.t.cdf((mu - margin) / se, df)
    p_tost = max(pL, pU)
    return lo, hi, p_tost, "YES" if (lo > -margin and hi < margin) else "NO"


def report(name, df, col, anova, pooled_flag):
    print("=" * 92)
    print(f"{name}  [Delta = CIFA - SWAD, points; {col}]")
    print("=" * 92)
    print(f"  batch means: ref = {anova['mean_ref']:+.3f}  uni = {anova['mean_uni']:+.3f}")
    print(f"  batch fixed effect (uni-ref) = {anova['batch_diff']:+.3f}  "
          f"paired t = {anova['t_batch']:+.3f}, df=9, p = {anova['p_batch']:.4f}")
    print(f"  cross-batch within-seed correlation r = {anova['r_seed']:+.3f} "
          f"(p={anova['p_seed']:.3f})   ICC(block) = {anova['ICC']:.3f}")
    print(f"  variance components: sigma_u^2(seed/block) = {anova['sig_u2']:.4f}, "
          f"sigma_e^2 = {anova['sig_e2']:.4f}")
    print(f"  pooled grand mean mu = {anova['grand']:+.3f}")
    print(f"     SE naive n={len(df)} (independence) = {anova['se_mu_naive']:.3f}")
    print(f"     SE correct (mixed effects)          = {anova['se_mu_correct']:.3f}  "
          f"[df={anova['df_mu']:.1f}]")
    for mg in (0.5, 1.0, 2.0):
        lo, hi, p, eq = tost_ci(anova['grand'], anova['se_mu_correct'], anova['df_mu'], mg)
        print(f"     TOST +/-{mg}: 90pct-CI [{lo:+.2f},{hi:+.2f}]  p_TOST={p:.4f}  {eq}")
    print("  -- statsmodels MixedLM (REML) cross-check --")
    fit, names = mixed_fit(df, col, pooled_flag)
    for nm, est, se, pv in zip(names, fit.params, fit.bse, fit.pvalues):
        print(f"     {nm:20s} {est:+.4f}  (SE {se:.4f})  p={pv:.4f}")
    print(f"     group Var sigma_u^2 = {np.asarray(fit.cov_re).reshape(-1)[0]:.4f}; "
          f"resid Var sigma_e^2 = {fit.scale:.4f}")


print("\n################# MACRO (per-seed cross-domain average; 2 batches x 10 seeds) #################\n")
for col in ["overall", "worst"]:
    a = anova_macro(macro, col)
    report("MACRO", macro, col, a, False)
    print()

print("\n################# POOLED (domain x seed; 40 blocks x 2 batches) #################\n")
for col in ["overall", "worst"]:
    # De-mean by domain (fixed effect), then ANOVA over the 40 (domain,seed) blocks.
    d = pooled.copy()
    d["res"] = d[col] - d.groupby("domain")[col].transform("mean")
    piv = d.pivot_table(index="block", columns="batch", values="res")
    B, K = 2, len(piv)                       # 40 blocks, 2 batches each
    grand = pooled[col].mean()              # grand mean on the original scale
    mean_b = piv.mean(axis=0)               # residual batch offsets (effect retained)
    mean_blk = piv.mean(axis=1)
    SS_e = sum((piv.loc[k, b] - mean_b[b] - mean_blk[k]) ** 2
               for b in piv.columns for k in piv.index)
    df_e = K * (B - 1)
    MS_e = SS_e / df_e
    SS_blk = B * ((mean_blk - 0.0) ** 2).sum()
    df_blk = K - 1
    MS_blk = SS_blk / df_blk
    sig_u2 = max(0.0, (MS_blk - MS_e) / B)
    sig_e2 = MS_e
    a = sig_u2 / K; c = sig_e2 / (B * K)
    var_mu = a + c
    df_mu = (a + c) ** 2 / (a ** 2 / df_blk + c ** 2 / df_e) if (a > 0 and c > 0) else df_e
    # batch effect on the original scale (paired within each of the 40 blocks)
    piv0 = pooled.pivot_table(index="block", columns="batch", values=col)
    t_batch, p_batch = st.ttest_rel(piv0["uni"], piv0["ref"])
    r_corr, p_corr = st.pearsonr(piv0["ref"], piv0["uni"])
    aov = dict(B=B, S=K, grand=grand, mean_ref=piv0["ref"].mean(), mean_uni=piv0["uni"].mean(),
               batch_diff=(piv0["uni"].mean() - piv0["ref"].mean()),
               t_batch=t_batch, p_batch=p_batch,
               sig_u2=sig_u2, sig_e2=sig_e2, ICC=sig_u2 / (sig_u2 + sig_e2),
               r_seed=r_corr, p_seed=p_corr,
               se_mu_correct=np.sqrt(var_mu), df_mu=df_mu,
               se_mu_naive=naive(col, pooled)[1], se_batch_paired=np.sqrt(2 * sig_e2 / K))
    report("POOLED", pooled, col, aov, True)
    print()
