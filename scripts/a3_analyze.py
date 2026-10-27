"""Analyse A3 probe outputs and adjudicate Assumption 3.

Reads every `*.npz` produced by a3_probe.py (default: results/a3), builds the
trajectory ensemble (TE), the augmentation-family ensemble (AE) and the dense
weight average (hbar), and quantifies their alignment:

  core (Assumption 3) : TE vs AE
  Theorem 2 sanity    : hbar vs TE   (weight average ~ its own trajectory ens.)
  derived             : hbar vs AE
  negative control    : TE(seed i) vs TE(seed j), same target, different seed

Metrics per pair: Jensen-Shannon divergence (mean/median), total variation
(mean/median), decision agreement, flattened-probability Pearson correlation,
linear CKA of penultimate features, and each side's accuracy and the accuracy
gap. Results are aggregated per target domain and over the whole dataset, then
checked against pre-registered thresholds.

Run:
    python scripts/a3_analyze.py --a3_dir results/a3 --dataset pacs
"""
import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np

EPS = 1e-12

# ---- pre-registered adjudication thresholds (set before seeing results) ----
TH = {
    "agree": 0.95,          # TE-AE decision agreement
    "js_med": 0.01,         # median Jensen-Shannon (nats)
    "tv_med": 0.05,         # median total variation
    "acc_gap": 1.0,         # |acc_TE - acc_AE| in percentage points
    "thm2_agree": 0.95,
    "thm2_acc_gap": 1.0,
}


def ens_accuracy(p, y):
    return 100.0 * (p.argmax(1) == y).mean()


def js_div(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        return (a * (np.log(a + EPS) - np.log(b + EPS))).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tv_div(p, q):
    return 0.5 * np.abs(p - q).sum(1)


def prob_corr(p, q):
    a, b = p.flatten(), q.flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def linear_cka(x, y):
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    gx, gy = x @ x.T, y @ y.T
    num = (gx * gy).sum()
    den = np.linalg.norm(gx) * np.linalg.norm(gy) + EPS
    return float(num / den)


def pair_metrics(pA, fA, pB, fB, y):
    js = js_div(pA, pB)
    tv = tv_div(pA, pB)
    accA, accB = ens_accuracy(pA, y), ens_accuracy(pB, y)
    return {
        "js_mean": float(js.mean()),
        "js_med": float(np.median(js)),
        "tv_mean": float(tv.mean()),
        "tv_med": float(np.median(tv)),
        "agree": float((pA.argmax(1) == pB.argmax(1)).mean()),
        "corr": prob_corr(pA, pB),
        "cka": linear_cka(fA, fB),
        "acc_A": float(accA),
        "acc_B": float(accB),
        "acc_gap": float(abs(accA - accB)),
    }


def load_case(npz_path):
    d = np.load(npz_path)
    te_probs = d["te_probs"]
    te_feat = d["te_feat"]
    out = {
        "y": d["y"],
        "hbar_probs": d["hbar_probs"],
        "hbar_feat": d["hbar_feat"],
        "ae_probs": d["ae_probs"].mean(0),
        "ae_feat": d["ae_feat"].mean(0),
    }
    if te_probs.ndim == 3 and te_probs.shape[0] > 0:
        out["te_probs"] = te_probs.mean(0)
        out["te_feat"] = te_feat.mean(0)
        out["n_te"] = int(te_probs.shape[0])
    else:
        out["te_probs"] = d["hbar_probs"]
        out["te_feat"] = d["hbar_feat"]
        out["n_te"] = 0
    return out


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3_dir", default="results/a3")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or args.a3_dir

    npz_files = sorted(glob.glob(os.path.join(args.a3_dir, "*.npz")))
    if not npz_files:
        raise SystemExit(f"no .npz found in {args.a3_dir}")

    cases = {}   # (target, seed) -> case
    for fn in npz_files:
        meta_fn = fn[:-4] + ".json"
        if os.path.exists(meta_fn):
            meta = json.load(open(meta_fn))
            key = (meta["target_domain"], int(meta["seed"]))
        else:
            base = os.path.basename(fn)[:-4]
            key = (base, 0)
        cases[key] = load_case(fn)

    # ---- per-case core metrics ----
    rows = []
    for (target, seed), c in sorted(cases.items()):
        y = c["y"]
        te_ae = pair_metrics(c["te_probs"], c["te_feat"],
                             c["ae_probs"], c["ae_feat"], y)
        hb_te = pair_metrics(c["hbar_probs"], c["hbar_feat"],
                             c["te_probs"], c["te_feat"], y)
        hb_ae = pair_metrics(c["hbar_probs"], c["hbar_feat"],
                             c["ae_probs"], c["ae_feat"], y)
        rows.append({"target": target, "seed": seed, "n_te": c["n_te"],
                     "te_ae": te_ae, "hb_te": hb_te, "hb_ae": hb_ae})

    # ---- negative control: TE across seeds, same target ----
    neg = []
    targets = sorted({t for (t, s) in cases})
    for t in targets:
        seeds = sorted({s for (tt, s) in cases if tt == t})
        for si, sj in combinations(seeds, 2):
            A, B = cases[(t, si)], cases[(t, sj)]
            if len(A["y"]) != len(B["y"]):
                continue
            neg.append(pair_metrics(A["te_probs"], A["te_feat"],
                                    B["te_probs"], B["te_feat"], A["y"]))

    def agg(vals):
        vals = np.array(vals, dtype=float)
        return {"mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}

    # ---- dataset-level aggregation of core metrics ----
    keys = ["agree", "js_med", "tv_med", "corr", "cka", "acc_gap",
            "acc_A", "acc_B"]
    overall = {f"te_ae_{k}": agg([r["te_ae"][k] for r in rows])
               for k in keys}
    overall.update({f"hb_te_{k}": agg([r["hb_te"][k] for r in rows])
                    for k in keys})
    if neg:
        overall["neg_agree"] = agg([n["agree"] for n in neg])
        overall["neg_cka"] = agg([n["cka"] for n in neg])
        overall["neg_js_med"] = agg([n["js_med"] for n in neg])

    # ---- adjudication ----
    te = {k: overall[f"te_ae_{k}"]["mean"] for k in
          ["agree", "js_med", "tv_med", "acc_gap"]}
    hb = {k: overall[f"hb_te_{k}"]["mean"] for k in ["agree", "acc_gap"]}
    checks = {
        "A3_agree>=.95": te["agree"] >= TH["agree"],
        "A3_js_med<=.01": te["js_med"] <= TH["js_med"],
        "A3_tv_med<=.05": te["tv_med"] <= TH["tv_med"],
        "A3_acc_gap<=1": te["acc_gap"] <= TH["acc_gap"],
        "A3_beats_negative": (not neg) or
            (te["agree"] > overall["neg_agree"]["mean"]),
        "Thm2_agree>=.95": hb["agree"] >= TH["thm2_agree"],
        "Thm2_acc_gap<=1": hb["acc_gap"] <= TH["thm2_acc_gap"],
    }
    a3_keys = ["A3_agree>=.95", "A3_js_med<=.01", "A3_tv_med<=.05",
               "A3_acc_gap<=1", "A3_beats_negative"]
    verdict = {
        "assumption_3_supported": all(checks[k] for k in a3_keys),
        "theorem_2_sanity_ok": checks["Thm2_agree>=.95"] and
            checks["Thm2_acc_gap<=1"],
        "checks": checks,
    }

    summary = {"dataset": args.dataset, "n_cases": len(rows),
               "n_negative_pairs": len(neg), "per_case": rows,
               "negative_control": neg, "overall": overall,
               "thresholds": TH, "verdict": verdict}
    with open(os.path.join(out_dir, "a3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- markdown table ----
    lines = [f"# A3 alignment probe — {args.dataset}", "",
             f"Cases: {len(rows)}; cross-seed negative pairs: {len(neg)}", ""]
    lines.append("## Per case (TE vs AE = Assumption 3)")
    lines.append("")
    lines.append("| Target | Seed | n_TE | Agree | JS med | TV med | Corr | CKA | "
                 "acc_TE | acc_AE | gap |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["te_ae"]
        lines.append("| {t} | {s} | {n} | {ag} | {js} | {tv} | {co} | {ck} | "
                     "{aA:.2f} | {aB:.2f} | {g:.2f} |".format(
                         t=r["target"], s=r["seed"], n=r["n_te"],
                         ag=fmt(m["agree"]), js=fmt(m["js_med"], 4),
                         tv=fmt(m["tv_med"]), co=fmt(m["corr"]),
                         ck=fmt(m["cka"]), aA=m["acc_A"], aB=m["acc_B"],
                         g=m["acc_gap"]))
    lines.append("")
    lines.append("## Dataset-level means (±sd)")
    lines.append("")
    lines.append("| Pair | Agree | JS med | TV med | Corr | CKA | Acc gap |")
    lines.append("|---|---|---|---|---|---|---|")
    for tag, pre in [("TE vs AE (A3)", "te_ae"),
                     ("hbar vs TE (Thm 2)", "hb_te")]:
        def cell(k):
            d = overall[f"{pre}_{k}"]
            return f"{d['mean']:.3f}±{d['sd']:.3f}"
        gap_k = "acc_gap"
        lines.append(f"| {tag} | {cell('agree')} | {cell('js_med')} | "
                     f"{cell('tv_med')} | {cell('corr')} | {cell('cka')} | "
                     f"{cell(gap_k)} |")
    if neg:
        lines.append(f"| TE cross-seed (neg.) | "
                     f"{overall['neg_agree']['mean']:.3f} | "
                     f"{overall['neg_js_med']['mean']:.4f} | — | — | "
                     f"{overall['neg_cka']['mean']:.3f} | — |")
    lines.append("")
    lines.append("## Verdict (pre-registered thresholds)")
    lines.append("")
    for k, v in checks.items():
        lines.append(f"- [{'PASS' if v else 'FAIL'}] {k}")
    lines.append("")
    lines.append(f"**Assumption 3 supported: {verdict['assumption_3_supported']}**  ")
    lines.append(f"**Theorem 2 sanity ok: {verdict['theorem_2_sanity_ok']}**")
    md = "\n".join(lines)
    with open(os.path.join(out_dir, "a3_table.md"), "w") as f:
        f.write(md)

    # ---- console ----
    print(md)
    print("\nsaved", os.path.join(out_dir, "a3_summary.json"),
          "and a3_table.md")


if __name__ == "__main__":
    main()
