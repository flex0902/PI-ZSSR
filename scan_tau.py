"""
Post-hoc sensitivity of the discrepancy stopping rule to the threshold factor tau (and to a mis-estimated sigma).

For every run (*_curve.csv + *_metrics.json) and every tau in --taus, find the first evaluated epoch at which
    fit RMSE <= tau * sigma_eff
and read the truth RMSE at that epoch.  tau != 1 is equivalent to over/under-estimating sigma by the same factor.
Outputs
  <dir>/TAU_scan_runs.csv   per run x tau
  <dir>/TAU_scan_table.md   mean +/- std over seeds per (type, sigma, tau), plus oracle and Tikhonov(disc) for reference
  <dir>/TAU_scan.png        truth RMSE vs tau, one line per (type, sigma)

  python scan_tau.py --dirs e2 e2b --out e2/tau
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["e2"])
    ap.add_argument("--taus", default="0.8,0.9,1.0,1.1,1.2,1.3,1.5")
    ap.add_argument("--out", default="e2/tau")
    ap.add_argument("--prefer_fine", action="store_true", default=True,
                    help="if the same (type,sigma,seed) exists in several dirs keep the one with the finest eval interval")
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    runs = {}
    for d in args.dirs:
        for fn in sorted(glob.glob(os.path.join(d, "*_metrics.json"))):
            m = json.load(open(fn))
            if m["noise"] <= 0:
                continue
            cv = pd.read_csv(fn.replace("_metrics.json", "_curve.csv"))
            step = int(np.diff(cv.Epoch.values[:2])[0]) if len(cv) > 1 else 1
            key = (m["noise_type"], m["noise"], m["seed"])
            if key in runs and runs[key][2] <= step:
                continue
            runs[key] = (m, cv, step, os.path.basename(fn)[:-len("_metrics.json")], d)

    rows = []
    for (nt, sg, seed), (m, cv, step, name, d) in sorted(runs.items()):
        sig = m["sigma_eff"]
        base = {"dir": d, "run": name, "type": nt, "sigma": sg, "seed": seed, "eval_step": step,
                "oracle_epoch": m.get("oracle_epoch"), "oracle_rmse": m.get("oracle_rmse_truth"),
                "tik_disc_rmse": m.get("baseline", {}).get("rmse_truth_disc"),
                "tik_oracle_rmse": m.get("baseline", {}).get("rmse_truth_oracle")}
        for tau in taus:
            hit = cv[cv.rmse_fit <= tau * sig]
            r = dict(base, tau=tau)
            if len(hit):
                r["epoch"] = int(hit.Epoch.iloc[0]); r["rmse_truth"] = float(hit.rmse_truth.iloc[0]); r["fit"] = float(hit.rmse_fit.iloc[0])
            else:
                r["epoch"] = None; r["rmse_truth"] = None; r["fit"] = None
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(args.out + "_scan_runs.csv", index=False, float_format="%.4f")

    g = df.groupby(["type", "sigma", "tau"])
    mean, std = g[["epoch", "rmse_truth"]].mean(), g[["epoch", "rmse_truth"]].std()
    ref = df.groupby(["type", "sigma"])[["oracle_rmse", "tik_disc_rmse", "tik_oracle_rmse", "eval_step"]].mean()
    tab = pd.DataFrame(index=ref.index)
    tab["eval step"] = ref.eval_step.astype(int)
    for tau in taus:
        col = []
        for idx in tab.index:
            k = idx + (tau,)
            if k in mean.index and np.isfinite(mean.loc[k, "rmse_truth"]):
                col.append(f"{mean.loc[k,'rmse_truth']:.2f} ± {std.loc[k,'rmse_truth']:.2f} (ep {mean.loc[k,'epoch']:.0f})")
            else:
                col.append("never")
        tab[f"tau={tau:g}"] = col
    tab["PI-ZSSR (oracle)"] = [f"{v:.2f}" for v in ref.oracle_rmse]
    tab["Tikhonov-FFT (alpha by discrepancy)"] = [f"{v:.2f}" for v in ref.tik_disc_rmse]
    try:
        md = tab.reset_index().to_markdown(index=False)
    except Exception:
        md = tab.reset_index().to_string(index=False)
    open(args.out + "_scan_table.md", "w", encoding="utf-8").write(md)
    print(md)

    # figure
    types = [t for t in ("white", "corr") if (df.type == t).any()]
    fig, axes = plt.subplots(1, len(types), figsize=(6.5 * len(types), 5), sharey=False)
    axes = np.atleast_1d(axes)
    for a, t in zip(axes, types):
        sub = df[df.type == t]
        for i, sg in enumerate(sorted(sub.sigma.unique())):
            ss = sub[sub.sigma == sg].groupby("tau")["rmse_truth"]
            m_, s_ = ss.mean(), ss.std().fillna(0)
            a.errorbar(m_.index, m_.values, yerr=s_.values, marker="o", ms=4, capsize=3, color=f"C{i}", label=f"sigma={sg:g} mGal")
            orc = sub[sub.sigma == sg].oracle_rmse.mean(); tk = sub[sub.sigma == sg].tik_disc_rmse.mean()
            a.axhline(orc, color=f"C{i}", ls=":", lw=1)
            a.axhline(tk, color=f"C{i}", ls="-.", lw=1)
        a.axvline(1.0, color="grey", ls="--", lw=0.8)
        a.set_xlabel("tau  (stop when fit RMSE <= tau * sigma)"); a.set_ylabel("truth RMSE at stop (mGal)")
        a.set_title(f"{'white' if t == 'white' else 'along-track correlated'} noise\ndotted = PI-ZSSR (oracle), dash-dot = Tikhonov-FFT (alpha by discrepancy)")
        a.grid(alpha=0.3); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(args.out + "_scan.png", dpi=140); plt.close()
    print("saved", args.out + "_scan.png")


if __name__ == "__main__":
    main()
