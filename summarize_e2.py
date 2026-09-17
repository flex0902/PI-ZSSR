"""
Aggregate E2 runs (e2/*_metrics.json) into
  * e2/E2_table.csv / E2_table.md : mean +/- std over seeds per (noise type, sigma) of
        truth RMSE for PI-ZSSR stopped by {discrepancy, loss plateau, oracle} and for Tikhonov-FFT {oracle alpha, discrepancy alpha}
        noise-amplification factor (truth RMSE / sigma), amplification ratio, stopping epochs
  * e2/E2_rmse_vs_noise.png : truth RMSE vs noise sigma, one panel per noise type
  * e2/E2_fields_<type><sigma>.png : truth / prediction(disc) / Tikhonov(disc) / error maps for seed 0

  python summarize_e2.py --dir e2
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

METHODS = [("disc", "PI-ZSSR (discrepancy)", "C3", "-", "o"),
           ("old", "PI-ZSSR (loss plateau)", "k", "-", "s"),
           ("oracle", "PI-ZSSR (oracle)", "C3", ":", "^"),
           ("tik_disc", "Tikhonov-FFT (alpha by discrepancy)", "C0", "-", "o"),
           ("tik_oracle", "Tikhonov-FFT (best alpha)", "C0", ":", "^")]


def load(dirs):
    """Load runs from one or more directories. If the same (type, sigma, seed) appears in several
    directories, keep the run with the finest evaluation interval (e.g. e2b per-epoch runs override e2)."""
    best = {}
    for d in dirs:
        for fn in sorted(glob.glob(os.path.join(d, "*_metrics.json"))):
            m = json.load(open(fn))
            cvf = fn.replace("_metrics.json", "_curve.csv")
            step = 1
            if os.path.exists(cvf):
                ep = pd.read_csv(cvf, usecols=["Epoch"]).Epoch.values
                step = int(ep[1] - ep[0]) if len(ep) > 1 else 1
            key = (m["noise_type"] if m["noise"] > 0 else "none", m["noise"], m["seed"])
            if key in best:
                s0, d0, fn0, m0 = best[key]
                fine, coarse = ((m, d, fn, step), (m0, d0, fn0, s0)) if step < s0 else ((m0, d0, fn0, s0), (m, d, fn, step))
                mm = dict(fine[0])
                # the plateau rule needs the long run: take old_* from whichever run has it
                if mm.get("old_epoch") is None and coarse[0].get("old_epoch") is not None:
                    for k, v in coarse[0].items():
                        if k.startswith("old_"):
                            mm[k] = v
                best[key] = (fine[3], fine[1], fine[2], mm)
                continue
            best[key] = (step, d, fn, m)
    rows = []
    for (_, d, fn, m) in best.values():
        r = {"dir": d, "run": os.path.basename(fn)[:-len("_metrics.json")], "type": m["noise_type"] if m["noise"] > 0 else "none",
             "sigma": m["noise"], "sigma_eff": m["sigma_eff"], "seed": m["seed"], "truth_amp": m["truth_amp_ratio"],
             "time_sec": m["time_sec"]}
        for k in ("disc", "old", "final"):
            r[f"{k}_epoch"] = m.get(f"{k}_epoch"); r[f"{k}_rmse"] = m.get(f"{k}_rmse_truth"); r[f"{k}_amp"] = m.get(f"{k}_amp_ratio")
        r["oracle_epoch"] = m.get("oracle_epoch"); r["oracle_rmse"] = m.get("oracle_rmse_truth")
        b = m.get("baseline", {})
        r["tik_oracle_rmse"] = b.get("rmse_truth_oracle"); r["tik_oracle_amp"] = b.get("amp_oracle"); r["tik_oracle_alpha"] = b.get("alpha_oracle")
        r["tik_disc_rmse"] = b.get("rmse_truth_disc"); r["tik_disc_amp"] = b.get("amp_disc"); r["tik_disc_alpha"] = b.get("alpha_disc")
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="e2", help="output directory (and first input directory)")
    ap.add_argument("--extra", nargs="*", default=[], help="additional run directories; finer eval interval wins")
    args = ap.parse_args()
    df = load([args.dir] + list(args.extra))
    if df.empty:
        print("no runs"); return
    df = df.sort_values(["type", "sigma", "seed"])
    df.to_csv(os.path.join(args.dir, "E2_runs.csv"), index=False, float_format="%.4f")

    # noise-free runs: the discrepancy rule cannot fire (sigma = 0); report plateau / oracle only
    num_cols = [c for c in df.columns if c.endswith("_rmse") or c.endswith("_epoch") or c.endswith("_amp")]
    g = df.groupby(["type", "sigma"])
    mean, std, n = g[num_cols].mean(), g[num_cols].std(), g.size()
    tab = pd.DataFrame(index=mean.index)
    tab["n"] = n
    for k, lab, *_ in METHODS:
        tab[f"{lab} RMSE"] = [f"{m:.2f} ± {s:.2f}" if np.isfinite(m) else "–" for m, s in zip(mean[f"{k}_rmse"], std[f"{k}_rmse"])]
    for k in ("disc", "old", "oracle"):
        tab[f"{k} epoch"] = [f"{m:.0f} ± {s:.0f}" if np.isfinite(m) else "–" for m, s in zip(mean[f"{k}_epoch"], std[f"{k}_epoch"])]
    for k in ("disc", "old", "tik_disc"):
        tab[f"{k} amp"] = [f"{m:.2f}" if np.isfinite(m) else "–" for m in mean[f"{k}_amp"]]
    tab["truth amp"] = [f"{m:.2f}" for m in g["truth_amp"].mean()]
    # noise amplification factor (NAF) = truth RMSE / sigma  (raw: includes the noise-free error floor)
    # noise-only NAF = sqrt(RMSE_sigma^2 - RMSE_0^2) / sigma  (error attributable to the noise alone)
    # selectivity = NAF / signal amplification ratio; < 1 means noise is amplified LESS than signal
    floor = {}
    if ("none", 0.0) in mean.index:
        f0 = mean.loc[("none", 0.0)]
        floor = {"disc": f0["old_rmse"], "old": f0["old_rmse"], "tik_disc": f0["tik_disc_rmse"], "tik_oracle": f0["tik_oracle_rmse"]}
    compact = pd.DataFrame(index=mean.index)
    compact["n"] = n
    for k in ("disc", "old", "tik_disc"):
        naf, naf0, sel = [], [], []
        for (t, s), m, a in zip(mean.index, mean[f"{k}_rmse"], mean[f"{k}_amp"]):
            if s > 0 and np.isfinite(m):
                v = m / s
                v0 = np.sqrt(max(m ** 2 - floor.get(k, 0.0) ** 2, 0.0)) / s
                naf.append(f"{v:.2f}"); naf0.append(f"{v0:.2f}"); sel.append(f"{v0 / a:.2f}")
            else:
                naf.append("–"); naf0.append("–"); sel.append("–")
        tab[f"{k} NAF"] = naf; tab[f"{k} NAF(noise-only)"] = naf0; tab[f"{k} NAF/amp"] = sel
        compact[f"{k} RMSE"] = tab[f"{METHODS[[x[0] for x in METHODS].index(k)][1]} RMSE"]
        compact[f"{k} amp"] = tab[f"{k} amp"]; compact[f"{k} NAF"] = naf0; compact[f"{k} NAF/amp"] = sel
    compact["oracle RMSE"] = tab["PI-ZSSR (oracle) RMSE"]; compact["tik_oracle RMSE"] = tab["Tikhonov-FFT (best alpha) RMSE"]
    compact["truth amp"] = tab["truth amp"]
    tab.to_csv(os.path.join(args.dir, "E2_table.csv"))
    compact.to_csv(os.path.join(args.dir, "E2_table_compact.csv"))

    def md(d):
        try:
            return d.reset_index().to_markdown(index=False)
        except Exception:
            return d.reset_index().to_string(index=False)
    open(os.path.join(args.dir, "E2_table.md"), "w", encoding="utf-8").write(md(tab))
    open(os.path.join(args.dir, "E2_table_compact.md"), "w", encoding="utf-8").write(md(compact))
    print(md(compact))

    # ---------- figure: RMSE vs noise ----------
    types = [t for t in ("white", "corr") if (df.type == t).any()]
    fig, axes = plt.subplots(1, len(types), figsize=(6.5 * len(types), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    zero = df[df.type == "none"]
    for a, t in zip(axes, types):
        sub = df[(df.type == t)]
        if not zero.empty:
            sub = pd.concat([zero, sub])
        gg = sub.groupby("sigma")
        for k, lab, col, ls, mk in METHODS:
            m = gg[f"{k}_rmse"].mean(); s = gg[f"{k}_rmse"].std().fillna(0)
            ok = m.notna()
            a.errorbar(m.index[ok], m[ok], yerr=s[ok], color=col, ls=ls, marker=mk, ms=5, capsize=3, lw=1.4, label=lab)
        a.plot([0, sub.sigma.max()], [0, sub.sigma.max()], color="grey", lw=0.8, ls="--", label="RMSE = input sigma (NAF = 1)")
        a.set_xlabel("input noise sigma (mGal)"); a.grid(alpha=0.3)
        a.set_title(f"{'white' if t == 'white' else 'along-track correlated (20 km)'} noise\n"
                    f"truth RMSE at 1500 m vs input noise (mean ± std over seeds)")
    axes[0].set_ylabel("RMSE vs hidden truth (mGal)")
    axes[-1].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out = os.path.join(args.dir, "E2_rmse_vs_noise.png")
    plt.savefig(out, dpi=140); plt.close()
    print("saved", out)

    # ---------- field maps, seed 0 ----------
    for _, r in df[df.seed == df.seed.min()].iterrows():
        fz = os.path.join(r["dir"], r["run"] + "_fields.npz")
        if not os.path.exists(fz):
            continue
        z = np.load(fz)
        truth, roi = z["truth"], z["roi"]
        key = "pred_disc" if "pred_disc" in z else ("pred_old" if "pred_old" in z else "pred_final")
        pred = z[key]; tik = z["pred_tikhonov_disc"] if "pred_tikhonov_disc" in z else None
        ext = (z["lons"].min(), z["lons"].max(), z["lats"].min(), z["lats"].max())
        lim = np.nanmax(np.abs(truth[roi])); dl = max(np.nanmax(np.abs((truth - pred)[roi])), np.nanmax(np.abs((truth - tik)[roi])) if tik is not None else 0)
        fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
        def show(a, arr, title, lim, cmap="RdYlBu_r"):
            im = a.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=-lim, vmax=lim); a.set_title(title, fontsize=10)
            plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        show(ax[0, 0], z["obs_high"], f"input 5000 m, {r['type']} noise sigma={r['sigma']:g} mGal", np.nanmax(np.abs(z["obs_high"][roi])))
        show(ax[0, 1], truth, "truth 1500 m", lim)
        rule_name = {"disc": "discrepancy", "old": "loss plateau", "final": "final"}[key[5:]]
        show(ax[0, 2], pred, f"PI-ZSSR ({rule_name}), epoch {r[key[5:] + '_epoch']:.0f}  RMSE {r[key[5:] + '_rmse']:.2f}", lim)
        show(ax[1, 0], truth - pred, f"truth - PI-ZSSR ({rule_name})", dl, "RdBu_r")
        if tik is not None:
            show(ax[1, 1], tik, f"Tikhonov-FFT (alpha by discrepancy)  RMSE {r['tik_disc_rmse']:.2f}", lim)
            show(ax[1, 2], truth - tik, "truth - Tikhonov-FFT (alpha by discrepancy)", dl, "RdBu_r")
        for a in ax.ravel():
            a.add_patch(plt.Rectangle((z["lons"][20], z["lats"][20]), z["lons"][-21] - z["lons"][20], z["lats"][-21] - z["lats"][20],
                                      fill=False, ls="--", lw=0.8, color="k"))
        plt.tight_layout()
        fo = os.path.join(args.dir, f"E2_fields_{r['run']}.png"); plt.savefig(fo, dpi=120); plt.close()
    print("field maps written")


if __name__ == "__main__":
    main()
