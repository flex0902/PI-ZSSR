"""
E1 figure: fit / hold-out / truth RMSE vs epoch for several seeds, with the epochs selected by the
stopping rules (old loss-plateau, discrepancy, hold-out) and the truth-optimal (oracle) epoch.
Also writes a table: truth RMSE obtained by each rule, per seed and mean +/- std.

Figure conventions
  coloured solid lines  : one run (random seed) each; same colour in every panel
  vertical dashed lines : epoch at which a stopping rule fires (one per run)
  horizontal lines      : reference levels (noise sigma in the fit panel, Tikhonov FFT baseline in the truth panel)
  x axis                : logarithmic (epoch+1) so that the first few hundred epochs are visible

  python plot_e1_curves.py --dir e1 --pattern white1full_s*
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
from matplotlib.lines import Line2D

RULES = [("old", "k", "PI-ZSSR (loss plateau)"),
         ("disc", "m", "PI-ZSSR (discrepancy): fit RMSE = sigma"),
         ("hold", "g", "PI-ZSSR (hold-out minimum)"),
         ("oracle", "r", "PI-ZSSR (oracle): truth minimum")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="e1")
    ap.add_argument("--pattern", default="white1full_s*")
    ap.add_argument("--out", default="")
    ap.add_argument("--linear_x", action="store_true", help="use linear instead of log epoch axis")
    args = ap.parse_args()

    runs = []
    for fn in sorted(glob.glob(os.path.join(args.dir, args.pattern + "_curve.csv"))):
        base = fn[:-len("_curve.csv")]
        met = json.load(open(base + "_metrics.json"))
        runs.append((os.path.basename(base), pd.read_csv(fn), met))
    if not runs:
        print("no runs"); return

    has_holdout = any(np.isfinite(df.rmse_holdout).any() for _, df, _ in runs)
    npan = 3 if has_holdout else 2
    fig, axes = plt.subplots(1, npan, figsize=(6.2 * npan, 5.8))
    ax_fit, ax_truth = axes[0], axes[-1]
    ax_hold = axes[1] if has_holdout else None
    seed_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    xo = 0 if args.linear_x else 1  # epoch offset for log axis

    rows = []
    rule_rmse = {k: [] for k, _, _ in RULES}
    for i, (name, df, met) in enumerate(runs):
        c = seed_cols[i % len(seed_cols)]
        x = df.Epoch + xo
        ax_fit.plot(x, df.rmse_fit, color=c, lw=1.0, alpha=0.7, label=name)
        if ax_hold is not None:
            ax_hold.plot(x, df.rmse_holdout, color=c, lw=1.0, alpha=0.7, label=name)
        ax_truth.plot(x, df.rmse_truth, color=c, lw=1.0, alpha=0.7, label=name)
        row = {"run": name, "sigma_eff": met["sigma_eff"], "truth_amp_ratio": met["truth_amp_ratio"]}
        for key, col, lab in RULES:
            e = met.get(f"{key}_epoch")
            if e is None:
                continue
            r = met.get(f"{key}_rmse_truth")
            row[f"{key}_epoch"] = e; row[f"{key}_rmse_truth"] = r
            rule_rmse[key].append(r)
            if key != "oracle":
                row[f"{key}_amp"] = met.get(f"{key}_amp_ratio")
            # vertical rule lines only in the truth panel; in the fit / hold-out panels mark the firing
            # epoch with a dot ON the curve, so that curve spikes cannot be mistaken for rule lines
            ax_truth.axvline(e + xo, color=col, ls=(0, (6, 3)), lw=1.3, alpha=0.75)
            ax_truth.plot(e + xo, r, "o", color=col, ms=7, mec="k", mew=0.5, zorder=5)
            i_e = int(np.argmin(np.abs(df.Epoch.values - e)))
            ax_fit.plot(e + xo, df.rmse_fit.values[i_e], "o", color=col, ms=6, mec="k", mew=0.5, zorder=5)
            if ax_hold is not None:
                ax_hold.plot(e + xo, df.rmse_holdout.values[i_e], "o", color=col, ms=6, mec="k", mew=0.5, zorder=5)
        row["final_rmse_truth"] = met.get("final_rmse_truth"); row["final_amp"] = met.get("final_amp_ratio")
        if "baseline" in met:
            row["tik_oracle_rmse"] = met["baseline"]["rmse_truth_oracle"]; row["tik_disc_rmse"] = met["baseline"]["rmse_truth_disc"]
            row["tik_disc_amp"] = met["baseline"]["amp_disc"]
        rows.append(row)

    # reference levels
    s = runs[0][2]["sigma_eff"]
    if s > 0:
        ax_fit.axhline(s, color="grey", ls=":", lw=1.2)
        ax_fit.text(0.02, s * 1.05, f"noise sigma = {s:.2f} mGal", transform=ax_fit.get_yaxis_transform(), fontsize=8, color="grey")
    base_handles = []
    if rows and "tik_oracle_rmse" in rows[0]:
        tb = np.mean([r["tik_oracle_rmse"] for r in rows]); td = np.mean([r["tik_disc_rmse"] for r in rows])
        ax_truth.axhline(tb, color="c", ls="-.", lw=1.3); ax_truth.axhline(td, color="c", ls=":", lw=1.3)
        base_handles = [Line2D([], [], color="c", ls="-.", label=f"Tikhonov-FFT (best alpha): {tb:.2f}"),
                        Line2D([], [], color="c", ls=":", label=f"Tikhonov-FFT (alpha by discrepancy): {td:.2f}")]

    # titles with mean rule results
    def m(k):
        v = [x for x in rule_rmse[k] if x is not None]
        return f"{np.mean(v):.2f}" if v else "n/a"
    ax_fit.set_title("Fit: RMSE between upward-continued prediction\nand the noisy observed input (mGal)")
    if ax_hold is not None:
        ax_hold.set_title("Hold-out: same RMSE, but on the hidden blocks (mGal)")
    ax_truth.set_title(f"Accuracy: RMSE vs hidden truth (mGal)\n"
                       f"mean over runs -> loss plateau {m('old')} | discrepancy {m('disc')} | "
                       f"{'hold-out ' + m('hold') + ' | ' if has_holdout else ''}oracle {m('oracle')}", fontsize=10)

    for a in axes:
        a.set_xlabel("epoch" + ("" if args.linear_x else " (log scale)")); a.grid(alpha=0.3, which="both")
        if not args.linear_x:
            a.set_xscale("log")
            epochs_lab = [0, 10, 100, 1000]
            emax = int(max(df.Epoch.max() for _, df, _ in runs))
            if emax > 3000:
                epochs_lab.append(emax + 1)
            a.set_xticks([e + xo for e in epochs_lab]); a.set_xticklabels([str(e) for e in epochs_lab])
            a.set_xticks([], minor=True)
    ax_fit.set_yscale("log")
    ax_fit.set_ylabel("mGal"); ax_truth.set_ylabel("mGal")

    # legends: seeds (lines), rules (vertical dashed), baseline (horizontal)
    seed_handles = [Line2D([], [], color=seed_cols[i % len(seed_cols)], lw=1.4, label=f"seed {met['seed']}") for i, (name, _, met) in enumerate(runs)]
    rule_dot_handles = [Line2D([], [], marker="o", color="w", mfc=c, mec="k", ms=7, label=l) for k, c, l in RULES if rule_rmse[k]]
    rule_line_handles = [Line2D([], [], color=c, ls=(0, (6, 3)), lw=1.3, label=l) for k, c, l in RULES if rule_rmse[k]]
    leg1 = ax_fit.legend(handles=seed_handles, fontsize=8, title="solid lines = random seeds, same noise level", title_fontsize=8, loc="upper right")
    ax_fit.add_artist(leg1)
    ax_fit.legend(handles=rule_dot_handles, fontsize=8, title="dot = epoch at which a stopping rule fires", title_fontsize=8, loc="lower left")
    if ax_hold is not None:
        ax_hold.legend(handles=rule_dot_handles, fontsize=8, title="dot = epoch at which a stopping rule fires", title_fontsize=8, loc="upper right")
    ax_truth.legend(handles=rule_line_handles + base_handles, fontsize=8,
                    title="vertical dashed + dot = stopping rule fires", title_fontsize=8, loc="upper right")

    plt.tight_layout()
    out = args.out or os.path.join(args.dir, f"E1_{args.pattern.replace('*','').rstrip('_s')}_curves.png")
    plt.savefig(out, dpi=140); plt.close()

    tab = pd.DataFrame(rows)
    tab.to_csv(out.replace(".png", ".csv"), index=False, float_format="%.3f")
    cols = [c for c in tab.columns if c.endswith("_epoch") or c.endswith("_rmse_truth") or c in ("final_rmse_truth", "tik_oracle_rmse", "tik_disc_rmse")]
    print(tab[["run"] + cols].round(3).to_string(index=False))
    num = tab[cols].apply(pd.to_numeric, errors="coerce")
    print("\nmean:\n", num.mean().round(3).to_string())
    print("std:\n", num.std().round(3).to_string())
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

