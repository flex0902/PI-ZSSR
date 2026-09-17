"""
E2: noise robustness with the noise-aware stopping rule (replaces Table 2).

  noise levels 0 / 0.5 / 1 / 2 / 3 mGal, white and along-track correlated, 3 seeds each,
  full grid (no hold-out), all stopping rules recorded (--no_stop), Tikhonov-FFT baseline.

Resumable: runs whose *_metrics.json already exists are skipped.

  python run_e2.py                       # everything
  python run_e2.py --types white         # white only
"""
import argparse
import os
import subprocess
import sys

PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="e2")
    ap.add_argument("--size", type=int, default=301)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--noises", default="0,0.5,1,2,3")
    ap.add_argument("--types", default="white,corr")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--corr_len_km", type=float, default=20.0)
    ap.add_argument("--file_high", default="TW_EGM_grid_g_res_h5000.xyz")
    ap.add_argument("--file_low", default="TW_EGM_grid_g_res_h1500.xyz")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    noises = [float(x) for x in args.noises.split(",")]
    types = args.types.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    jobs = []
    for nt in types:
        for nz in noises:
            if nz == 0 and nt == "corr":
                continue  # noise-free case is identical for both types
            for s in seeds:
                tag = f"{nt}{nz:g}_s{s}"
                prefix = os.path.join(args.dir, tag)
                if os.path.exists(prefix + "_metrics.json"):
                    print(f"skip {tag} (done)"); continue
                cmd = [PY, "blind_pisr_v2.py", "--size", str(args.size), "--epochs", str(args.epochs),
                       "--file_high", args.file_high, "--file_low", args.file_low,
                       "--obs_h", "5000", "--target_h", "1500",
                       "--noise", str(nz), "--noise_type", nt, "--corr_len_km", str(args.corr_len_km),
                       "--seed", str(s), "--holdout_frac", "0", "--no_stop", "--eval_every", str(args.eval_every),
                       "--baseline", "--prefix", prefix, "--tag", f"E2_{tag}"]
                jobs.append((tag, cmd))

    print(f"{len(jobs)} runs to do")
    for i, (tag, cmd) in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {tag} ===", flush=True)
        if args.dry:
            print(" ".join(cmd)); continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!!! {tag} failed (code {r.returncode})", flush=True)


if __name__ == "__main__":
    main()
