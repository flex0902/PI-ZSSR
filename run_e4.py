"""
E4: grid-size consistency with a physically defined (km) Gaussian layer and the discrepancy stopping rule.

  grid sizes 301 / 601 / 901 / 1201 on the same physical field; 1 mGal white noise x 3 seeds, plus a
  noise-free run (seed 0) per size; all rules recorded (--no_stop); Tikhonov-FFT baseline on each grid.

Resumable: runs whose *_metrics.json already exists are skipped.

  python run_e4.py
"""
import argparse
import os
import subprocess
import sys

PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="e4")
    ap.add_argument("--sizes", default="301,601,901,1201")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--file_high", default="TW_EGM_grid_g_res_h5000.xyz")
    ap.add_argument("--file_low", default="TW_EGM_grid_g_res_h1500.xyz")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    jobs = []
    for S in [int(s) for s in args.sizes.split(",")]:
        for nz, seeds in ((0.0, [0]), (args.noise, [int(s) for s in args.seeds.split(",")])):
            for s in seeds:
                tag = f"g{S}_n{nz:g}_s{s}"
                prefix = os.path.join(args.dir, tag)
                if os.path.exists(prefix + "_metrics.json"):
                    print(f"skip {tag} (done)"); continue
                cmd = [PY, "blind_pisr_v2.py", "--size", str(S), "--epochs", str(args.epochs),
                       "--file_high", args.file_high, "--file_low", args.file_low,
                       "--obs_h", "5000", "--target_h", "1500",
                       "--noise", str(nz), "--noise_type", "white", "--seed", str(s),
                       "--holdout_frac", "0", "--no_stop", "--eval_every", str(args.eval_every),
                       "--baseline", "--prefix", prefix, "--tag", f"E4_{tag}"]
                jobs.append((tag, cmd))

    print(f"{len(jobs)} runs to do", flush=True)
    for i, (tag, cmd) in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {tag} ===", flush=True)
        if args.dry:
            print(" ".join(cmd)); continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!!! {tag} failed (code {r.returncode})", flush=True)


if __name__ == "__main__":
    main()
