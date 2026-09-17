"""
E3: dual-constraint ablation on EGM residuals with real survey coverage.

  A / C / D  x  seeds 0,1,2    301^2    discrepancy stopping

  python run_e3.py
"""
import argparse
import os
import subprocess
import sys

PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="e3_r01")
    ap.add_argument("--configs", default="A,C,D")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--size", type=int, default=301)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--file_0", default=None, help="passed through to blind_pisr_e3.py")
    ap.add_argument("--file_1500", default=None)
    ap.add_argument("--file_5000", default=None)
    ap.add_argument("--mask_low", default=None)
    ap.add_argument("--mask_high", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    passthrough = []
    for k in ("file_0", "file_1500", "file_5000", "mask_low", "mask_high"):
        v = getattr(args, k)
        if v is not None:
            passthrough += [f"--{k}", v]
    os.makedirs(args.dir, exist_ok=True)

    jobs = []
    for cfg in args.configs.split(","):
        for s in [int(x) for x in args.seeds.split(",")]:
            tag = f"{cfg}_s{s}"
            prefix = os.path.join(args.dir, tag)
            if os.path.exists(prefix + "_metrics.json"):
                print(f"skip {tag}")
                continue
            cmd = [PY, "blind_pisr_e3.py", "--config", cfg, "--seed", str(s),
                   "--size", str(args.size), "--epochs", str(args.epochs),
                   "--noise", str(args.noise), "--stop_rule", "disc",
                   "--eval_every", "1", "--prefix", prefix] + passthrough
            jobs.append((tag, cmd))
    print(f"{len(jobs)} runs")
    for i, (tag, cmd) in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {tag} ===", flush=True)
        if args.dry:
            print(" ".join(cmd))
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!!! {tag} failed ({r.returncode})", flush=True)


if __name__ == "__main__":
    main()
