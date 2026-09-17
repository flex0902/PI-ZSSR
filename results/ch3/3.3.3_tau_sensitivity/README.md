# 3.3.3 Sensitivity to the noise estimate

Post-hoc scan of the already-recorded learning curves: first epoch with r_t ≤ τσ for τ ∈ {0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5}.

## Inputs
Learning curves of Section 3.3.2 (`../3.3.2_noise_robustness/outputs/e2` and `e2b`). No new PI-ZSSR training.

## Code
- `code/scan_tau.py`

## Command (from project root)
```
python scan_tau.py --dirs e2 e2b --out e2/tau_scan
```

## Outputs
| File | Role |
|---|---|
| `tau_scan_table.md` | Table 3 |
| `tau_scan_runs.csv` | per seed × τ |
| `tau_scan.png` / `Fig07_tau_scan.png` | Figure 7 |
