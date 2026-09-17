# 3.4 Dual physics constraint

Predict unobserved 0 m residual from masked 1500 m and 5000 m EGM residuals. Coverage Ω follows the real Taiwan LSC NaN patterns (values remain synthetic). Configurations A / C / D, three seeds, σ = 1 mGal white on observed pixels only.

## Inputs
| File | Role |
|---|---|
| `TW_EGM_grid_g_res_h0.xyz` | hidden 0 m truth |
| `TW_EGM_grid_g_res_h1500.xyz` | 1500 m values |
| `TW_EGM_grid_g_res_h5000.xyz` | 5000 m values |
| `1500(10s)_new.xyg_pred_grid_0.1.xyz` | Ω_z1 (R1) |
| `5000(10s)_new.xyg_pred_grid_0.1.xyz` | Ω_z2 (R1 ∪ R2); default in `blind_pisr_e3.py`, same 0.1° LSC grid as Section 4.4 (R2 = 10446, R3 = 25734) |

## Code
- `code/blind_pisr_e3.py` — dual-examiner solver
- `code/run_e3.py` — A/C/D × seeds
- `code/summarize_e3.py` — Table 5, Figure 10
- `../../00_shared/code/blind_restoration_pisr.py`, `blind_pisr_v2.py` (noise, Tikhonov helper)

## Command (from project root)
```
python run_e3.py --dir e3_r01
python summarize_e3.py --dir e3_r01
```
(`e3_r01` = run with the 0.1° 5000 m mask; the earlier `e3/` directory used the 0.05° mask and is superseded.)

## Outputs
| Path | Role |
|---|---|
| `outputs/e3/{A,C,D}_s{0,1,2}_{metrics.json,curve.csv,0m.xyz}` | per-run fields and scores |
| `outputs/e3/E3_table.md`, `E3_runs.csv`, `E3_maps_s0.png` | Table 5 and Figure 10 source |
| `outputs/figures/Fig09_dual_workflow.png` | Figure 9 |
| `outputs/figures/Fig10_dual_maps.png` | Figure 10 |
