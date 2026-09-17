# 4.2 LSC prediction (RCR + T–R + LOO + grid)

Remove EGM2008 *n* ≤ 2000, fit Tscherning–Rapp at survey height (*B* searched; *s* = (*R_B*/(*R*+*h*))²), LOO at the observation points (nugget 3.0 mGal, radius 0.1°), then the same operator on the 301² 1′ grid.

## Inputs
- `5000(10s)_new.xyg`, `1500(10s)_new.xyg`
- `TW_EGM_grid_g_d2000_h5156.xyz`, `TW_EGM_grid_g_d2000_h1620.xyz`

## Code
- `airborne_data_process.py` (`--fit_only`, then `--pred_type all` / `grid`)
- `lsc_covariance_fit.py`, `lsc_covariance_fit_B.py`

## Commands (from project root)
```
python airborne_data_process.py --data 5000(10s)_new.xyg --grid TW_EGM_grid_g_d2000_h5156.xyz --max_dist_deg 0.1 --pred_type all --flight_height 5156 --fit_only
python airborne_data_process.py --data 1500(10s)_new.xyg --grid TW_EGM_grid_g_d2000_h1620.xyz --max_dist_deg 0.1 --pred_type all --flight_height 1620 --fit_only
python airborne_data_process.py --data 5000(10s)_new.xyg --grid TW_EGM_grid_g_d2000_h5156.xyz --max_dist_deg 0.1 --pred_type all --flight_height 5156 --params tr_parameters_optimized_h5156.json
python airborne_data_process.py --data 5000(10s)_new.xyg --grid TW_EGM_grid_g_d2000_h5156.xyz --max_dist_deg 0.1 --pred_type grid --flight_height 5156 --params tr_parameters_optimized_h5156.json
python airborne_data_process.py --data 1500(10s)_new.xyg --grid TW_EGM_grid_g_d2000_h1620.xyz --max_dist_deg 0.1 --pred_type all --flight_height 1620 --params tr_parameters_optimized_h1620.json
python airborne_data_process.py --data 1500(10s)_new.xyg --grid TW_EGM_grid_g_d2000_h1620.xyz --max_dist_deg 0.1 --pred_type grid --flight_height 1620 --params tr_parameters_optimized_h1620.json
```

Fitted parameters: 5156 m *A* ≈ 74.5 mGal², *B* = 2397; 1620 m *A* ≈ 46.2 mGal², *B* = 3641. LOO RMSE 2.07 / 1.73 mGal.

## Outputs
| File | Role |
|---|---|
| `tr_parameters_optimized_h{5156,1620}.json` | T–R *A*, *B*, *R_B* |
| `{5000,1500}(10s)_new.xyg_pred_grid_0.1.xyz` | LSC residual grids (Ω) |
| `{5000,1500}(10s)_new.xyg_pred_all_0.1.csv/.png` | LOO at observation points |
| `*_pred_all_0.1_error.png` | LOO errors (Figure 14) |
| `lsc_covariance_fit_B_optimized_*.png` | Figures 11–12 source |
| `outputs/figures/Fig11`–`Fig14` | manuscript figures |
