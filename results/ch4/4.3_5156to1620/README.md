# 4.3 5156 → 1620 m (labelled test)

Three-layer fill of the 5156 m LSC grid (Ω / 20-px cosine / bias-corrected EGM residual). PI-ZSSR with *L_z* and discrepancy only on Ω, σ = 0.7 mGal (LSC pred-std median). Independent 1620 m LSC scores the common pixels Ω_5156 ∩ Ω_1620 (*n* = 21224). LSC-only continuation uses the 5156 m T–R model at prediction height 1620 m.

## Inputs
- unfilled LSC grids of Section 4.2
- `TW_EGM_grid_g_res_h5156.xyz` (far-field fill)
- `tr_parameters_optimized_h5156.json`

## Code
- `fill_grid_gaps.py`, `blind_pisr_v2.py`, `lsc_downward_continue.py`
- `find_common_grid_points.py`, `mask_gravity.py`, `calculate_gravity_difference.py`
- `plot_ch4_43_figures.py` → Figures 15–17

## Commands (from project root)
```
python find_common_grid_points.py --file1 "1500(10s)_new.xyg_pred_grid_0.1.xyz" --file2 "5000(10s)_new.xyg_pred_grid_0.1.xyz" --output ch4/check_grid_overlap_lsc.xyz
python mask_gravity.py --grid "5000(10s)_new.xyg_pred_grid_0.1.xyz" --full "1500(10s)_new.xyg_pred_grid_0.1.xyz" --output ch4/1500_lsc_on_omega5156.xyz
python fill_grid_gaps.py --lsc_grid "5000(10s)_new.xyg_pred_grid_0.1.xyz" --egm_grid TW_EGM_grid_g_res_h5156.xyz --output ch4/5000_pred_grid_0.1_filled_feather20.xyz --feather_px 20 --smooth 0 --plot
python blind_pisr_v2.py --file_high ch4/5000_pred_grid_0.1_filled_feather20.xyz --obs_mask ch4/5000_pred_grid_0.1_filled_feather20_omega.xyz --file_low "1500(10s)_new.xyg_pred_grid_0.1.xyz" --obs_h 5156 --target_h 1620 --size 301 --margin 20 --epochs 6000 --gauss_km 1.8 --lam_tik 0.1 --holdout_frac 0 --stop_rule disc --sigma_disc 0.7 --eval_every 1 --noise 0 --prefix ch4/pizssr_5156to1620
python lsc_downward_continue.py --input ch4/5000_pred_grid_0.1_filled_feather20.xyz --output ch4/lsc_dc_5156to1620.xyz --params tr_parameters_optimized_h5156.json --obs_h 5156 --pred_h 1620 --max_dist_deg 0.1 --noise_std 3.0
python calculate_gravity_difference.py --truth ch4/1500_lsc_on_omega5156.xyz --pred ch4/pizssr_5156to1620_1620m.xyz --output_prefix ch4/diff_pizssr --common ch4/check_grid_overlap_lsc.xyz
python calculate_gravity_difference.py --truth ch4/1500_lsc_on_omega5156.xyz --pred ch4/lsc_dc_5156to1620.xyz --output_prefix ch4/diff_lscdc --common ch4/check_grid_overlap_lsc.xyz
python plot_ch4_43_figures.py
```
Optional curve (no early stop, 2000 epochs): `--stop_rule none --eval_every 25 --prefix ch4/pizssr_5156to1620_curve`.

Paper scores: PI-ZSSR RMSE 3.55 mGal (mean +0.03); LSC-only 3.59 mGal (mean +0.18). Stop at epoch 41, fit on Ω = 0.59 mGal, residual evaluated every epoch (`--eval_every 1`; the script default of 25 gives epoch 50 / 3.64 mGal and is *not* the paper protocol).

Sigma sensitivity (`outputs/sigma_sensitivity/`, same command with `--sigma_disc 0.5|0.7|1.0`): eval every 1 → epochs 46/41/35, RMSE 3.67/3.55/3.42 mGal; eval every 25 → 3.82/3.64/3.64. `curve_eval1_*`: 400 epochs without stopping, per-epoch fit and truth RMSE (minimum 3.38 mGal at epoch 33, fit 1.20 mGal).

## Outputs
Filled 5156 m grid + Ω mask; PI-ZSSR and LSC-only 1620 m fields; common-point diffs; Figures 15–17.
