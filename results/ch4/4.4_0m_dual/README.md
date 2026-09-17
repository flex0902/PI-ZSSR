# 4.4 Unobserved 0 m (dual constraint)

Same three-layer fill on the **1620 m** LSC grid (bias 0.07, scatter 4.09 on Ω_1620). Independent 5156 m LSC is examiner 2 only — not merged into the input. Configurations A (*w*₂ = 0) and C (*w*₁ = *w*₂ = 1). σ_1620 = 0.9 mGal, σ_5156 = 0.7 mGal. Restore: residual + EGM2008 *n* ≤ 2000 at 0 m.

## Inputs
- unfilled 1620 / 5156 m LSC grids
- `TW_EGM_grid_g_res_h1620.xyz` (1620 m fill)
- `TW_EGM_grid_g_d2000_h{0,1620,5156}.xyz` (restore)

## Code
- `fill_grid_gaps.py`, `blind_pisr_real0.py`, `restore_egm.py`, `plot_ch4_44_figures.py`

## Commands (from project root)
```
python fill_grid_gaps.py --lsc_grid "1500(10s)_new.xyg_pred_grid_0.1.xyz" --egm_grid TW_EGM_grid_g_res_h1620.xyz --output ch4/1500_pred_grid_0.1_filled_feather20.xyz --feather_px 20 --smooth 0 --plot
python blind_pisr_real0.py --config A --eval_every 1 --prefix ch4/real0_A
python blind_pisr_real0.py --config C --eval_every 1 --prefix ch4/real0_C
python restore_egm.py --resg ch4/real0_C_0m.xyz --egm TW_EGM_grid_g_d2000_h0.xyz --output ch4/real0_C_full_g_h0.xyz --plot
python restore_egm.py --resg ch4/real0_A_0m.xyz --egm TW_EGM_grid_g_d2000_h0.xyz --output ch4/real0_A_full_g_h0.xyz --plot
python restore_egm.py --resg ch4/1500_pred_grid_0.1_filled_feather20.xyz --egm TW_EGM_grid_g_d2000_h1620.xyz --output ch4/real0_lsc_full_g_h1620.xyz --plot
python restore_egm.py --resg "5000(10s)_new.xyg_pred_grid_0.1.xyz" --egm TW_EGM_grid_g_d2000_h5156.xyz --output ch4/real0_lsc_full_g_h5156.xyz --plot
python plot_ch4_44_figures.py
```

Paper: A stops at epoch 20 (fit 0.90 / 3.02 mGal); C at 230 (0.69 / 1.38 mGal, residual floor at 5156 m). Residual evaluated every epoch (`--eval_every 1`, now the script default; the earlier default of 5 gave identical stopping epochs and fields because both stops fell on multiples of 5).

## Outputs
Filled 1620 m prior; `real0_{A,C}_*` residuals, metrics, fields; restored full-*g*; Figures 18–19.
