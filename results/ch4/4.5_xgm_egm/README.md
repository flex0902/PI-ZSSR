# 4.5 Spectral comparison with XGM2019e and EGM2008

Native 1′ grid of configuration C. Models interpolated onto those nodes (C is not upsampled to 30″). ROI = 20-px margin; 2-D Hann; full *g* and residual after subtracting EGM *n* ≤ 2000. Not a 0 m truth test.

## Inputs
- `real0_C_full_g_h0.xyz`, `real0_A_full_g_h0.xyz` (from Section 4.4)
- `TW_XGM_grid_g_d5540_h0.xyz`, `TW_EGM_grid_g_d2160_h0.xyz`, `TW_EGM_grid_g_d2000_h0.xyz`
- LSC masks for R1/R2/R3

## Code
- `evaluate_xgm2019.py`
- `restore_egm.py` (if A full *g* must be rebuilt)

## Command (from project root)
```
python evaluate_xgm2019.py --pred_file ch4/real0_C_full_g_h0.xyz --pred_b ch4/real0_A_full_g_h0.xyz --mode both --output_prefix ch4/fig20
```

## Outputs
| File | Role |
|---|---|
| `fig20_full_psd_coh.png` | Figure 20 |
| `fig20_residual_CA_psd_coh.png` | Figure 21 |
| `fig20_stats.json` | Table 9 numbers |
| `fig20_full_CA_psd_coh.png`, `fig20_residual_psd_coh.png` | extra panels |
