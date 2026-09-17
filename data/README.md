# data/

All files are plain ASCII `lon lat value` on the 21–26°N, 119–124°E window, 1′ spacing (301 × 301 = 90601 rows, longitude fastest). Gravity in mGal; `nan` marks unobserved pixels.

## ggm/ — global-model grids (GrafLab, Bucha and Janák 2013)

| File | Model / band | Height | Used in |
|---|---|---|---|
| `TW_EGM_grid_g_res_h{0,1500,5000}.xyz` | EGM2008, n = 1081–2160 (residual) | 0 / 1500 / 5000 m | §3 synthetic truth and observations |
| `TW_EGM_grid_g_d2160_h{0,5000}.xyz` | EGM2008, n ≤ 2160 | 0 / 5000 m | §3.1 (Figure 3), §4.5 reference |
| `TW_EGM_grid_g_d2000_h{0,1620,5156}.xyz` | EGM2008, n ≤ 2000 (RCR reference) | 0 / 1620 / 5156 m | §4.2 remove, §4.4 restore |
| `TW_EGM_grid_g_res_h{1620,5156}.xyz` | EGM2008, n = 2001–2160 (residual) | 1620 / 5156 m | §4.2 residual comparison, §4.3–4.4 far-field fill |
| `TW_XGM_grid_g_d2160_h{1620,5156}.xyz` | XGM2019e, n ≤ 2160 | 1620 / 5156 m | §4.1 data check |
| `TW_XGM_grid_g_d5540_h0.xyz` | XGM2019e, n ≤ 5540 | 0 m | §4.5 spectral comparison |

Heights are ellipsoidal/orthometric as used in the paper (0 m ≈ quasi-geoid). `res` grids are the full-degree grid minus the reference grid at the same height.

## masks/ — LSC coverage

| File | Content |
|---|---|
| `mask_lsc_1620m_r0.1.xyz` | Ω_1620: 1 where the 1620 m LSC grid (search radius 0.1°) has a prediction, `nan` elsewhere (33028 observed pixels) |
| `mask_lsc_5156m_r0.1.xyz` | Ω_5156: same for the 5156 m survey (31670 observed pixels) |

These files carry only the NaN pattern of the LSC grids, not their values. `blind_pisr_e3.py`, `blind_pisr_real0.py`, `deploy_pisr_ablation.py` and `evaluate_xgm2019.py` read masks through `np.isfinite(...)`, so they accept these files wherever the paper's commands name a `*_pred_grid_0.1.xyz` grid as a mask.

## lsc/ — Tscherning–Rapp parameters

`tr_parameters_optimized_h{5156,1620}.json`: fitted A (mGal²), B (degree) and Bjerhammar-sphere radius R_B for the two surveys (§4.2; used by `lsc_downward_continue.py` in §4.3).

## Not distributed

Taiwan airborne gravity observations (`5000(10s)_new.xyg`, `1500(10s)_new.xyg`), the LSC residual grids (`*_pred_grid_0.1.xyz`), LOO point predictions, and every grid derived from them. See *Data availability* in the top-level README.
