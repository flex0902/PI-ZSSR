# 3.1 Synthetic data generation

## Inputs (external, not in the repo)
- EGM2008 spherical-harmonic coefficients (Pavlis et al., 2012)
- GrafLab (Bucha and Janák, 2013)
- Window: 21–26°N, 119–124°E; 301 × 301, 1′ grid

## Code
- `code/plot_res_g_compare.py` — Figure 3 (5000 m vs 1500 m residual maps)

## Outputs
| File | Role |
|---|---|
| `TW_EGM_grid_g_res_h5000.xyz` | residual at 5000 m |
| `TW_EGM_grid_g_res_h1500.xyz` | residual at 1500 m |
| `TW_EGM_grid_g_res_h0.xyz` | residual at 0 m (Section 3.4 truth) |
| `TW_EGM_grid_g_d2160_h5000.xyz`, `…_h0.xyz` | full-spectrum grids that remain in the project |
| `Fig03_residual_5000_1500.png` | manuscript Figure 3 |

The n≤1080 long-wavelength grids at 5000 / 1500 / 0 m were not retained; only the residuals (and some d2160 grids) are archived.
