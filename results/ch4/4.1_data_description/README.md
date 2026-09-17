# 4.1 Data description

Taiwan airborne surveys (Hwang et al., 2007; 2012; 2014): 5156 m over the island, 1620 m over the Strait and Kuroshio. Flight-line files are along-track filtered residuals before LSC.

## Inputs
| File | Role |
|---|---|
| `5000(10s).xyg` | 5156 m flight lines |
| `1500(10s).xyg` | 1620 m flight lines |
| `TW_XGM_grid_g_d2160_h{5156,1620}.xyz` | line-bias comparison surfaces |

## Code
- `code/analyze_obs_vs_egm.py` — line-wise comparison / bias check used to write `*_new.xyg`

## Commands (from project root)
```
python analyze_obs_vs_egm.py --obs 5000(10s).xyg --egm TW_XGM_grid_g_d2160_h5156.xyz --output 5000(10s)_new.xyg
python analyze_obs_vs_egm.py --obs 1500(10s).xyg --egm TW_XGM_grid_g_d2160_h1620.xyz --output 1500(10s)_new.xyg
```

## Outputs
`5000(10s)_new.xyg`, `1500(10s)_new.xyg` — flight residuals used by Section 4.2. The XGM grids at survey height are the comparison surfaces of that step, not the RCR remove (that is EGM2008 *n* ≤ 2000).
