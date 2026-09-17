# 3.3.4 Grid-size consistency

Same physical field resampled to 301² / 601² / 901² / 1201². Three series: noise-free; 1 mGal white per pixel; PSD-matched σ ∝ 1/Δx (2, 3, 4 mGal). Extra decreasing-noise series (0.5 mGal on 601², 0.33 mGal on 901²) is the residual-floor check in the text.

## Inputs
- `TW_EGM_grid_g_res_h5000.xyz`
- `TW_EGM_grid_g_res_h1500.xyz`

## Code
- `../../00_shared/code/blind_pisr_v2.py`
- `code/run_e4.py`
- `code/summarize_e4.py` → Table 4, Figure 8

## Commands (from project root)
```
python run_e4.py --dir e4
python summarize_e4.py --dir e4
```
PSD-matched and decreasing-noise jobs were extra `blind_pisr_v2.py` calls with `--size` and `--noise` set by hand (tags `g601_n2_*`, `g901_n3_*`, `g1201_n4_*`, `g601_n0.5_*`, `g901_n0.3333_*`).

## Outputs
| Path | Role |
|---|---|
| `outputs/e4/` | `g<size>_n<sigma>_s<seed>_{metrics.json,curve.csv,1500m.xyz,baseline.csv}`, `E4_table.md/.csv`, `E4_vs_grid.png` |
| `outputs/figures/Fig08_grid_spacing.png` | Figure 8 |
