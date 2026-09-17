# 3.3.2 Noise robustness

White and along-track correlated noise at 0 / 0.5 / 1 / 2 / 3 mGal, three seeds. For σ ≥ 2 mGal the residual was evaluated every epoch (`e2b`); `summarize_e2.py` prefers the finer `e2b` curves and takes the loss-plateau epoch from the long `e2` run.

## Inputs
- `TW_EGM_grid_g_res_h5000.xyz`
- `TW_EGM_grid_g_res_h1500.xyz`

## Code
- `../../00_shared/code/blind_pisr_v2.py`
- `code/run_e2.py`
- `code/summarize_e2.py` → Table 2, Figures 5–6

## Commands (from project root)
```
python run_e2.py --dir e2
python run_e2.py --dir e2b --noises 2,3 --eval_every 1
python summarize_e2.py --dirs e2 e2b --dir e2
```

## Outputs
| Path | Role |
|---|---|
| `outputs/e2/` | all σ, eval every 10 epochs; `E2_table.md/.csv`, field maps |
| `outputs/e2b/` | σ = 2 and 3 mGal, eval every epoch |
| `outputs/figures/Fig05_rmse_vs_noise.png` | Figure 5 |
| `outputs/figures/Fig06_fields_white1.png` | Figure 6 |
