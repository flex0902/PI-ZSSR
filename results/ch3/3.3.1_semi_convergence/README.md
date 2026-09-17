# 3.3.1 Semi-convergence and stopping rule

Full-grid 5000 → 1500 m on the 301² residual, three seeds. Paper Table 1 and Figure 4 use the **e2** runs (`holdout_frac=0`). The hold-out rejection paragraph uses the **e1** runs (`holdout_frac=0.2`, 15 × 15 blocks).

## Inputs
- `TW_EGM_grid_g_res_h5000.xyz`
- `TW_EGM_grid_g_res_h1500.xyz`

## Code
- `../../00_shared/code/blind_pisr_v2.py` — solver
- `code/run_e2.py` — batch driver (full grid)
- `code/plot_e1_curves.py` — Figure 4

## Commands (from project root)
```
python run_e2.py --noises 0,1 --types white,corr --dir e2
python plot_e1_curves.py --dir e2 --pattern white1_s* --out e2/E1_white1_curves
python plot_e1_curves.py --dir e2 --pattern corr1_s* --out e2/E1_corr1_curves
python blind_pisr_v2.py --noise 1 --holdout_frac 0.2 --holdout_block 15 --no_stop --seed 0 --prefix e1/white1_s0
```

## Outputs
| Path | Role |
|---|---|
| `outputs/full_grid/` | e2 noise-free, white 1 mGal, corr 1 mGal: `*_metrics.json`, `*_curve.csv`, `*_1500m.xyz`, baselines, Figure 4 source curves |
| `outputs/holdout/` | e1 20 % spatial hold-out (white/corr 1 mGal) |
| `outputs/figures/Fig04_learning_curves.png` | manuscript Figure 4 |
