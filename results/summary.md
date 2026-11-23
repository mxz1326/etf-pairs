# Results

Sample 2016-04-01 to 2026-11-19 (2693 trading days). Parameters chosen on 2016-04-01 to 2021-12-31, tested once from 2022-01-01.

## The spread

- first_day: 2016-04-01
- last_day: 2026-11-19
- trading_days: 2693
- ann_tr_IOZ_pct: 9.405
- ann_tr_STW_pct: 9.476
- tracking_diff_bp_per_yr: -6.416
- daily_diff_std_bp: 23.555
- daily_diff_median_abs_bp: 5.673
- daily_diff_lag1_autocorr: -0.473
- half_life_days_vs_60d_mean: 0.508
- days_abs_diff_over_50bp: 63

## Training grid (net of 2 bp per leg, next-close fill)

| window | entry | ann_ret_bp | sharpe | t_stat | trades | days_in_market_pct | breakeven_cost_bp |
|---|---|---|---|---|---|---|---|
| 20 | 1.00 | -25.28 | -0.09 | -0.21 | 213 | 30.71 | 1.83 |
| 20 | 1.50 | -18.30 | -0.09 | -0.23 | 114 | 15.70 | 1.77 |
| 20 | 2.00 | -13.42 | -0.09 | -0.22 | 59 | 8.64 | 1.67 |
| 20 | 2.50 | -14.69 | -0.11 | -0.27 | 33 | 4.32 | 1.36 |
| 60 | 1.00 | -85.39 | -0.37 | -0.89 | 150 | 22.69 | 1.18 |
| 60 | 1.50 | -69.01 | -0.39 | -0.93 | 78 | 10.97 | 0.72 |
| 60 | 2.00 | -27.54 | -0.18 | -0.43 | 50 | 7.20 | 1.20 |
| 60 | 2.50 | -4.30 | -0.03 | -0.07 | 31 | 4.25 | 1.80 |
| 120 | 1.00 | -86.46 | -0.35 | -0.83 | 113 | 19.74 | 0.89 |
| 120 | 1.50 | -44.84 | -0.29 | -0.69 | 64 | 11.17 | 0.99 |
| 120 | 2.00 | -41.99 | -0.29 | -0.71 | 37 | 6.85 | 0.36 |
| 120 | 2.50 | -3.32 | -0.03 | -0.07 | 26 | 4.46 | 1.81 |

Chosen: window 120 days, entry |z| > 2.5.

## Chosen rule by cost level

| period | cost_bp_per_leg | ann_ret_bp | ann_vol_bp | sharpe | t_stat | trades | gross_bp_per_trade |
|---|---|---|---|---|---|---|---|
| train | 0.00 | 32.60 | 115.08 | 0.28 | 0.68 | 26 | 7.26 |
| test | 0.00 | -8.12 | 35.42 | -0.23 | -0.51 | 21 | -1.90 |
| train | 1.00 | 14.64 | 116.04 | 0.13 | 0.30 | 26 | 7.26 |
| test | 1.00 | -25.26 | 37.81 | -0.67 | -1.48 | 21 | -1.90 |
| train | 2.00 | -3.32 | 117.31 | -0.03 | -0.07 | 26 | 7.26 |
| test | 2.00 | -42.40 | 40.92 | -1.04 | -2.29 | 21 | -1.90 |
| train | 4.00 | -39.25 | 120.74 | -0.33 | -0.78 | 26 | 7.26 |
| test | 4.00 | -76.68 | 48.70 | -1.57 | -3.49 | 21 | -1.90 |

## Test period: fill assumption and dividend blackout

| variant | gross_ann_bp | net_ann_bp | net_sharpe | trades | breakeven_cost_bp |
|---|---|---|---|---|---|
| same-close fill (lag 0), blackout | 169.46 | 134.36 | 1.16 | 22 | 9.66 |
| next-close fill (lag 1), blackout | -8.12 | -42.40 | -1.04 | 21 | -0.47 |
| next-close fill (lag 1), no blackout | -5.97 | -45.96 | -1.04 | 25 | -0.29 |
| same-close fill (lag 0), no blackout | 187.88 | 147.07 | 1.26 | 26 | 9.03 |
