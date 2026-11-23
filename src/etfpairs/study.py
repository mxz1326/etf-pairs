"""The full study: describe the spread, pick parameters on 2016-2021, test once on 2022-2026."""
from itertools import product

import numpy as np
import pandas as pd

from .backtest import breakeven_cost_bp, positions_from_z, strategy_returns, summarise
from .data import dividend_blackout, load_study_data, tracking_difference
from .signal import ar1_half_life, lag1_autocorr, log_spread, rolling_zscore

# Train/test split, parameter grid and cost assumptions
TRAIN = ("2016-04-01", "2021-12-31")
TEST = ("2022-01-01", "2026-12-31")
WINDOWS = (20, 60, 120)
ENTRIES = (1.0, 1.5, 2.0, 2.5)
BASE_COST_BP = 2.0          # per leg, one way: about half the bid-ask spread plus brokerage
COSTS_BP = (0.0, 1.0, 2.0, 4.0)


# Load data and build the spread, daily spread return and blackout mask
def setup():
    px, cash, tr, ret = load_study_data()
    spread = log_spread(tr)
    spread_ret = (ret["IOZ"] - ret["STW"]).reindex(tr.index).fillna(0.0)
    blackout = dividend_blackout(tr.index, cash)
    return px, cash, tr, ret, spread, spread_ret, blackout


# Backtest one rule: z-score window, entry level, cost, lag and blackout on/off
def run_rule(spread, spread_ret, blackout, window, entry, cost_bp, lag=1, use_blackout=True):
    z = rolling_zscore(spread, window)
    if use_blackout:
        z = z.where(~blackout)          # NaN -> forced flat
    pos = positions_from_z(z, entry)
    return strategy_returns(pos, spread_ret, cost_bp, lag=lag), z, pos


# Slice results to the train or test period
def period(res, start_end):
    return res.loc[start_end[0]:start_end[1]]


# Descriptive statistics of the spread before any trading
def describe(tr, ret, spread):
    d = (ret["IOZ"] - ret["STW"]) * 1e4
    dev = spread - spread.rolling(60).mean()
    return {
        "first_day": tr.index[0].date().isoformat(),
        "last_day": tr.index[-1].date().isoformat(),
        "trading_days": int(len(ret)),
        "ann_tr_IOZ_pct": float((tr["IOZ"].iloc[-1] ** (365.25 / (tr.index[-1] - tr.index[0]).days) - 1) * 100),
        "ann_tr_STW_pct": float((tr["STW"].iloc[-1] ** (365.25 / (tr.index[-1] - tr.index[0]).days) - 1) * 100),
        "tracking_diff_bp_per_yr": tracking_difference(tr),
        "daily_diff_std_bp": float(d.std()),
        "daily_diff_median_abs_bp": float(d.abs().median()),
        "daily_diff_lag1_autocorr": lag1_autocorr(d),
        "half_life_days_vs_60d_mean": ar1_half_life(dev),
        "days_abs_diff_over_50bp": int((d.abs() > 50).sum()),
    }


# Try every window/entry combination on the training period only
def grid_search(spread, spread_ret, blackout, cost_bp=BASE_COST_BP):
    rows = []
    for w, k in product(WINDOWS, ENTRIES):
        res, _, _ = run_rule(spread, spread_ret, blackout, w, k, cost_bp)
        st = summarise(period(res, TRAIN))
        rows.append({"window": w, "entry": k, **st.as_dict(),
                     "breakeven_cost_bp": breakeven_cost_bp(period(res, TRAIN))})
    return pd.DataFrame(rows)


# Pick the best rule on training data, then test it once on 2022-2026
def run_all():
    px, cash, tr, ret, spread, spread_ret, blackout = setup()
    out = {"describe": describe(tr, ret, spread)}

    grid = grid_search(spread, spread_ret, blackout)
    best = grid.sort_values("sharpe", ascending=False).iloc[0]
    w, k = int(best.window), float(best.entry)
    out["grid"] = grid
    out["chosen"] = {"window": w, "entry": k}

    # Train and test at every cost level, with the realistic one-day lag.
    rows = []
    for c in COSTS_BP:
        res, _, _ = run_rule(spread, spread_ret, blackout, w, k, c)
        for name, pe in (("train", TRAIN), ("test", TEST)):
            st = summarise(period(res, pe))
            rows.append({"period": name, "cost_bp_per_leg": c, **st.as_dict()})
    out["by_cost"] = pd.DataFrame(rows)

    # What the timing assumption and the dividend blackout do to the result (test period).
    rows = []
    for label, lag, bo in (("same-close fill (lag 0), blackout", 0, True),
                           ("next-close fill (lag 1), blackout", 1, True),
                           ("next-close fill (lag 1), no blackout", 1, False),
                           ("same-close fill (lag 0), no blackout", 0, False)):
        res, _, _ = run_rule(spread, spread_ret, blackout, w, k, BASE_COST_BP, lag=lag, use_blackout=bo)
        pe = period(res, TEST)
        st_g, st_n = summarise(pe, "gross"), summarise(pe, "net")
        rows.append({"variant": label, "gross_ann_bp": st_g.ann_ret_bp, "net_ann_bp": st_n.ann_ret_bp,
                     "net_sharpe": st_n.sharpe, "trades": st_n.trades,
                     "breakeven_cost_bp": breakeven_cost_bp(pe)})
    out["variants"] = pd.DataFrame(rows)

    # Series for the figures.
    res_base, z, pos = run_rule(spread, spread_ret, blackout, w, k, BASE_COST_BP)
    res_lag0, _, _ = run_rule(spread, spread_ret, blackout, w, k, BASE_COST_BP, lag=0)
    out["series"] = {"spread": spread, "z": z, "pos": pos, "res": res_base, "res_lag0": res_lag0,
                     "spread_ret": spread_ret, "blackout": blackout}
    return out