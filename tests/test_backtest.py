import numpy as np
import pandas as pd
import pytest

from etfpairs.backtest import breakeven_cost_bp, positions_from_z, strategy_returns, summarise
from etfpairs.signal import rolling_zscore


# Entries, exits, flips and NaN handling on a hand-made z-score series
def test_state_machine_hand_example():
    z = pd.Series([0.0, 2.1, 1.0, -0.1, -2.2, -0.5, 0.3, np.nan, 2.5, 3.0, -2.6])
    pos = positions_from_z(z, entry=2.0, exit=0.0)
    #             0    2.1   1.0  -0.1 -2.2 -0.5  0.3  nan  2.5   3.0  -2.6
    expected = [0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0]
    assert pos.tolist() == expected


# With a one-day lag, a signal at close t only earns returns from close t+1 onwards
def test_lag_one_earns_the_return_two_closes_later():
    idx = pd.RangeIndex(6)
    pos = pd.Series([0, 1, 1, 0, 0, 0], index=idx, dtype=float)      # decided at close 1, out at close 3
    sret = pd.Series([0, 0.01, 0.02, 0.03, 0.04, 0.05], index=idx)  # return over (t-1, t]
    res = strategy_returns(pos, sret, cost_bp=0.0, lag=1)
    # traded at close 2, held over (2,3] and (3,4], closed at close 4
    assert res["gross"].tolist() == pytest.approx([0, 0, 0, 0.03, 0.04, 0])
    res0 = strategy_returns(pos, sret, cost_bp=0.0, lag=0)
    assert res0["gross"].tolist() == pytest.approx([0, 0, 0.02, 0.03, 0, 0])


# Changing future prices must not change past positions (no look-ahead)
def test_positions_never_use_future_data():
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(size=400).cumsum() * 1e-3)
    p1 = positions_from_z(rolling_zscore(s, 20), 1.5)
    s2 = s.copy()
    s2.iloc[250:] -= 0.5
    p2 = positions_from_z(rolling_zscore(s2, 20), 1.5)
    pd.testing.assert_series_equal(p1.iloc[:250], p2.iloc[:250])


# Costs never add profit, and one round trip costs 4 x the per-leg cost
def test_costs_only_reduce_returns_and_round_trip_costs_four_times():
    idx = pd.RangeIndex(5)
    pos = pd.Series([1, 1, 0, 0, 0], index=idx, dtype=float)
    sret = pd.Series([0.0, 0.001, -0.002, 0.0, 0.0], index=idx)
    res = strategy_returns(pos, sret, cost_bp=2.0, lag=0)
    assert (res["net"] <= res["gross"] + 1e-15).all()
    assert res["cost"].sum() == pytest.approx(4 * 2.0 / 1e4)          # in and out, two legs


# Breakeven cost matches a hand calculation, and two entries count as two trades
def test_breakeven_cost_and_trade_count():
    idx = pd.RangeIndex(6)
    pos = pd.Series([1, 1, 0, -1, 0, 0], index=idx, dtype=float)
    sret = pd.Series([0, 0.001, 0.001, 0.0, -0.002, 0.0], index=idx)
    res = strategy_returns(pos, sret, cost_bp=0.0, lag=0)
    units = 1 + 1 + 1 + 1                                             # in, out, in, out
    assert breakeven_cost_bp(res) == pytest.approx(res["gross"].sum() / (2 * units) * 1e4)
    assert summarise(res).trades == 2