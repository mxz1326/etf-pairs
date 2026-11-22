import numpy as np
import pandas as pd
import pytest

from etfpairs.signal import ar1_half_life, log_spread, rolling_zscore


# z-score on a 4-number series should match a hand calculation
def test_zscore_matches_hand_calculation():
    s = pd.Series([1.0, 2.0, 3.0, 10.0])
    z = rolling_zscore(s, 3)
    w = np.array([2.0, 3.0, 10.0])
    assert z.iloc[3] == pytest.approx((10.0 - w.mean()) / w.std(ddof=1))
    assert z.iloc[:2].isna().all()


# Changing future data must not change past z-scores (no look-ahead)
def test_zscore_never_uses_future_data():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=300).cumsum())
    z1 = rolling_zscore(s, 20)
    s2 = s.copy()
    s2.iloc[200:] += 50.0                       # change only the future
    z2 = rolling_zscore(s2, 20)
    pd.testing.assert_series_equal(z1.iloc[:200], z2.iloc[:200])


# A simulated AR(1) with phi = 0.5 should have a half-life of 1 day
def test_half_life_of_known_ar1():
    rng = np.random.default_rng(1)
    phi, n = 0.5, 20000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    assert ar1_half_life(pd.Series(x)) == pytest.approx(1.0, rel=0.05)   # log(0.5)/log(0.5)


# Spread should rise when IOZ beats STW
def test_spread_rises_when_first_fund_outperforms():
    tr = pd.DataFrame({"IOZ": [1.0, 1.02], "STW": [1.0, 1.01]})
    s = log_spread(tr)
    assert s.iloc[1] > s.iloc[0]