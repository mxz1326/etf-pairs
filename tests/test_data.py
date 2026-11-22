from pathlib import Path

import pandas as pd
import pytest

#from etfpairs.data import dividends_on_trading_days, load_dividends, total_return_index

# Import the functions under test, including the new data checks
from etfpairs.data import (dividend_blackout, dividends_on_trading_days, flag_suspect_days,
                           load_dividends, total_return_index)

ROOT = Path(__file__).resolve().parents[1]


def test_price_file_spot_checks():
    """Guards against the data file being edited or truncated by accident."""
    px = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"], index_col="date")
    assert len(px) == 3011
    for d, ioz, stw in (("2015-01-02", 23.43, 50.40), ("2020-03-23", 18.80, 43.14),
                        ("2024-11-14", 33.20, 74.52), ("2026-11-19", 35.06, 78.73)):
        assert px.loc[d, "IOZ"] == pytest.approx(ioz)
        assert px.loc[d, "STW"] == pytest.approx(stw)


def test_total_return_hand_example():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    px = pd.DataFrame({"A": [10.0, 9.9, 10.1]}, index=idx)
    cash = pd.DataFrame({"A": [0.0, 0.2, 0.0]}, index=idx)
    tr = total_return_index(px, cash)
    assert tr["A"].iloc[0] == pytest.approx(1.0)
    assert tr["A"].iloc[1] == pytest.approx((9.9 + 0.2) / 10.0)
    assert tr["A"].iloc[2] == pytest.approx((9.9 + 0.2) / 10.0 * 10.1 / 9.9)


def test_dividend_on_non_trading_day_goes_to_next_trading_day():
    idx = pd.to_datetime(["2020-01-03", "2020-01-06"])            # Friday, Monday
    divs = pd.DataFrame({"ticker": ["A", "A"], "ex_date": pd.to_datetime(["2020-01-04", "2020-02-01"]),
                         "amount": [0.5, 0.7]})
    cash = dividends_on_trading_days(divs, idx)
    assert cash.loc["2020-01-06", "A"] == pytest.approx(0.5)
    assert cash["A"].sum() == pytest.approx(0.5)                     # after the last day: dropped


def test_duplicate_dividend_removed(tmp_path):
    f = tmp_path / "d.csv"


# One dividend on day 8 with a 2-day window: only days 6-10 should be blacked out
def test_blackout_marks_days_either_side():
    idx = pd.bdate_range("2020-01-01", periods=15)
    cash = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    cash.iloc[7, 0] = 0.3
    bo = dividend_blackout(idx, cash, days=2)
    assert bo.values.tolist() == [False] * 5 + [True] * 5 + [False] * 5


# A 70bp gap on day 2 should be the only day flagged
def test_flag_suspect_days():
    idx = pd.bdate_range("2020-01-01", periods=3)
    ret = pd.DataFrame({"IOZ": [0.001, 0.010, 0.0], "STW": [0.001, 0.003, 0.0]}, index=idx)
    f = flag_suspect_days(ret)
    assert list(f.index) == [idx[1]]
    assert f["diff_bp"].iloc[0] == pytest.approx(70.0)