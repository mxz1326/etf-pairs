"""Load prices and dividends, and build total-return series."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"

# STW's Yahoo dividend history is missing the Mar-2015, Sep-2015 and Mar-2016 distributions,
# so the study starts after the last gap.
STUDY_START = "2016-04-01"


def load_prices(path=DATA / "prices.csv") -> pd.DataFrame:
    """Daily closes, indexed by date. Days where either ETF has no close are dropped."""
    px = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    return px.dropna()


def load_dividends(path=DATA / "dividends.csv") -> pd.DataFrame:
    """Ex-dates and cash amounts. Exact duplicates within 5 days are removed (Yahoo artefact)."""
    dv = pd.read_csv(path, parse_dates=["ex_date"]).sort_values(["ticker", "ex_date"])
    keep = []
    for _, g in dv.groupby("ticker"):
        last = None
        for idx, row in g.iterrows():
            if last is not None and row.amount == last.amount and (row.ex_date - last.ex_date).days <= 5:
                continue  # same amount listed twice a few days apart: keep the first
            keep.append(idx)
            last = row
    return dv.loc[keep].reset_index(drop=True)


def dividends_on_trading_days(divs: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cash dividend per share credited on each trading day (ex-date, or next trading day)."""
    out = pd.DataFrame(0.0, index=index, columns=sorted(divs.ticker.unique()))
    for _, row in divs.iterrows():
        pos = index.searchsorted(row.ex_date)
        if pos < len(index):
            out.iloc[pos, out.columns.get_loc(row.ticker)] += row.amount
    return out


def total_return_index(px: pd.DataFrame, div_cash: pd.DataFrame) -> pd.DataFrame:
    """Total-return index starting at 1: TR_t = TR_{t-1} * (P_t + D_t) / P_{t-1}."""
    d = div_cash.reindex(px.index).fillna(0.0)[px.columns]
    gross = (px + d) / px.shift(1)
    return gross.fillna(1.0).cumprod()


def daily_returns(tr: pd.DataFrame) -> pd.DataFrame:
    return tr.pct_change().iloc[1:]


def load_study_data(start: str = STUDY_START):
    """Prices, dividend cash on trading days, total-return index and daily returns from `start`."""
    px = load_prices()
    divs = load_dividends()
    cash = dividends_on_trading_days(divs, px.index)
    tr = total_return_index(px, cash)
    tr = tr.loc[start:] / tr.loc[start:].iloc[0]
    ret = daily_returns(tr)
    return px.loc[start:], cash.loc[start:], tr, ret


def tracking_difference(tr: pd.DataFrame, a: str = "IOZ", b: str = "STW") -> float:
    """Annualised total-return difference a - b over the whole sample, in basis points."""
    years = (tr.index[-1] - tr.index[0]).days / 365.25
    return float((np.log(tr[a].iloc[-1]) - np.log(tr[b].iloc[-1])) / years * 1e4)