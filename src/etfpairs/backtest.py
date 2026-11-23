"""Turn the z-score into positions and positions into returns after trading costs.

Conventions
-----------
* Position +1 = long IOZ / short STW (A$1 each side). Position -1 = the reverse.
* The z-score is known at the close of day t. With lag=1 (the default) the trade is done at the
  close of day t+1, so the position earns the spread return from t+1 to t+2. With lag=0 the
  trade is done at the same close the signal was computed from, which is not achievable in
  practice and is shown only as an upper bound.
* Cost: `cost_bp` per leg per unit of position change, i.e. half the bid-ask spread plus
  brokerage. A full round trip (in and out) costs 4 * cost_bp: two legs, twice.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Enter when |z| passes the entry level, exit when z crosses back to 0
def positions_from_z(z: pd.Series, entry: float, exit: float = 0.0) -> pd.Series:
    """Mean-reversion state machine.

    Flat -> short the spread (-1) when z > entry, long (+1) when z < -entry.
    Short -> flat when z <= exit. Long -> flat when z >= -exit.
    A position can flip directly if z jumps through the opposite entry band.
    """
    pos = np.zeros(len(z))
    cur = 0.0
    for i, v in enumerate(z.values):
        if np.isnan(v):
            cur = 0.0
        elif cur == 0.0:
            if v > entry:
                cur = -1.0
            elif v < -entry:
                cur = 1.0
        elif cur == -1.0:
            if v < -entry:
                cur = 1.0
            elif v <= exit:
                cur = 0.0
        elif cur == 1.0:
            if v > entry:
                cur = -1.0
            elif v >= -exit:
                cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=z.index, name="position")


# Daily profit after a trading delay of `lag` days, minus costs on every position change
def strategy_returns(pos: pd.Series, spread_ret: pd.Series, cost_bp: float, lag: int = 1) -> pd.DataFrame:
    """Daily gross and net returns per A$1 of position on each leg.

    `pos` is the decided position at each close; `spread_ret` on day t is r_IOZ,t - r_STW,t
    (return from close t-1 to close t).
    """
    held = pos.shift(lag).fillna(0.0)            # position actually traded at close, after the lag
    earn = held.shift(1).fillna(0.0)             # position held over (t-1, t]
    gross = earn * spread_ret
    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * 2 * cost_bp / 1e4          # two legs
    out = pd.DataFrame({"held": held, "gross": gross, "cost": cost})
    out["net"] = out.gross - out.cost
    return out


# Summary statistics for one backtest
@dataclass
class Stats:
    ann_ret_bp: float
    ann_vol_bp: float
    sharpe: float
    t_stat: float
    trades: int
    days_in_market_pct: float
    avg_hold_days: float
    gross_bp_per_trade: float

    def as_dict(self):
        return self.__dict__.copy()


# Annualised return, volatility, Sharpe, t-stat and trade count
def summarise(res: pd.DataFrame, col: str = "net") -> Stats:
    r = res[col]
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    ann_ret = mu * 252 * 1e4
    ann_vol = sd * np.sqrt(252) * 1e4
    sharpe = (mu / sd) * np.sqrt(252) if sd > 0 else float("nan")
    t_stat = mu / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    held = res["held"]
    entries = int(((held != 0) & (held.shift(1).fillna(0) != held)).sum())
    in_mkt = float((held != 0).mean() * 100)
    avg_hold = float((held != 0).sum() / entries) if entries else float("nan")
    gross_per_trade = float(res["gross"].sum() / entries * 1e4) if entries else float("nan")
    return Stats(ann_ret, ann_vol, sharpe, t_stat, entries, in_mkt, avg_hold, gross_per_trade)


# Cost per leg (bp) at which the strategy would exactly break even
def breakeven_cost_bp(res: pd.DataFrame) -> float:
    """Cost per leg (bp) at which net return over the period is exactly zero."""
    units = res["held"].diff().abs().fillna(res["held"].abs()).sum() * 2
    return float(res["gross"].sum() / units * 1e4) if units else float("nan")