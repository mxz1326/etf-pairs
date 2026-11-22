"""The relative-value signal: log total-return spread and its rolling z-score."""
import numpy as np
import pandas as pd


# Log total-return spread: rises when IOZ outperforms STW
def log_spread(tr: pd.DataFrame, a: str = "IOZ", b: str = "STW") -> pd.Series:
    """log(TR_a) - log(TR_b). Rises when `a` outperforms `b`."""
    return np.log(tr[a]) - np.log(tr[b])


# How many standard deviations today's spread is from its recent average (no future data)
def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """z_t = (s_t - mean of s over the last `window` days incl. t) / std over the same days.

    Uses only data up to and including day t, so it can be computed at the close of day t.
    """
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    return (s - m) / sd


# Days for a gap to halve, from an AR(1) fit
def ar1_half_life(x: pd.Series) -> float:
    """Half-life in days from an AR(1) fit x_t = c + phi * x_{t-1} + e_t."""
    x = x.dropna()
    y, lag = x.iloc[1:].values, x.iloc[:-1].values
    X = np.column_stack([np.ones_like(lag), lag])
    c, phi = np.linalg.lstsq(X, y, rcond=None)[0]
    if phi <= 0 or phi >= 1:
        return float("nan")
    return float(np.log(0.5) / np.log(phi))


# Correlation of each day's value with the previous day's (negative = moves reverse)
def lag1_autocorr(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.autocorr(lag=1))