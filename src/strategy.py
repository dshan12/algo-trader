# src/strategy.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint


class CrossSectionalReversal:
    """
    Your original bottom‑k contrarian:
    lookback days to compute return, long the worst bottom_k names.
    """

    def __init__(self, lookback: int = 21, bottom_k: int = 5):
        self.lookback = lookback
        self.bottom_k = bottom_k

    def generate_weights(self, price_df: pd.DataFrame) -> pd.Series:
        past_ret = price_df.pct_change(self.lookback).iloc[-1].dropna()
        losers = past_ret.nsmallest(self.bottom_k).index
        w = pd.Series(0.0, index=past_ret.index)
        w.loc[losers] = 1.0 / self.bottom_k
        return w


class WeeklyReversal:
    """Long bottom_k worst 1‑week performers, short top_k best."""

    def __init__(self, lookback=5, top_k=10, bottom_k=10):
        self.lookback, self.top_k, self.bottom_k = lookback, top_k, bottom_k

    def generate_weights(self, price_df: pd.DataFrame) -> pd.Series:
        ret = price_df.pct_change(self.lookback).iloc[-1].dropna()
        longs = ret.nsmallest(self.bottom_k).index
        shorts = ret.nlargest(self.top_k).index
        w = pd.Series(0.0, index=ret.index)
        w.loc[longs] = 1.0 / self.bottom_k
        w.loc[shorts] = -1.0 / self.top_k
        return w


class PairsTrading:
    """Trade one top cointegrated pair per rebalance to mean‑revert."""

    def __init__(self, lookback=252, z_thresh=2.0):
        self.lookback = lookback
        self.z_thresh = z_thresh
        self.pair = None

    def _select_pair(self, price_df: pd.DataFrame):
        best_pair, best_stat = None, -np.inf
        cols = price_df.columns
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                t_stat, pvalue, crit_vals = coint(price_df[a], price_df[b])
                if t_stat > best_stat:
                    best_stat, best_pair = t_stat, (a, b)
        return best_pair

    def generate_weights(self, price_df: pd.DataFrame) -> pd.Series:
        if self.pair is None or len(price_df) < self.lookback:
            self.pair = self._select_pair(price_df.tail(self.lookback))
        a, b = self.pair
        spread = price_df[a] - price_df[b]
        z = (spread.iloc[-1] - spread.mean()) / spread.std()
        w = pd.Series(0.0, index=price_df.columns)
        if z > self.z_thresh:
            w[a], w[b] = -0.5, 0.5
        elif z < -self.z_thresh:
            w[a], w[b] = 0.5, -0.5
        return w


class CoveredCalls:
    """Overlay: write weekly ATM calls on SPY (modeled as constant yield)."""

    def __init__(self, weekly_premium=0.002):
        self.weekly_premium = weekly_premium

    def generate_weights(self, price_df: pd.DataFrame) -> pd.Series:
        w = pd.Series(0.0, index=price_df.columns)
        if "SPY" in w.index:
            w["SPY"] = 1.0
        return w

    def adjust_equity(self, eq_series: pd.Series) -> pd.Series:
        return eq_series * (1 + self.weekly_premium)


class LowVol:
    """Long equal‑weight lowest‑volatility decile."""

    def __init__(self, lookback=60, decile=0.1):
        self.lookback = lookback
        self.decile = decile

    def generate_weights(self, price_df: pd.DataFrame) -> pd.Series:
        vol = price_df.pct_change().rolling(self.lookback).std().iloc[-1]
        n = max(1, int(len(vol) * self.decile))
        longs = vol.nsmallest(n).index
        w = pd.Series(0.0, index=price_df.columns)
        w.loc[longs] = 1.0 / n
        return w


class CompositeStrategy:
    """
    Blend multiple strategies with fixed weights.
    """

    def __init__(self, strat_weights: dict):
        total = sum(strat_weights.values())
        self.strats = [(s, w / total) for s, w in strat_weights.items()]

    def generate_weights(self, price_df: pd.DataFrame) -> pd.Series:
        w_agg = pd.Series(0.0, index=price_df.columns)
        for strat, α in self.strats:
            w = strat.generate_weights(price_df).reindex(w_agg.index).fillna(0)
            w_agg += α * w
        if w_agg.abs().sum() > 0:
            w_agg /= w_agg.abs().sum()
        return w_agg

    def post_process_equity(self, eq_series: pd.Series) -> pd.Series:
        for strat, _ in self.strats:
            if hasattr(strat, "adjust_equity"):
                eq_series = strat.adjust_equity(eq_series)
        return eq_series
