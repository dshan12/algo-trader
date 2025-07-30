# backtest.py

import pandas as pd
import yfinance as yf
from pathlib import Path
from src.strategy import (
    WeeklyReversal,
    PairsTrading,
    CoveredCalls,
    LowVol,
    CompositeStrategy,
)


def load_universe(symbols, period="2y"):
    """
    Fetch daily Close for all symbols into one DataFrame.
    """
    data = {}
    for s in symbols:
        df = yf.Ticker(s).history(period=period, interval="1d", auto_adjust=False)[
            "Close"
        ]
        if not df.empty:
            data[s] = df
    price_df = pd.DataFrame(data).ffill().dropna()
    return price_df


def simulate_reversal(price_df, strategy, capital=100_000, rebalance_freq="M"):
    """
    price_df: DataFrame of Close prices
    strategy: strategy instance with generate_weights(...) and optional post_process_equity(...)
    rebalance_freq: 'M' month‑end (will alias to 'ME'), '5D' every 5 trading days, etc.
    """
    # alias 'M' → 'ME' to avoid FutureWarning
    freq = rebalance_freq if rebalance_freq != "M" else "ME"
    rebal_dates = price_df.resample(freq).last().index

    dates = price_df.index
    eq = pd.Series(index=dates, dtype=float)
    eq.iloc[0] = capital

    holdings = pd.DataFrame(0.0, index=dates, columns=price_df.columns)

    for i in range(1, len(dates)):
        prev = dates[i - 1]
        today = dates[i]
        eq_prev = eq.loc[prev]

        # carry forward yesterday's positions
        holdings_today = holdings.loc[prev].copy()

        if today in rebal_dates:
            # rebalance at today's open using history up to yesterday
            hist = price_df.loc[:prev]
            w = strategy.generate_weights(hist).reindex(price_df.columns).fillna(0)
            holdings_today = w * eq_prev

        # today's P&L
        daily_ret = price_df.loc[today] / price_df.loc[prev] - 1
        pnl = (holdings_today * daily_ret).sum()

        eq.loc[today] = eq_prev + pnl
        holdings.loc[today] = holdings_today

    # apply any equity adjustments (e.g. covered‑calls premium)
    if hasattr(strategy, "post_process_equity"):
        eq = strategy.post_process_equity(eq)

    return eq


if __name__ == "__main__":
    # 1) Load your 40‑stock universe
    symbols = [
        s.strip() for s in Path("symbols.txt").read_text().splitlines() if s.strip()
    ]
    prices2 = load_universe(symbols, period="2y")
    prices5 = prices2.tail(25)  # last 25 trading days ≈ 5 weeks

    # 2) Instantiate sub‑strategies
    rev = WeeklyReversal(lookback=5, top_k=10, bottom_k=10)
    pairs = PairsTrading(lookback=252, z_thresh=2.0)
    cc = CoveredCalls(weekly_premium=0.002)
    lv = LowVol(lookback=60, decile=0.1)

    # 3) Composite blend (weights normalized internally)
    blend = CompositeStrategy(
        {
            rev: 0.30,
            pairs: 0.30,
            cc: 0.20,
            lv: 0.20,
        }
    )

    # 4) Backtest over 2 years (month‑end)
    eq2 = simulate_reversal(prices2, blend, capital=100_000, rebalance_freq="ME")
    final2 = eq2.iloc[-1]
    pct2 = (final2 / 100_000 - 1) * 100
    print(f"2 Year Composite → ${final2:,.2f} ({pct2:+.2f}%)")

    # 5) Backtest last 5 weeks (every 5 trading days)
    eq5 = simulate_reversal(prices5, blend, capital=100_000, rebalance_freq="5D")
    final5 = eq5.iloc[-1]
    pct5 = (final5 / 100_000 - 1) * 100
    print(f"5 Week Composite → ${final5:,.2f} ({pct5:+.2f}%)")

    # 6) True 5‑week SPY buy‑&‑hold by slicing SPY’s full history
    #    to the same dates as prices5
    spy_full = (
        yf.Ticker("SPY")
        .history(period="2y", interval="1d", auto_adjust=False)["Close"]
        .dropna()
    )
    spy5 = spy_full.loc[prices5.index[0] : prices5.index[-1]]
    if len(spy5) >= 2:
        bh5 = (spy5.iloc[-1] / spy5.iloc[0] - 1) * 100
        print(f"5 Week SPY B&H → {bh5:+.2f}%")
    else:
        print("5 Week SPY B&H → data unavailable")
