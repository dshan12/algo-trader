#!/usr/bin/env python3

import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
# from dotenv import load_dotenv

from alpaca_trade_api.rest import REST

# yfinance for DRY‑RUN price fetch
import yfinance as yf

from src.strategy import (
    WeeklyReversal,
    PairsTrading,
    CoveredCalls,
    LowVol,
    CompositeStrategy,
)
from src.github_util import commit_trade

# ─── Setup & Constants ─────────────────────────────────────────────────────────

# When DRY_RUN=1, we never call Alpaca or Github; we just simulate logic.
DRY_RUN = False

# Load Alpaca creds from .env (won’t be used in dry‑run)
# load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

# Universe file
SYMBOL_FILE = Path(__file__).parents[1] / "symbols.txt"
symbols = [s.strip() for s in SYMBOL_FILE.read_text().splitlines() if s.strip()]

# Build your composite strategy
rev = WeeklyReversal(lookback=5, top_k=10, bottom_k=10)
pairs = PairsTrading(lookback=252, z_thresh=2.0)
cc = CoveredCalls(weekly_premium=0.002)
lv = LowVol(lookback=60, decile=0.1)
blend = CompositeStrategy(
    {
        rev: 0.25,
        pairs: 0.25,
        cc: 0.20,
        lv: 0.30,
    }
)

# Trading horizon: today → today+5 weeks
START_DATE = date.today()
END_DATE = START_DATE + timedelta(weeks=5)

# How many days back to fetch via yfinance in DRY‑RUN
# (we only need enough to cover the longest lookback)
DRY_DAYS = 2 * 252  # 2 years


def fetch_universe_history_alpaca(
    api: REST, symbols: list[str], days: int
) -> pd.DataFrame:
    """
    Fetch daily close prices via Alpaca through yesterday.
    """
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)
    price_data = {}

    for sym in symbols:
        try:
            barset = api.get_bars(sym, "1Day", start.isoformat())
            df = barset.df
            if not df.empty:
                # Just take the 'close' column — no symbol filter needed
                price_data[sym] = df["close"]
        except Exception as e:
            print(f"  → failed to fetch {sym} via Alpaca: {e}")

    price_df = pd.DataFrame(price_data).ffill().dropna()
    return price_df


def fetch_universe_history_yf(symbols: list[str], days: int) -> pd.DataFrame:
    """
    Fetch daily close prices via yfinance through today.
    """
    price_data = {}
    for s in symbols:
        tmp = yf.Ticker(s).history(period=f"{days}d", interval="1d", auto_adjust=False)
        if "Close" in tmp:
            price_data[s] = tmp["Close"]
    df = pd.DataFrame(price_data).ffill().dropna()
    return df


def main():
    today = date.today()
    if today > END_DATE:
        print(f"Trading window ended on {END_DATE}. Exiting.")
        return

    # 1) Connect to Alpaca client (unless DRY‑RUN)
    api = None
    if not DRY_RUN:
        api = REST(ALPACA_KEY, ALPACA_SECRET, BASE_URL)

    # 2) Fetch EOD history up through yesterday
    if DRY_RUN:
        price_df = fetch_universe_history_yf(symbols, days=DRY_DAYS)
    else:
        price_df = fetch_universe_history_alpaca(api, symbols, days=DRY_DAYS)

    # 3) Compute target weights
    weights = blend.generate_weights(price_df)

    # 4) Dollar‐value targets = weight × equity
    if DRY_RUN:
        equity = 100_000.0
    else:
        equity = float(api.get_account().equity)
    target_vals = weights * equity

    # 5) Get current positions (zero for DRY‑RUN)
    current_vals = {}
    for sym in symbols:
        if DRY_RUN:
            current_vals[sym] = 0.0
        else:
            try:
                pos = api.get_position(sym)
                current_vals[sym] = float(pos.qty) * float(
                    api.get_latest_trade(sym).price
                )
            except Exception:
                current_vals[sym] = 0.0

    # 6) Loop through each symbol, build & (optionally) send orders
    for sym in symbols:
        tgt = target_vals.get(sym, 0.0)
        cur = current_vals.get(sym, 0.0)
        diff = tgt - cur

        # skip tiny adjustments (<0.1% of equity)
        if abs(diff) < equity * 0.001:
            continue

        # use last available price
        price = (
            float(api.get_latest_trade(sym).price)
            if not DRY_RUN
            else price_df[sym].iloc[-1]
        )
        qty = int(diff / price)
        if qty == 0:
            continue

        side = "buy" if qty > 0 else "sell"
        order = dict(
            symbol=sym,
            qty=abs(qty),
            side=side,
            type="market",
            time_in_force="day",
        )

        timestamp = datetime.utcnow().isoformat()
        prefix = "DRY RUN →" if DRY_RUN else "ORDER →"
        print(
            f"[{timestamp}] {prefix} {sym}: {side} {abs(qty)} shares (Δ ${diff:,.2f})"
        )

        # Build rich Markdown reason
        reason_lines = []

        # 1. Weekly Reversal
        rev_w = rev.generate_weights(price_df).get(sym, 0)
        if rev_w > 0:
            ret1w = price_df[sym].pct_change(5).iloc[-1]
            reason_lines.append(
                f"Weekly Reversal: {sym} was among the {rev.bottom_k} worst 1‑week performers ({ret1w:.2%}), long."
            )
        elif rev_w < 0:
            ret1w = price_df[sym].pct_change(5).iloc[-1]
            reason_lines.append(
                f"Weekly Reversal: {sym} was among the {rev.top_k} best 1‑week performers ({ret1w:.2%}), short."
            )

        # 2. Pairs Trading
        pairs_w = pairs.generate_weights(price_df).get(sym, 0)
        if pairs_w != 0:
            a, b = pairs.pair
            spread = price_df[a] - price_df[b]
            z = (spread.iloc[-1] - spread.mean()) / spread.std()
            target = a if pairs_w > 0 else b
            side_txt = "long" if pairs_w > 0 else "short"
            reason_lines.append(
                f"Pairs Trading: {a}/{b} spread z‑score={z:.2f}, {side_txt} {target}."
            )

        # 3. Covered Calls
        cc_w = cc.generate_weights(price_df).get(sym, 0)
        if cc_w > 0:
            reason_lines.append(
                "Covered Calls: writing weekly ATM calls on SPY to harvest premium."
            )

        # 4. Low Volatility
        lv_w = lv.generate_weights(price_df).get(sym, 0)
        if lv_w > 0:
            vol60 = price_df[sym].pct_change().rolling(60).std().iloc[-1]
            reason_lines.append(
                f"Low Volatility: {sym} is in the lowest decile of 60‑day vol ({vol60:.2%})."
            )

        if not reason_lines:
            cur_w = (cur / equity) * 100 if equity else 0
            tgt_w = (tgt / equity) * 100 if equity else 0
            direction = "up" if diff > 0 else "down"
            reason_lines.append(
                (
                    "Rebalance: Moving {direction} toward target weight "
                    "{tgt_w:.1f}% from current {cur_w:.1f}% based on baseline allocation."
                ).format(direction=direction, tgt_w=tgt_w, cur_w=cur_w)
            )

        trade_record = {
            "timestamp": timestamp,
            "symbol": sym,
            "side": side,
            "qty": abs(qty),
            "price": price,
            "reason": reason_lines,
        }

        if DRY_RUN:
            print("  → DRY RUN TRADE RECORD:", trade_record)
        else:
            try:
                api.submit_order(**order)
                time.sleep(1)
                # refresh fill price
                fill_price = float(api.get_latest_trade(sym).price)
                trade_record["price"] = fill_price
                commit_trade(trade_record)
                print("  → Logged to GitHub:", trade_record)
            except Exception as e:
                print(f"  ❌ order/commit failed for {sym}: {e}")


if __name__ == "__main__":
    main()
