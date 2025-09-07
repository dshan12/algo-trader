from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

TZ = "America/Chicago"


def _norm_time(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert(TZ)


def load_trades(trades_path: Path) -> pd.DataFrame:
    df = pd.read_json(trades_path, convert_dates=["timestamp"])
    df["timestamp"] = _norm_time(df["timestamp"])
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.lower()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    # POS sign: buy/cover = +qty ; sell/short = -qty
    buy_like = df["side"].isin(["buy", "cover"])
    sell_like = df["side"].isin(["sell", "short"])
    df["signed_qty"] = np.where(
        buy_like, df["qty"], np.where(sell_like, -df["qty"], 0.0)
    )

    # CASH sign: buying spends cash; selling/shorting receives cash
    # cash_flow is the *change* in cash; cum-summed then added to start_cash.
    df["cash_flow"] = -df["price"] * df["signed_qty"]

    # Optional fees/commissions (if your trades have a 'fees' column)
    if "fees" in df.columns:
        df["cash_flow"] -= pd.to_numeric(df["fees"], errors="coerce").fillna(0.0)

    # Optional deposits/withdrawals ledger (if you keep 'cash_event' + 'amount')
    if "cash_event" in df.columns and "amount" in df.columns:
        # cash_event could be "deposit" or "withdrawal" etc.
        amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["cash_flow"] += np.where(
            df["cash_event"].str.lower().eq("deposit"),
            amt,
            np.where(df["cash_event"].str.lower().eq("withdrawal"), -amt, 0.0),
        )

    return df.sort_values("timestamp")


def load_symbol_prices(symbol: str, data_dir: Path) -> pd.DataFrame:
    p = data_dir / "clean" / f"{symbol}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing price file: {p}")
    df = pd.read_csv(p)

    # yfinance CSVs often have the date in 'Date' or unnamed index column
    for col in ("Date", "date", "datetime", "Unnamed: 0"):
        if col in df.columns:
            tcol = col
            break
    else:
        raise ValueError(f"{p} needs Date/datetime column.")
    ts = pd.to_datetime(df[tcol], utc=True).dt.tz_convert(TZ)

    close_col = (
        "Close"
        if "Close" in df.columns
        else ("close" if "close" in df.columns else None)
    )
    if close_col is None:
        raise ValueError(f"{p} needs Close/close column.")

    out = (
        pd.DataFrame(
            {"timestamp": ts, "close": pd.to_numeric(df[close_col], errors="coerce")}
        )
        .dropna()
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    return out


def build_price_panel(
    symbols: list[str], idx: pd.DatetimeIndex, data_dir: Path
) -> pd.DataFrame:
    frames = []
    for s in symbols:
        px = load_symbol_prices(s, data_dir).reindex(idx, method="ffill")
        px.columns = [s]
        frames.append(px)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=idx)


def equity_breakdown_daily(
    trades_path: Path,
    data_dir: Path,
    start_cash: float,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Returns a daily DataFrame with columns: ['cash','stocks','equity'].
    CASH CHANGES here via cash_flow; STOCKS is mark-to-market; EQUITY = cash + stocks.
    """
    t = load_trades(trades_path)
    if t.empty:
        raise ValueError("No trades found")

    if start is None:
        start = t["timestamp"].min().floor("D")
    if end is None:
        end = pd.Timestamp.now(tz=TZ).floor("D")

    # Build daily index with a small warmup to allow ffill before 'start'
    idx = pd.date_range(
        (start - pd.Timedelta(days=5)).floor("D"), end.ceil("D"), freq="1D", tz=TZ
    )

    symbols = sorted(t["symbol"].unique().tolist())
    panel = build_price_panel(symbols, idx, data_dir)  # cols=symbols

    # Positions over time (cumulative signed quantities)
    pos = (
        t[["timestamp", "symbol", "signed_qty"]]
        .set_index("timestamp")
        .groupby("symbol")["signed_qty"]
        .apply(lambda s: s.reindex(idx, method="ffill").fillna(0.0).cumsum())
        .unstack(0)
        .reindex(idx)
        .fillna(0.0)
    )

    # Stocks value (mark-to-market)
    stocks_value = (pos * panel).sum(axis=1).fillna(method="ffill").fillna(0.0)

    # Cash timeline from cash flows
    cash_cf = (
        t[["timestamp", "cash_flow"]]
        .set_index("timestamp")
        .reindex(idx, fill_value=0.0)["cash_flow"]
        .cumsum()
    )
    cash = (start_cash + cash_cf).astype(float)

    # Pack + clip to requested window, then drop tz for QuantStats
    out = pd.DataFrame({"cash": cash, "stocks": stocks_value}, index=idx)
    out["equity"] = out["cash"] + out["stocks"]
    out = out.loc[(out.index >= start) & (out.index <= end)]
    out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out
