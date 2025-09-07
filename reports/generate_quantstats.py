#!/usr/bin/env python
# reports/generate_quantstats.py

from __future__ import annotations

import argparse
from pathlib import Path

# --- Friendly dependency check ---------------------------------------------
missing = []
for mod in ("pandas", "yfinance", "quantstats", "IPython"):
    try:
        __import__(mod)
    except ModuleNotFoundError:
        missing.append(mod)
if missing:
    raise SystemExit(
        "Missing packages: "
        + ", ".join(missing)
        + "\nInstall with: uv add "
        + " ".join(missing)
    )

import pandas as pd
import numpy as np
import yfinance as yf
import quantstats.reports as qsr  # use submodule to avoid notebook display

# ============== Quiet matplotlib + patch for pandas-resampler =================
import matplotlib
import logging

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Patch pandas Resampler.sum to ignore kwargs (axis/min_count) that QS passes
try:
    from pandas.core.resample import Resampler

    _orig_resampler_sum = Resampler.sum

    def _resampler_sum_noargs(self, *args, **kwargs):
        # Ignore deprecated kwargs passed by QS; call the real implementation
        return _orig_resampler_sum(self)

    Resampler.sum = _resampler_sum_noargs  # type: ignore[attr-defined]
except Exception as _e:
    print(f"[warn] could not patch pandas Resampler.sum: {_e}")

# And patch QS plotter in case a Resampler object sneaks through
try:
    import quantstats._plotting.core as _qs_core

    _orig_plot_timeseries = _qs_core.plot_timeseries

    def _patched_plot_timeseries(returns, *args, **kwargs):
        # If it's a pandas Resampler, reduce it to concrete data first
        if "Resampler" in type(returns).__name__:
            try:
                returns = returns.sum()  # reduce
            except Exception:
                returns = returns.last()
        return _orig_plot_timeseries(returns, *args, **kwargs)

    _qs_core.plot_timeseries = _patched_plot_timeseries
except Exception as _e:
    print(f"[warn] could not patch quantstats plotter: {_e}")

# --- Paths / constants ------------------------------------------------------
TZ = "America/Chicago"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRADES = ROOT / "trades.json"
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
RAW_DIR = DATA_DIR / "raw"


# =========================
# Helpers: tickers & fetch
# =========================


def normalize_ticker(t: str) -> str:
    """Rough broker->Yahoo normalization: BRK.B -> BRK-B, RDS/A -> RDS-A."""
    t = str(t).strip().upper()
    return t.replace("/", "-").replace(".", "-")


def fetch_symbol(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    t = yf.Ticker(symbol)
    df = t.history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {symbol} ({period=}, {interval=})"
        )

    df.index.name = "Date"
    df.sort_index(inplace=True)

    # save raw + clean
    (RAW_DIR / f"{symbol}.csv").write_text(df.to_csv())
    clean = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    (CLEAN_DIR / f"{symbol}.csv").write_text(clean.to_csv())
    return clean


def fetch_many(
    symbols: list[str], period="5y", interval="1d"
) -> dict[str, pd.DataFrame]:
    out = {}
    for s in sorted(set(symbols)):
        try:
            print(f"[fetch] {s} period='{period}' interval='{interval}'")
            out[s] = fetch_symbol(s, period=period, interval=interval)
        except Exception as e:
            print(f"[warn] fetch {s}: {e}")
    return out


# =========================
# Equity reconstruction
# =========================


def _norm_time(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert(TZ)


def load_trades(trades_path: Path) -> pd.DataFrame:
    if not trades_path.exists():
        raise FileNotFoundError(f"trades.json not found at {trades_path}")

    df = pd.read_json(trades_path, convert_dates=["timestamp"])
    if df.empty:
        raise ValueError("trades.json is empty")

    # sanitize
    df["timestamp"] = _norm_time(df["timestamp"])
    for col in ("symbol", "side"):
        if col in df:
            df[col] = df[col].astype(str)
    df["symbol"] = df["symbol"].map(normalize_ticker)
    df["side"] = df["side"].str.lower()

    df["qty"] = pd.to_numeric(df.get("qty", 0), errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0.0)

    # position sign: buy/cover +, sell/short -
    buy_like = df["side"].isin(["buy", "cover"])
    sell_like = df["side"].isin(["sell", "short"])
    df["signed_qty"] = np.where(
        buy_like, df["qty"], np.where(sell_like, -df["qty"], 0.0)
    )

    # cash changes: buying spends cash; selling/short receives cash
    df["cash_flow"] = -df["price"] * df["signed_qty"]

    # optional fees
    if "fees" in df.columns:
        df["fees"] = pd.to_numeric(df["fees"], errors="coerce").fillna(0.0)
        df["cash_flow"] -= df["fees"]

    return df.sort_values("timestamp")


def load_symbol_prices_local(symbol: str) -> pd.DataFrame:
    p = CLEAN_DIR / f"{symbol}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing local price file: {p} (run without --no-fetch to download)"
        )
    df = pd.read_csv(p)

    # yfinance CSV date column can be 'Date' or sometimes an unnamed index ('Unnamed: 0')
    tcol = next(
        (c for c in ("Date", "date", "datetime", "Unnamed: 0") if c in df.columns), None
    )
    if tcol is None:
        raise ValueError(f"{p} needs a Date/datetime column")

    ts = pd.to_datetime(df[tcol], utc=True).dt.tz_convert(TZ)

    close_col = (
        "Close"
        if "Close" in df.columns
        else ("close" if "close" in df.columns else None)
    )
    if close_col is None:
        raise ValueError(f"{p} needs Close/close column")

    out = (
        pd.DataFrame(
            {"timestamp": ts, "close": pd.to_numeric(df[close_col], errors="coerce")}
        )
        .dropna()
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    return out


def build_price_panel(symbols: list[str], idx: pd.DatetimeIndex) -> pd.DataFrame:
    frames, skipped = [], []
    for s in symbols:
        try:
            px = load_symbol_prices_local(s).reindex(idx, method="ffill")
            px.columns = [s]
            frames.append(px)
        except FileNotFoundError:
            skipped.append(s)
        except Exception as e:
            skipped.append(s)
            print(f"[warn] skipping {s}: {e}")
    if skipped:
        print(f"[warn] no local data for: {', '.join(skipped)}")
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=idx)


def equity_breakdown_daily(
    trades_path: Path,
    start_cash: float,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Returns daily DataFrame with ['cash','stocks','equity'].
    CASH changes via trade cash_flow; STOCKS is mark-to-market; EQUITY = cash + stocks.
    Index is naive datetime (no tz) for QuantStats.
    """
    t = load_trades(trades_path)

    if start is None:
        start = t["timestamp"].min().floor("D")
    if end is None:
        end = pd.Timestamp.now(tz=TZ).floor("D")

    # Daily index with 5-day warmup to allow ffill before start
    idx = pd.date_range(
        (start - pd.Timedelta(days=5)).floor("D"), end.ceil("D"), freq="1D", tz=TZ
    )

    symbols = sorted(t["symbol"].unique().tolist())
    panel = build_price_panel(symbols, idx)

    # positions timeline (cumulative)
    if symbols:
        pos = (
            t[["timestamp", "symbol", "signed_qty"]]
            .set_index("timestamp")
            .groupby("symbol")["signed_qty"]
            .apply(lambda s: s.reindex(idx, method="ffill").fillna(0.0).cumsum())
            .unstack(0)
            .reindex(idx)
            .fillna(0.0)
        )
    else:
        pos = pd.DataFrame(index=idx)

    # multiply only across symbols we actually have prices for
    if panel.empty or pos.empty:
        stocks_value = pd.Series(0.0, index=idx)
    else:
        pos2 = pos.reindex(columns=panel.columns).fillna(0.0)
        stocks_value = (
            (pos2 * panel).sum(axis=1).ffill().fillna(0.0)
        )  # deprecation-safe

    # cash from cash_flow
    cash_cf = (
        t[["timestamp", "cash_flow"]]
        .set_index("timestamp")
        .reindex(idx, fill_value=0.0)["cash_flow"]
        .cumsum()
    )
    cash = (start_cash + cash_cf).astype(float)

    out = pd.DataFrame({"cash": cash, "stocks": stocks_value}, index=idx)
    out["equity"] = out["cash"] + out["stocks"]
    out = out.loc[(out.index >= start) & (out.index <= end)]
    out.index = out.index.tz_convert("UTC").tz_localize(
        None
    )  # QuantStats expects naive dates
    return out


# =========================
# CLI
# =========================


def main():
    ap = argparse.ArgumentParser(
        description="Generate a QuantStats HTML report from trades.json. "
        "Automatically fetches prices into data/clean/ (unless --no-fetch)."
    )
    ap.add_argument(
        "--trades", type=Path, default=DEFAULT_TRADES, help="Path to trades.json"
    )
    ap.add_argument(
        "--benchmark", type=str, default="SPY", help="Benchmark ticker (also fetched)"
    )
    ap.add_argument(
        "--period", type=str, default="5y", help="Fetch period (e.g., 2y, 5y)"
    )
    ap.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Fetch interval (1d recommended for QS)",
    )
    ap.add_argument(
        "--start-cash",
        type=float,
        default=100000.0,
        help="Starting cash at window start",
    )
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "quantstats_report.html",
        help="Output HTML file",
    )
    ap.add_argument(
        "--title", type=str, default=None, help="Custom title for the report"
    )
    ap.add_argument(
        "--dump-csv",
        action="store_true",
        help="Also write reports/portfolio_daily.csv (cash/stocks/equity)",
    )
    ap.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip fetching; use existing CSVs in data/clean",
    )
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Parse dates
    start = pd.to_datetime(args.start) if args.start else None
    end = pd.to_datetime(args.end) if args.end else None
    if start is not None:
        start = start.tz_localize(TZ)
    if end is not None:
        end = end.tz_localize(TZ)

    # 1) Load trades & collect symbols
    tdf = pd.read_json(args.trades, convert_dates=["timestamp"])
    if "symbol" not in tdf.columns or tdf.empty:
        raise SystemExit("trades.json is empty or missing the 'symbol' column.")
    symbols = sorted(set(normalize_ticker(s) for s in tdf["symbol"].astype(str)))
    if args.benchmark:
        symbols.append(normalize_ticker(args.benchmark))

    # 2) Fetch prices (unless --no-fetch)
    if not args.no_fetch:
        print(f"[info] fetching {len(symbols)} symbols to data/clean: {', '.join(symbols)}")
        fetch_many(symbols, period=args.period, interval=args.interval)
    else:
        print("[info] skipping fetch (--no-fetch). Using existing data/clean/*.csv")

    # 3) Build daily equity (cash moves here)
    daily = equity_breakdown_daily(args.trades, args.start_cash, start=start, end=end)

    # Optional: dump CSV for inspection
    if args.dump_csv:
        csv_path = args.output.parent / "portfolio_daily.csv"
        daily.to_csv(csv_path, index_label="date")
        print(f"[write] daily breakdown -> {csv_path}")

    # 4) Generate QuantStats report
    returns = daily["equity"].pct_change(fill_method=None).dropna()  # be explicit re: FutureWarning
    title = args.title or (f"Strategy vs {args.benchmark}" if args.benchmark else "Strategy Report")

    try:
        qsr.html(
            returns=returns,
            benchmark=args.benchmark or None,  # QuantStats fetches benchmark itself
            output=str(args.output),
            title=title,
        )
    except Exception as e:
        print(f"[warn] QuantStats with benchmark failed ({e}). Generating without benchmark.")
        qsr.html(
            returns=returns,
            benchmark=None,
            output=str(args.output),
            title=title,
        )

    print(f"[done] report -> {args.output}")


if __name__ == "__main__":
    main()

