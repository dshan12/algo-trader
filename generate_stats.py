"""
generate_stats_from_alpaca.py

Lifetime (full portfolio_history) performance for your Alpaca account:
- Summary: total & annual returns, vol, Sharpe/Sortino, Max DD
- FIFO realized PnL by order_id (clean hit rate & profit factor)
- Benchmark vs SPY: alpha (annualized) / beta / R² and a normalized comparison plot

Outputs in ./report:
    equity_curve.png, drawdown.png, rolling_sharpe_30d.png, allocation_bar.png,
    monthly_returns_heatmap.csv, fills.csv, per_symbol_realized_pnl.csv,
    per_order_realized_pnl.csv, equity_vs_spy.png

Env:
- APCA_API_KEY_ID / APCA_API_SECRET_KEY (or ALPACA_API_KEY / ALPACA_API_SECRET)
- APCA_API_BASE_URL (optional, defaults to paper)
"""

import os, sys, math, time
from collections import defaultdict, deque
from typing import Tuple

# Optional .env
try:
    import dotenv

    dotenv.load_dotenv()
except Exception:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ---------------- Config ----------------
REPORT_DIR = "./report"
ROLLING_SHARPE_WINDOW = 30  # trading days
RISK_FREE_ANNUAL = 0.0
COST_BPS = 0.0  # extra per-side cost on sells for realized PnL
ACTIVITY_PAGES_MAX = 200  # pagination safety


# -------------- Helpers --------------
def _ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _annualized_return(monthly_returns: pd.Series) -> float:
    m = monthly_returns.dropna()
    if m.empty:
        return np.nan
    total = (1 + m).prod()
    years = len(m) / 12.0
    return total ** (1 / years) - 1 if years > 0 else np.nan


def _annualized_vol(daily_returns: pd.Series, periods=252) -> float:
    d = daily_returns.dropna()
    return d.std(ddof=0) * math.sqrt(periods) if not d.empty else np.nan


def _sharpe(daily_returns: pd.Series, rf_annual=0.0, periods=252) -> float:
    d = daily_returns.dropna()
    if d.empty:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / periods) - 1
    ex = d - rf_daily
    den = ex.std(ddof=0)
    return (
        ex.mean() / den * math.sqrt(periods)
        if den and not np.isclose(den, 0)
        else np.nan
    )


def _sortino(daily_returns: pd.Series, rf_annual=0.0, periods=252) -> float:
    d = daily_returns.dropna()
    if d.empty:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / periods) - 1
    ex = d - rf_daily
    downside = ex[ex < 0]
    den = downside.std(ddof=0)
    return (
        ex.mean() / den * math.sqrt(252) if den and not np.isclose(den, 0) else np.nan
    )


def _max_drawdown(equity: pd.Series) -> Tuple[float, pd.Series]:
    e = equity.dropna()
    if e.empty:
        return np.nan, pd.Series(index=equity.index, dtype=float)
    peak = e.cummax()
    dd = (e / peak) - 1.0
    return float(dd.min()), dd


def _profit_factor(pnl: pd.Series) -> float:
    w = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    if l <= 0:
        return math.inf if w > 0 else np.nan
    return float(w / l)


def _bps_cost(price: float, qty: float, bps: float) -> float:
    return abs(price * qty) * (bps / 1e4)


# -------------- Alpaca client --------------
def _get_alpaca():
    try:
        import alpaca_trade_api as tradeapi
    except Exception:
        print("Please `pip install alpaca-trade-api`.", file=sys.stderr)
        raise
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    key_id = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    if not key_id or not secret:
        raise RuntimeError(
            "Missing Alpaca creds. Set APCA_API_KEY_ID/APCA_API_SECRET_KEY or ALPACA_API_KEY/ALPACA_API_SECRET."
        )
    return tradeapi.REST(key_id, secret, base_url=base_url)


# -------------- Data pulls --------------
def fetch_account_positions(api):
    acct = api.get_account()
    positions = api.list_positions()
    account = {
        "equity": _to_float(getattr(acct, "equity", None)),
        "cash": _to_float(getattr(acct, "cash", None)),
        "portfolio_value": _to_float(getattr(acct, "portfolio_value", None)),
        "buying_power": _to_float(getattr(acct, "buying_power", None)),
        "last_equity": _to_float(getattr(acct, "last_equity", None)),
        "multiplier": _to_float(getattr(acct, "multiplier", 1)),
        "status": getattr(acct, "status", None),
    }
    rows = []
    for p in positions:
        rows.append(
            {
                "symbol": p.symbol,
                "qty": _to_float(p.qty),
                "avg_entry_price": _to_float(p.avg_entry_price),
                "market_price": _to_float(
                    getattr(p, "current_price", None)
                    or getattr(p, "asset_current_price", None)
                ),
                "market_value": _to_float(p.market_value),
                "cost_basis": _to_float(p.cost_basis),
                "unrealized_pl": _to_float(p.unrealized_pl),
                "unrealized_plpc": _to_float(p.unrealized_plpc),
                "asset_class": getattr(p, "asset_class", None),
            }
        )
    pos_df = (
        pd.DataFrame(rows).sort_values("market_value", ascending=False)
        if rows
        else pd.DataFrame()
    )
    if not pos_df.empty and "market_value" in pos_df.columns:
        total_mv = pos_df["market_value"].sum()
        pos_df["weight_pct"] = (
            (pos_df["market_value"] / total_mv * 100.0) if total_mv else 0.0
        )
    return account, pos_df


def fetch_portfolio_history(api):
    ph = api.get_portfolio_history(period="all", timeframe="1D", extended_hours=True)
    try:
        df = ph.df
    except Exception:
        df = pd.DataFrame(
            {
                "timestamp": getattr(ph, "timestamp", []),
                "equity": getattr(ph, "equity", []),
                "profit_loss": getattr(ph, "profit_loss", []),
            }
        )
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="s", utc=True, errors="coerce"
        )
        df = df.set_index("timestamp").sort_index()
    for col in ["equity", "profit_loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(how="all")


def fetch_activities_single(api, type_str: str):
    """Fetch ONE activity type with pagination (e.g., 'FILL')."""
    page_token, rows = None, []
    for _ in range(ACTIVITY_PAGES_MAX):
        acts = api.get_activities(activity_types=type_str, page_token=page_token)
        if not acts:
            break
        rows.extend(acts)
        if hasattr(acts, "next_page_token") and acts.next_page_token:
            page_token = acts.next_page_token
            time.sleep(0.15)
        else:
            break
    return rows


def fetch_fills_all(api):
    acts = fetch_activities_single(api, "FILL")
    rows = []
    for a in acts:
        rows.append(
            {
                "time": pd.to_datetime(
                    getattr(a, "transaction_time", None), utc=True, errors="coerce"
                ),
                "symbol": getattr(a, "symbol", None),
                "side": str(getattr(a, "side", "")).lower(),
                "qty": _to_float(getattr(a, "qty", None)),
                "price": _to_float(getattr(a, "price", None)),
                "order_id": getattr(a, "order_id", None),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("time").reset_index(drop=True)
    return df


# -------------- FIFO realized PnL (by order_id) --------------
def fifo_from_fills(fills: pd.DataFrame, cost_bps=0.0):
    if fills.empty:
        return fills, pd.Series(dtype=float), pd.Series(dtype=float)
    from collections import defaultdict, deque

    lots = defaultdict(deque)  # symbol -> deque of [qty, price]
    realized = []
    for r in fills.itertuples(index=False):
        sym, side, qty, px = r.symbol, r.side, float(r.qty or 0), float(r.price or 0)
        if side == "buy":
            lots[sym].append([qty, px])
            realized.append(0.0)
        else:
            rem, pnl = qty, 0.0
            while rem > 1e-12 and lots[sym]:
                lot_qty, lot_px = lots[sym][0]
                take = min(rem, lot_qty)
                pnl += (px - lot_px) * take
                lot_qty -= take
                rem -= take
                if lot_qty <= 1e-12:
                    lots[sym].popleft()
                else:
                    lots[sym][0][0] = lot_qty
            pnl -= _bps_cost(px, qty, cost_bps)  # cost on sell notional
            realized.append(pnl)
    out = fills.copy()
    out["realized_pnl"] = realized
    per_symbol = (
        out.groupby("symbol")["realized_pnl"].sum().sort_values(ascending=False)
    )
    per_order = (
        out.groupby("order_id")["realized_pnl"].sum().sort_values(ascending=False)
        if "order_id" in out.columns
        else out["realized_pnl"]
    )
    return out, per_symbol, per_order


# -------------- Benchmark (SPY) --------------
def fetch_spy_series(start_ts, end_ts):
    """Fetch SPY Close for [start_ts, end_ts] (UTC-aware) and return a Series named 'SPY'."""
    if start_ts is None or end_ts is None:
        return pd.Series(dtype=float)
    start = pd.Timestamp(start_ts).tz_convert("UTC").date().strftime("%Y-%m-%d")
    end = pd.Timestamp(end_ts).tz_convert("UTC").date().strftime("%Y-%m-%d")
    data = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if data is None or data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float)
    s = data["Close"].copy()
    s.index = pd.to_datetime(s.index, utc=True)
    s.name = "SPY"
    return s


def ols_alpha_beta(port_ret: pd.Series, mkt_ret: pd.Series):
    """OLS: Rp = alpha + beta*Rm + eps. Returns (alpha_daily, beta, r2)."""
    df = pd.concat([port_ret, mkt_ret], axis=1).dropna()
    if df.empty or df.shape[0] < 20:
        return np.nan, np.nan, np.nan
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    X = np.column_stack([np.ones_like(x), x])
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]  # [alpha, beta]
    y_hat = X @ beta_hat
    resid = y - y_hat
    ss_res = (resid**2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    alpha_daily, beta = float(beta_hat[0]), float(beta_hat[1])
    return alpha_daily, beta, float(r2)


# -------------- Figures --------------
def plot_equity_and_dd(equity: pd.Series):
    if equity.empty:
        return None, None
    mdd, dd = _max_drawdown(equity)
    plt.figure()
    plt.plot(equity.index, equity.values, label="Equity")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    eq_path = os.path.join(REPORT_DIR, "equity_curve.png")
    plt.savefig(eq_path)
    plt.close()

    plt.figure()
    plt.plot(dd.index, dd.values, label="Drawdown")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    dd_path = os.path.join(REPORT_DIR, "drawdown.png")
    plt.savefig(dd_path)
    plt.close()
    return eq_path, dd_path


def plot_rolling_sharpe(daily_returns: pd.Series, window=ROLLING_SHARPE_WINDOW):
    if daily_returns.empty:
        return None
    roll = daily_returns.rolling(window).mean() / daily_returns.rolling(window).std(
        ddof=0
    )
    if roll.dropna().empty:
        return None
    plt.figure()
    plt.plot(roll.index, roll.values, label=f"Rolling Sharpe ({window}d)")
    plt.title("Rolling Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "rolling_sharpe_30d.png")
    plt.savefig(path)
    plt.close()
    return path


def monthly_table(daily_returns: pd.Series) -> pd.DataFrame:
    if daily_returns.empty:
        return pd.DataFrame()
    m = daily_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    df = m.to_frame(name="ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    return df.pivot(index="year", columns="month", values="ret").sort_index()


def plot_allocation_bar(pos_df: pd.DataFrame):
    if pos_df is None or pos_df.empty or "market_value" not in pos_df.columns:
        return None
    top = pos_df.head(12).copy()
    plt.figure()
    plt.bar(top["symbol"], top["market_value"])
    plt.title("Top 12 Holdings by Market Value")
    plt.xlabel("Symbol")
    plt.ylabel("Market Value")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "allocation_bar.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_equity_vs_spy(equity: pd.Series, spy: pd.Series):
    if equity.empty or spy.empty:
        return None
    equity.name = "Equity"
    spy.name = "SPY"
    df = pd.concat([equity, spy], axis=1).dropna()
    if df.empty:
        return None
    eq_norm = df["Equity"] / df["Equity"].iloc[0]
    spy_norm = df["SPY"] / df["SPY"].iloc[0]
    plt.figure()
    plt.plot(eq_norm.index, eq_norm.values, label="Portfolio (normalized)")
    plt.plot(spy_norm.index, spy_norm.values, label="SPY (normalized)")
    plt.title("Portfolio vs SPY (Normalized)")
    plt.xlabel("Date")
    plt.ylabel("Index (Start=1.0)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "equity_vs_spy.png")
    plt.savefig(path)
    plt.close()
    return path


# -------------- Main --------------
def main():
    _ensure_report_dir()
    api = _get_alpaca()

    # Current account & positions (for allocation bar)
    account, pos_df = fetch_account_positions(api)
    print("=== Account (current) ===")
    for k, v in account.items():
        print(f"{k}: {v}")
    print("\n=== Open Positions (current) ===")
    if pos_df.empty:
        print("(none)")
    else:
        cols = [
            "symbol",
            "qty",
            "avg_entry_price",
            "market_price",
            "market_value",
            "unrealized_pl",
            "unrealized_plpc",
            "weight_pct",
        ]
        print(pos_df[cols].to_string(index=False))

    # Lifetime history
    ph = fetch_portfolio_history(api)
    equity = ph["equity"] if "equity" in ph.columns else pd.Series(dtype=float)
    if equity.empty:
        print("\nNo portfolio history returned; cannot compute lifetime stats.")
        return

    # daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan)
    daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    monthly = monthly_table(daily_returns)
    if not monthly.empty:
        monthly.to_csv(os.path.join(REPORT_DIR, "monthly_returns_heatmap.csv"))

    ann_ret = _annualized_return(monthly.stack()) if not monthly.empty else np.nan
    ann_vol = _annualized_vol(daily_returns)
    sharpe = _sharpe(daily_returns, rf_annual=RISK_FREE_ANNUAL)
    sortino = _sortino(daily_returns, rf_annual=RISK_FREE_ANNUAL)
    mdd, _dd = _max_drawdown(equity)

    # Fills -> realized PnL (by order_id)
    fills = fetch_fills_all(api)
    if not fills.empty:
        fills.to_csv(os.path.join(REPORT_DIR, "fills.csv"), index=False)
        _, per_symbol_pnl, per_order_pnl = fifo_from_fills(fills, cost_bps=COST_BPS)
        per_symbol_pnl.to_csv(os.path.join(REPORT_DIR, "per_symbol_realized_pnl.csv"))
        per_order_pnl.to_csv(os.path.join(REPORT_DIR, "per_order_realized_pnl.csv"))
        n_realized = int((per_order_pnl != 0).sum())
        wins = int((per_order_pnl > 0).sum())
        losses = int((per_order_pnl < 0).sum())
        hit_rate = (wins / n_realized) if n_realized else np.nan
        profit_factor = _profit_factor(per_order_pnl)
    else:
        n_realized = wins = losses = 0
        hit_rate = profit_factor = np.nan

    # Lifetime P&L from history (sum over full span)
    pl_from_history = (
        float(pd.to_numeric(ph["profit_loss"], errors="coerce").sum())
        if "profit_loss" in ph.columns
        else None
    )

    # ----- Lifetime summary -----
    print("\n=== Performance Summary (lifetime: first→last portfolio_history date) ===")
    print(f"Start Equity       : ${equity.iloc[0]:,.2f}")
    print(f"End Equity         : ${equity.iloc[-1]:,.2f}")
    print(f"Total Return       : {equity.iloc[-1] / equity.iloc[0] - 1: .2%}")
    print(f"Annual Return      : {ann_ret: .2%}")
    print(f"Annual Volatility  : {ann_vol: .2%}")
    print(f"Sharpe (rf={RISK_FREE_ANNUAL:.2%}) : {sharpe: .2f}")
    print(f"Sortino            : {sortino: .2f}")
    print(f"Max Drawdown       : {mdd: .2%}")
    print(f"Trades (realized)  : {n_realized}")
    print(f"Wins/Losses        : {wins}/{losses}")
    print(f"Hit Rate           : {hit_rate: .2%}")
    print(f"Profit Factor      : {profit_factor: .2f}")

    # ----- (Optional) simple reconciliation print (no transfers/dividends) -----

    # ----- SPY comparison + alpha/beta -----
    start_ts = equity.index[0]
    end_ts = equity.index[-1]
    spy_close = fetch_spy_series(start_ts, end_ts)

    if spy_close is None or spy_close.empty:
        print("\n=== Market Comparison (SPY) ===")
        print("SPY data unavailable; skipping alpha/beta and comparison plot.")
        vs_path = None
    else:
        # Build returns
        port_ret = equity.pct_change()
        spy_ret = spy_close.pct_change()

        # Force consistent column names (prevents KeyError)
        df_cmp = pd.concat([port_ret, spy_ret], axis=1)
        if df_cmp.shape[1] == 2:
            df_cmp.columns = ["Port_ret", "SPY_ret"]
        else:
            # Fallback: take first two columns with deterministic names
            df_cmp = df_cmp.iloc[:, :2]
            df_cmp.columns = ["Port_ret", "SPY_ret"]

        df_cmp = df_cmp.dropna()

        # Recompute series with clean names for downstream code
        port_ret = df_cmp["Port_ret"]
        spy_ret = df_cmp["SPY_ret"]

        # Alpha/Beta/R^2
        alpha_daily, beta, r2 = ols_alpha_beta(port_ret, spy_ret)
        if not np.isnan(alpha_daily):
            alpha_annual = (1 + alpha_daily) ** 252 - 1
            print("\n=== Market Comparison (SPY) ===")
            print(f"Beta (daily OLS)   : {beta: .3f}")
            print(f"Alpha (annualized) : {alpha_annual: .2%}")
            print(f"R²                 : {r2: .3f}")

            # Excess Sharpe vs SPY
            if port_ret.std(ddof=0) and spy_ret.std(ddof=0):
                excess = port_ret - spy_ret
                excess_sharpe = (
                    (excess.mean() / excess.std(ddof=0)) * math.sqrt(252)
                    if excess.std(ddof=0)
                    else np.nan
                )
                print(f"Excess Sharpe vs SPY: {excess_sharpe: .2f}")
        else:
            print("\n=== Market Comparison (SPY) ===")
            print("Insufficient overlapping data to compute alpha/beta.")

        # Plot normalized equity vs SPY
        vs_path = plot_equity_vs_spy(equity, spy_close)
    # ----- Plots -----
    eq_path, dd_path = plot_equity_and_dd(equity)
    rs_path = plot_rolling_sharpe(daily_returns, window=ROLLING_SHARPE_WINDOW)
    alloc_path = plot_allocation_bar(pos_df)

    print("\nSaved to ./report:")
    for p in [
        eq_path,
        dd_path,
        rs_path,
        alloc_path,
        os.path.join(REPORT_DIR, "monthly_returns_heatmap.csv"),
        os.path.join(REPORT_DIR, "per_symbol_realized_pnl.csv"),
        os.path.join(REPORT_DIR, "per_order_realized_pnl.csv"),
        os.path.join(REPORT_DIR, "fills.csv"),
        vs_path if "vs_path" in locals() else None,
    ]:
        if p and os.path.exists(p):
            print(" -", p)


if __name__ == "__main__":
    main()
