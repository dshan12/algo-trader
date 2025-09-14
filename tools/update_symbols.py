#!/usr/bin/env python3
"""
Update symbols.txt with a SPY-like subset (Top-N by weight when available).

Priority of data sources (first one that works wins):
  1) SPDR (State Street) SPY daily holdings CSV
  2) iShares IVV holdings CSV (proxy for S&P 500)
  3) yfinance fund_holdings / fund_top_holdings
  4) Wikipedia S&P 500 table (no weights)
  5) Static Top-100 fallback (baked in)

Usage:
  python tools/update_symbols.py                # default N=50, yahoo format
  python tools/update_symbols.py --n 100
  python tools/update_symbols.py --format alpaca  # convert BRK.B -> BRK/B
  python tools/update_symbols.py --source spdr    # force a specific source

Notes:
- Requires internet for sources 1-4. Falls back to static list if all fail.
- Install deps: pip install requests yfinance pandas lxml html5lib
"""

from __future__ import annotations
import sys, time, argparse
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

# optional deps
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    import requests
except Exception:
    requests = None

ROOT = Path(__file__).resolve().parents[1]
SYMS_TXT = ROOT / "symbols.txt"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)

# ----------- last-resort static Top-100 (approx. SPY-weighted 2024/25) ----------
STATIC_TOP100 = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "GOOG",
    "META",
    "BRK.B",
    "LLY",
    "TSLA",
    "AVGO",
    "JPM",
    "V",
    "XOM",
    "UNH",
    "JNJ",
    "PG",
    "HD",
    "MA",
    "CVX",
    "MRK",
    "COST",
    "ABBV",
    "PEP",
    "KO",
    "BAC",
    "ADBE",
    "WMT",
    "CSCO",
    "ORCL",
    "MCD",
    "ACN",
    "WFC",
    "TMO",
    "DIS",
    "ABT",
    "LIN",
    "CRM",
    "TXN",
    "NFLX",
    "DHR",
    "AMD",
    "INTU",
    "PM",
    "AMGN",
    "VZ",
    "COP",
    "IBM",
    "RTX",
    "INTC",
    "LOW",
    "NEE",
    "UPS",
    "QCOM",
    "HON",
    "MS",
    "CAT",
    "ELV",
    "GE",
    "BKNG",
    "UBER",
    "SBUX",
    "GS",
    "SPGI",
    "PH",
    "BLK",
    "LMT",
    "PLD",
    "SYK",
    "ADP",
    "CI",
    "AXP",
    "CVS",
    "MDT",
    "MO",
    "NOW",
    "CME",
    "DE",
    "PGR",
    "TJX",
    "T",
    "AMT",
    "TMUS",
    "MMC",
    "REGN",
    "BDX",
    "SCHW",
    "ZTS",
    "SO",
    "ISRG",
    "EQIX",
    "PNC",
    "MU",
    "CB",
    "ETN",
    "VRTX",
    "CRWD",
    "EOG",
    "ICE",
    "GILD",
    "CSX",
    "KLAC",
    "PFE",
]


# ---------------- Helpers ----------------
def _clean_symbol(sym: str) -> Optional[str]:
    if not isinstance(sym, str):
        return None
    s = sym.strip().upper()
    if not s:
        return None
    bad = {"CASH", "USD", "—", "-", "N/A", "ZZZ"}
    if s in bad:
        return None
    return s


def _apply_format(symbols: List[str], fmt: str) -> List[str]:
    if fmt == "alpaca":
        # Yahoo -> Alpaca class tickers: BRK.B -> BRK/B, BF.B -> BF/B, etc.
        return [s.replace(".", "/") for s in symbols]
    # default "yahoo"
    return symbols


def _http_get(url: str, timeout=20, retries=2) -> Optional[bytes]:
    if requests is None:
        return None
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception as e:
            last_exc = e
        time.sleep(0.6 * (attempt + 1))
    return None


def _to_weights(
    df: pd.DataFrame, symbol_col: str, weight_col: Optional[str]
) -> pd.DataFrame:
    out = pd.DataFrame({"symbol": df[symbol_col].map(_clean_symbol)})
    if weight_col and weight_col in df.columns:
        w = df[weight_col]
        if w.dtype == object:
            w = (
                w.astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
            )
        w = pd.to_numeric(w, errors="coerce")
        # If percentages like 6.54, convert to 0.0654
        if pd.notna(w).any() and (w > 1.5).any():
            w = w / 100.0
        out["weight"] = w
    else:
        out["weight"] = pd.NA
    out = out.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
    return out


# ---------------- Sources ----------------
def source_spdr() -> Optional[pd.DataFrame]:
    """
    SPDR (State Street) official SPY holdings CSV.
    Known endpoint pattern: fundNumber=33 for SPY.
    """
    url = "https://www.ssga.com/bin/etf/holdings/download?fileType=csv&fundNumber=33"
    content = _http_get(url)
    if not content:
        return None
    try:
        df = pd.read_csv(pd.compat.StringIO(content.decode("utf-8")), skiprows=0)
    except Exception:
        # Some locales deliver ; separated files
        try:
            df = pd.read_csv(pd.compat.StringIO(content.decode("utf-8")), sep=";")
        except Exception:
            return None

    # Try common columns
    candidates = [
        ("Ticker", "Weight"),
        ("Ticker", "Weight (%)"),
        ("Ticker", "% Weight"),
        ("Symbol", "Weight"),
        ("Symbol", "Weight (%)"),
    ]
    sym_col, w_col = None, None
    for s, w in candidates:
        if s in df.columns and w in df.columns:
            sym_col, w_col = s, w
            break
    # If still None, try to infer symbol col
    if sym_col is None:
        for c in df.columns:
            if "ticker" in c.lower() or "symbol" in c.lower():
                sym_col = c
                break
    if sym_col is None:
        return None
    # weight may be missing in some downloads -> handled in _to_weights
    return _to_weights(df, sym_col, w_col)


def source_ishares_ivv() -> Optional[pd.DataFrame]:
    """
    iShares IVV holdings CSV (proxy for S&P 500).
    """
    url = (
        "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/"
        "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
    )
    content = _http_get(url)
    if not content:
        return None
    try:
        df = pd.read_csv(pd.compat.StringIO(content.decode("utf-8")))
    except Exception:
        try:
            df = pd.read_csv(pd.compat.StringIO(content.decode("latin1")))
        except Exception:
            return None

    # Common columns on iShares CSVs
    # e.g., "Ticker", "Weight (%)"
    sym_col, w_col = None, None
    for c in df.columns:
        cl = c.lower()
        if sym_col is None and ("ticker" in cl or "symbol" in cl):
            sym_col = c
        if w_col is None and ("weight" in cl and "%" in cl):
            w_col = c
    if sym_col is None:
        return None
    return _to_weights(df, sym_col, w_col)


def source_yfinance() -> Optional[pd.DataFrame]:
    if yf is None:
        return None
    try:
        t = yf.Ticker("SPY")
    except Exception:
        return None

    # Try fund_holdings first
    try:
        df = t.fund_holdings
        if isinstance(df, pd.DataFrame) and not df.empty:
            sym_col = next(
                (c for c in df.columns if c.lower().startswith("symbol")), None
            )
            w_col = next(
                (c for c in df.columns if "weight" in c.lower() or "%" in c.lower()),
                None,
            )
            if sym_col:
                return _to_weights(df, sym_col, w_col)
    except Exception:
        pass

    # Then fund_top_holdings
    try:
        df = t.fund_top_holdings
        if isinstance(df, pd.DataFrame) and not df.empty:
            sym_col = next(
                (c for c in df.columns if c.lower().startswith("symbol")), None
            )
            w_col = next(
                (c for c in df.columns if "weight" in c.lower() or "%" in c.lower()),
                None,
            )
            if sym_col:
                return _to_weights(df, sym_col, w_col)
    except Exception:
        pass

    return None


def source_wikipedia() -> Optional[pd.DataFrame]:
    # Wikipedia S&P 500 constituents (no weights)
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
    except Exception:
        return None
    if not tables:
        return None
    df = tables[0]
    sym_col = next(
        (c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), None
    )
    if sym_col is None:
        return None
    out = pd.DataFrame({"symbol": df[sym_col].map(_clean_symbol)})
    out = out.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
    out["weight"] = pd.NA
    return out


# ---------------- Builder ----------------
def build_spy_subset(top_n: int = 50, prefer: Optional[str] = None) -> pd.Series:
    """
    Returns a Series of tickers (length <= top_n), ideally sorted by weight desc.
    prefer: one of {spdr, ishares, yfinance, wikipedia} to force a source.
    """
    sources = [
        ("spdr", source_spdr),
        ("ishares", source_ishares_ivv),
        ("yfinance", source_yfinance),
        ("wikipedia", source_wikipedia),
    ]

    if prefer:
        # move preferred source to front
        names = [name for name, _ in sources]
        if prefer not in names:
            raise ValueError(f"--source must be one of {names}")
        sources = sorted(sources, key=lambda x: 0 if x[0] == prefer else 1)

    df = None
    for name, fn in sources:
        try:
            df = fn()
            if df is not None and not df.empty:
                print(f"[info] using source: {name} ({len(df)} rows)")
                break
        except Exception as e:
            print(f"[warn] source {name} failed: {e}")

    if df is None or df.empty:
        # final fallback to static list
        print("[warn] all online sources failed; using static Top-100 fallback.")
        df = pd.DataFrame({"symbol": STATIC_TOP100, "weight": pd.NA})

    if "weight" in df.columns and df["weight"].notna().any():
        df = df.sort_values("weight", ascending=False)
    else:
        # no weights: keep alphabetical for determinism
        df = df.sort_values("symbol")

    subset = df["symbol"].dropna().astype(str).head(top_n).reset_index(drop=True)
    return subset


def write_symbols_txt(symbols: pd.Series, path=SYMS_TXT, fmt="yahoo"):
    syms = _apply_format(symbols.tolist(), fmt=fmt)
    Path(path).write_text("\n".join(syms) + "\n", encoding="utf-8")


# ---------------- CLI ----------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update symbols.txt with a SPY-like Top-N subset."
    )
    p.add_argument(
        "--n", type=int, default=50, help="Top N symbols to keep (default 50)"
    )
    p.add_argument(
        "--format",
        choices=["yahoo", "alpaca"],
        default="yahoo",
        help="Ticker format (default yahoo)",
    )
    p.add_argument(
        "--source",
        choices=["spdr", "ishares", "yfinance", "wikipedia"],
        default=None,
        help="Force a specific source",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(SYMS_TXT),
        help="Output file path (default repo root symbols.txt)",
    )
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    try:
        syms = build_spy_subset(top_n=args.n, prefer=args.source)
    except Exception as e:
        print(f"[error] build_spy_subset failed: {e}")
        print("[warn] falling back to static Top-100 list.")
        syms = pd.Series(STATIC_TOP100).head(args.n)
    write_symbols_txt(syms, path=args.out, fmt=args.format)
    print(f"Updated {args.out} with Top-{len(syms)} symbols (format={args.format}).")


if __name__ == "__main__":
    main()
