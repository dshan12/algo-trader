#!/usr/bin/env python3
"""
Build a representative SPY subset (Top-N by weight) and write symbols.txt.

Usage:
  python tools/update_symbols_spy_subset.py            # default N=50
  python tools/update_symbols_spy_subset.py 100        # Top 100
"""

import sys
from pathlib import Path
import pandas as pd

# Optional internet deps
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SYMS_TXT = ROOT / "symbols.txt"


def _clean_symbol(sym: str) -> str:
    if not isinstance(sym, str) or not sym.strip():
        return None
    s = sym.strip().upper()
    # many feeds include cash or placeholders; filter those out
    bad = {"CASH", "USD", "ZZZ", "—", "-", "N/A"}
    if s in bad:
        return None
    # yfinance uses dots for classes (BRK.B), Alpaca uses slashes (BRK/B).
    # If you need Alpaca format later, flip the next two lines:
    # s = s.replace("/", ".")  # Alpaca->Yahoo
    # s = s.replace(".", "/")  # Yahoo->Alpaca
    return s


def _fetch_spy_holdings_yf():
    """
    Try to get holdings with weights from yfinance.
    Returns DataFrame with columns: ['symbol','weight'] if possible.
    """
    t = yf.Ticker("SPY")
    # Try fund_holdings (full table, may include weight or %)
    try:
        df = t.fund_holdings
        if isinstance(df, pd.DataFrame) and not df.empty:
            # normalize columns
            cols = {c.lower().strip(): c for c in df.columns}
            sym_col = next(
                (c for c in df.columns if c.lower().startswith("symbol")), None
            )
            # weight could be 'weight', 'holding %', or similar
            w_col = None
            for cand in df.columns:
                cl = cand.lower()
                if "weight" in cl or "%" in cl:
                    w_col = cand
                    break
            if sym_col is not None:
                out = pd.DataFrame({"symbol": df[sym_col].map(_clean_symbol)})
                if w_col is not None:
                    # normalize weight to float 0..1
                    w = df[w_col]
                    if w.dtype == object:
                        w = (
                            w.astype(str)
                            .str.replace("%", "", regex=False)
                            .str.replace(",", "", regex=False)
                        )
                    w = pd.to_numeric(w, errors="coerce")
                    if (w > 1.5).any():  # likely in percent
                        w = w / 100.0
                    out["weight"] = w
                else:
                    out["weight"] = pd.NA
                out = out.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
                return out
    except Exception:
        pass

    # Try fund_top_holdings (fewer rows)
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
            if sym_col is not None:
                out = pd.DataFrame({"symbol": df[sym_col].map(_clean_symbol)})
                if w_col is not None:
                    w = df[w_col]
                    if w.dtype == object:
                        w = (
                            w.astype(str)
                            .str.replace("%", "", regex=False)
                            .str.replace(",", "", regex=False)
                        )
                    w = pd.to_numeric(w, errors="coerce")
                    if (w > 1.5).any():
                        w = w / 100.0
                    out["weight"] = w
                else:
                    out["weight"] = pd.NA
                out = out.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
                return out
    except Exception:
        pass

    return None


def _fetch_sp500_from_wikipedia():
    """
    Fallback: returns DataFrame with 'symbol' (no weights).
    """
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        if not tables:
            return None
        # first table is usually the constituents
        df = tables[0]
        sym_col = next(
            (c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()),
            None,
        )
        if sym_col is None:
            return None
        out = pd.DataFrame({"symbol": df[sym_col].map(_clean_symbol)})
        out = out.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
        out["weight"] = pd.NA
        return out
    except Exception:
        return None


def build_spy_subset(top_n: int = 50) -> pd.Series:
    """
    Returns a Series of tickers (length <= top_n), ideally sorted by weight desc.
    """
    df = _fetch_spy_holdings_yf()
    if df is None or df.empty:
        df = _fetch_sp500_from_wikipedia()
    if df is None or df.empty:
        raise RuntimeError(
            "Could not fetch SPY/S&P 500 constituents from available sources."
        )

    if "weight" in df.columns and df["weight"].notna().any():
        df = df.sort_values("weight", ascending=False)
    else:
        # no weights: keep alphabetical for determinism
        df = df.sort_values("symbol")

    subset = df["symbol"].head(top_n).reset_index(drop=True)
    return subset


def write_symbols_txt(symbols: pd.Series, path=SYMS_TXT):
    path.write_text("\n".join(symbols.tolist()) + "\n", encoding="utf-8")


def main():
    top_n = 50
    if len(sys.argv) >= 2:
        try:
            top_n = int(sys.argv[1])
        except Exception:
            pass
    syms = build_spy_subset(top_n=top_n)
    write_symbols_txt(syms)
    print(f"Updated {SYMS_TXT} with Top-{len(syms)} SPY subset.")


if __name__ == "__main__":
    main()
