import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"
CLEAN_DIR = Path(__file__).parents[1] / "data" / "clean"


def fetch_symbol(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=False)

    df.index.name = "Date"
    df.sort_index(inplace=True)

    raw_path = RAW_DIR / f"{symbol}.csv"
    df.to_csv(raw_path)

    clean_df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    clean_path = CLEAN_DIR / f"{symbol}.csv"
    clean_df.to_csv(clean_path)

    return clean_df


if __name__ == "__main__":
    fetch_symbol("SPY")
