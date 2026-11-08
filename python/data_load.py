"""Download and prepare Yahoo Finance FX data."""
import numpy as np
import pandas as pd
import yfinance as yf

from config import DATA_DIR, MA_WINDOW, PRIMARY, REALIZED_VOL_WINDOW, TICKERS, TICKER_LABELS


def download_fx(start: str, end: str | None = None) -> pd.DataFrame:
    """Pull adjusted OHLCV for EUR/USD and GBP/USD; align on business days."""
    frames = []
    for t in TICKERS:
        raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"No data returned for {t}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=lambda c: f"{TICKER_LABELS[t]}_{c}")
        frames.append(raw)
    df = pd.concat(frames, axis=1).sort_index()
    df = df.dropna(how="any")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log returns, moving average, and realized volatility for primary pair."""
    label = TICKER_LABELS[PRIMARY]
    close = df[f"{label}_Close"]
    out = df.copy()
    out[f"{label}_log_return"] = np.log(close).diff()
    out[f"{label}_ma{MA_WINDOW}"] = close.rolling(MA_WINDOW).mean()
    out[f"{label}_realized_vol"] = (
        out[f"{label}_log_return"].rolling(REALIZED_VOL_WINDOW).std() * np.sqrt(252)
    )
    for t in TICKERS:
        if t == PRIMARY:
            continue
        lb = TICKER_LABELS[t]
        out[f"{lb}_log_return"] = np.log(out[f"{lb}_Close"]).diff()
    return out.dropna()


def train_test_split_ts(series: pd.Series, train_ratio: float):
    """Chronological split — no shuffle."""
    n = len(series)
    split = int(n * train_ratio)
    return series.iloc[:split], series.iloc[split:]


def cache_data(df: pd.DataFrame, name: str = "fx_panel.csv") -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    df.to_csv(path)
    return df


def load_or_download(start: str, end: str | None = None) -> pd.DataFrame:
    path = DATA_DIR / "fx_panel.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        if len(df) > 500:
            return df
    df = download_fx(start, end)
    df = add_features(df)
    return cache_data(df)
