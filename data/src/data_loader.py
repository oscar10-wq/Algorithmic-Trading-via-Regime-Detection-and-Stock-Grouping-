"""
data_loader.py
--------------
Downloads daily OHLCV data for major stock indices from Yahoo Finance,
grouped by region.  Designed as the first stage of a clustering-based
trading pipeline.

Regions & tickers
    America : S&P 500 (^GSPC), Dow Jones (^DJI), Nasdaq Composite (^IXIC)
    Europe  : FTSE 100 (^FTSE), CAC 40 (^FCHI), DAX (^GDAXI)
    Asia    : KOSPI (^KS11), Nikkei 225 (^N225), Hang Seng (^HSI)

Usage
-----
    from data_loader import load_index_data

    # All regions
    df = load_index_data("2018-01-01", "2024-01-01")

    # Only Europe + Asia
    df = load_index_data("2020-01-01", "2024-01-01", regions=["europe", "asia"])

    # Raw per-index dict instead of a merged DataFrame
    data = load_index_data("2020-01-01", "2024-01-01", merge=False)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Index registry ──────────────────────────────────────────────────────────

INDICES: Dict[str, Dict[str, str]] = {
    "america": {
        "SP500": "^GSPC",
        "DowJones": "^DJI",
        "Nasdaq": "^IXIC",
    },
    "europe": {
        "FTSE100": "^FTSE",
        "CAC40": "^FCHI",
        "DAX": "^GDAXI",
    },
    "asia": {
        "KOSPI": "^KS11",
        "Nikkei225": "^N225",
        "HangSeng": "^HSI",
    },
}

ALL_REGIONS = list(INDICES.keys())


# ── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_regions(regions: Optional[List[str]] = None) -> List[str]:
    """Validate and normalise region names."""
    if regions is None:
        return ALL_REGIONS
    out = []
    for r in regions:
        r_low = r.strip().lower()
        if r_low not in INDICES:
            raise ValueError(
                f"Unknown region '{r}'. Choose from {ALL_REGIONS}"
            )
        out.append(r_low)
    return out


def _build_ticker_map(regions: List[str]) -> Dict[str, str]:
    """Return {friendly_name: yahoo_ticker} for the requested regions."""
    ticker_map: Dict[str, str] = {}
    for region in regions:
        ticker_map.update(INDICES[region])
    return ticker_map


def _download_single(
    name: str,
    ticker: str,
    start: str,
    end: str,
    interval: str,
) -> Optional[pd.DataFrame]:
    """Download one index; return None on failure."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            logger.warning("No data returned for %s (%s)", name, ticker)
            return None

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        df.columns = pd.MultiIndex.from_product(
            [[name], ["Open", "High", "Low", "Close", "Volume"]]
        )
        return df

    except Exception as exc:
        logger.error("Failed to download %s (%s): %s", name, ticker, exc)
        return None


# ── Feature engineering helpers (useful for clustering) ─────────────────────

def compute_returns(
    close_df: pd.DataFrame,
    periods: List[int] = None,
) -> pd.DataFrame:
    """
    From a DataFrame of closing prices (columns = index names),
    compute log-returns over multiple horizons.

    Parameters
    ----------
    close_df : DataFrame with DatetimeIndex; one column per index.
    periods  : Return horizons in trading days. Default [1, 5, 21]
               (daily, weekly, monthly).

    Returns
    -------
    DataFrame with MultiIndex columns: (index_name, f"ret_{p}d").
    """
    import numpy as np

    if periods is None:
        periods = [1, 5, 21]

    frames = []
    for col in close_df.columns:
        series = close_df[col]
        for p in periods:
            ret = np.log(series / series.shift(p))
            ret.name = (col, f"ret_{p}d")
            frames.append(ret)

    return pd.concat(frames, axis=1)


def compute_volatility(
    close_df: pd.DataFrame,
    windows: List[int] = None,
) -> pd.DataFrame:
    """
    Rolling annualised volatility of log-returns.

    Parameters
    ----------
    close_df : DataFrame of closing prices.
    windows  : Rolling window sizes in days. Default [21, 63]
               (1-month, 3-month).

    Returns
    -------
    DataFrame with MultiIndex columns: (index_name, f"vol_{w}d").
    """
    import numpy as np

    if windows is None:
        windows = [21, 63]

    log_ret = np.log(close_df / close_df.shift(1))
    frames = []
    for col in log_ret.columns:
        for w in windows:
            vol = log_ret[col].rolling(w).std() * np.sqrt(252)
            vol.name = (col, f"vol_{w}d")
            frames.append(vol)

    return pd.concat(frames, axis=1)


def build_feature_matrix(
    close_df: pd.DataFrame,
    return_periods: List[int] = None,
    vol_windows: List[int] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    One-stop feature builder for clustering.

    Combines multi-horizon returns and rolling volatilities into a single
    matrix where each row is a date and each column is
    (index_name, feature_name).

    Parameters
    ----------
    close_df       : Closing prices, one column per index.
    return_periods : Passed to compute_returns.
    vol_windows    : Passed to compute_volatility.
    dropna         : Drop rows with any NaN (warm-up period).

    Returns
    -------
    DataFrame ready for sklearn clustering / PCA.
    """
    rets = compute_returns(close_df, return_periods)
    vols = compute_volatility(close_df, vol_windows)
    features = pd.concat([rets, vols], axis=1)
    if dropna:
        features.dropna(inplace=True)
    return features


# ── Main loader ─────────────────────────────────────────────────────────────

def load_index_data(
    start_date: str,
    end_date: str,
    regions: Optional[List[str]] = None,
    interval: str = "1d",
    merge: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Download index OHLCV data from Yahoo Finance.

    Parameters
    ----------
    start_date : str  – ISO date, e.g. "2020-01-01".
    end_date   : str  – ISO date, e.g. "2024-01-01".
    regions    : list  – Any subset of ["america", "europe", "asia"].
                         None (default) downloads all three.
    interval   : str  – Bar size: "1d", "1wk", "1mo", etc.
    merge      : bool – If True, return one wide DataFrame with a
                         MultiIndex on columns (index_name, OHLCV).
                         If False, return a dict {name: DataFrame}.

    Returns
    -------
    pd.DataFrame or dict[str, pd.DataFrame]
    """
    regions = _resolve_regions(regions)
    ticker_map = _build_ticker_map(regions)

    logger.info(
        "Downloading %d indices for regions %s  [%s → %s]",
        len(ticker_map), regions, start_date, end_date,
    )

    results: Dict[str, pd.DataFrame] = {}
    frames: List[pd.DataFrame] = []

    for name, ticker in ticker_map.items():
        df = _download_single(name, ticker, start_date, end_date, interval)
        if df is not None:
            results[name] = df
            frames.append(df)

    if not frames:
        raise RuntimeError("All downloads failed – check network / dates.")

    if not merge:
        # Strip the MultiIndex so each frame has plain OHLCV columns
        plain = {}
        for name, df in results.items():
            flat = df.copy()
            flat.columns = flat.columns.get_level_values(1)
            plain[name] = flat
        return plain

    merged = pd.concat(frames, axis=1).sort_index()
    logger.info("Merged shape: %s", merged.shape)
    return merged


def get_close_prices(
    start_date: str,
    end_date: str,
    regions: Optional[List[str]] = None,
    interval: str = "1d",
    fill_method: Optional[str] = "ffill",
) -> pd.DataFrame:
    """
    Convenience wrapper – returns *only* closing prices as a simple
    DataFrame (columns = index names, rows = dates).

    Handy as direct input to ``build_feature_matrix``.

    Parameters
    ----------
    fill_method : How to handle NaN from different trading calendars.
                  "ffill" (default) forward-fills; None leaves gaps.
    """
    raw = load_index_data(
        start_date, end_date, regions=regions, interval=interval, merge=False
    )
    close = pd.DataFrame({name: df["Close"] for name, df in raw.items()})
    close.sort_index(inplace=True)

    if fill_method == "ffill":
        close.ffill(inplace=True)
    elif fill_method is not None:
        close.fillna(method=fill_method, inplace=True)

    return close


# ── CLI quick-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Downloading sample data  (2022-01-01 → 2024-01-01)")
    print("=" * 60)

    close = get_close_prices("2022-01-01", "2024-01-01")
    print(f"\nClose prices shape : {close.shape}")
    print(close.tail())

    print("\n── Feature matrix for clustering ──")
    features = build_feature_matrix(close)
    print(f"Feature matrix shape : {features.shape}")
    print(features.tail())
