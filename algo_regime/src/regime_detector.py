"""
regime_detector.py
------------------
Market-regime detection on multi-dimensional index time-series using
K-Means clustering.

Pipeline
    1. Feature engineering   – rolling returns, volatility, correlations,
                               momentum, mean-reversion z-scores
    2. Standardisation       – per-feature z-score (StandardScaler)
    3. Dimensionality reduction (optional PCA)
    4. Optimal K selection   – elbow + silhouette + gap statistic
    5. K-Means clustering    – assign each trading day to a regime
    6. Regime profiling      – mean return / vol / Sharpe per regime
    7. Trading signals       – simple regime-conditioned allocation

Usage
-----
    from data_loader import get_close_prices
    from regime_detector import RegimeDetector

    close = get_close_prices("2010-01-01", "2024-01-01", regions=["europe"])
    rd = RegimeDetector(close)
    rd.fit()                         # runs full pipeline
    rd.plot_regimes()                # matplotlib summary
    signals = rd.generate_signals()  # trading overlay
"""

from __future__ import annotations

from ast import For
from fileinput import close
import logging
from queue import Full
from socket import close
from unittest import signals
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from algo_regime.src import metrics 
from algo_regime.src import sWkmean




logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def _log_returns(close: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    frames = {}
    close = close.astype(float)

    for p in periods:
        for col in close.columns:
            ratio = close[col] / close[col].shift(p)
            ratio = ratio.where(ratio > 0)   # keep only valid positive ratios
            frames[f"{col}_ret_{p}d"] = np.log(ratio)

    return pd.DataFrame(frames, index=close.index)

def _safe_daily_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns safely, leaving invalid entries as NaN.
    """
    close = close.astype(float)
    ratio = close / close.shift(1)
    ratio = ratio.where(ratio > 0)
    return np.log(ratio)

def _rolling_volatility(close: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Annualised rolling volatility of daily log returns."""
    close = close.astype(float)
    ratio = close / close.shift(1)
    ratio = ratio.where(ratio > 0)
    daily = np.log(ratio)
    frames = {}
    for w in windows:
        for col in daily.columns:
            frames[f"{col}_vol_{w}d"] = daily[col].rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(frames, index=close.index)


def _rolling_correlation(close: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Rolling pairwise correlations (upper triangle only)."""
    close = close.astype(float)
    ratio = close / close.shift(1)
    ratio = ratio.where(ratio > 0)
    daily = np.log(ratio)
    cols = list(daily.columns)
    frames = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair = f"corr_{cols[i]}_{cols[j]}_{window}d"
            frames[pair] = daily[cols[i]].rolling(window).corr(daily[cols[j]])
    return pd.DataFrame(frames, index=close.index)


def _momentum_zscore(close: pd.DataFrame, lookback: int = 63, ma_window: int = 21) -> pd.DataFrame:
    """Z-score of current price vs rolling mean (mean-reversion signal)."""
    frames = {}
    for col in close.columns:
        ma = close[col].rolling(ma_window).mean()
        std = close[col].rolling(lookback).std()
        frames[f"{col}_zscore_{lookback}d"] = (close[col] - ma) / std
    return pd.DataFrame(frames, index=close.index)


def _rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index per column."""
    frames = {}
    for col in close.columns:
        delta = close[col].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        frames[f"{col}_rsi_{period}d"] = 100 - 100 / (1 + rs)
    return pd.DataFrame(frames, index=close.index)


def build_features(
    close: pd.DataFrame,
    return_periods: List[int] = None,
    vol_windows: List[int] = None,
    corr_window: int = 63,
    zscore_lookback: int = 63,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """
    Assemble a feature matrix from closing prices.

    Each row = one trading day.  Features capture the *cross-sectional*
    state of the market on that day (multi-horizon returns, volatility
    regime, inter-market correlation structure, momentum, RSI).
    """
    if return_periods is None:
        return_periods = [1, 5, 21]
    if vol_windows is None:
        vol_windows = [21, 63]

    parts = [
        _log_returns(close, return_periods),
        _rolling_volatility(close, vol_windows),
        _rolling_correlation(close, corr_window),
        _momentum_zscore(close, zscore_lookback),
        _rsi(close, rsi_period),
    ]
    features = pd.concat(parts, axis=1).dropna()
    #print(features.columns)
    logger.info("Feature matrix: %d rows × %d cols", *features.shape)
    return features


def split_by_date(
    close: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
):
    train_close = close.loc[train_start:train_end].copy()
    val_close = close.loc[val_start:val_end].copy()
    test_close = close.loc[test_start:test_end].copy()
    return train_close, val_close, test_close

def get_extended_window(
    close: pd.DataFrame,
    start_date: str,
    end_date: str,
    warmup_bars: int,
):
    idx = close.index
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # first available trading day on or after start_date
    start_loc = idx.searchsorted(start_ts)
    if start_loc >= len(idx):
        raise ValueError("start_date is after the last available observation.")

    ext_start_loc = max(0, start_loc - warmup_bars)
    ext_start = idx[ext_start_loc]

    return close.loc[ext_start:end_ts].copy()

def evaluate_backtest(bt: pd.DataFrame, initial_capital: float = 100.0) -> Dict[str, float]:
    ret = bt["strategy_ret"].dropna()
    cum = bt["cum_strategy"].dropna()

    total_return = cum.iloc[-1] / initial_capital - 1
    ann_return = (cum.iloc[-1] / initial_capital) ** (252 / len(cum)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  2.  OPTIMAL-K SELECTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KSelectionResult:
    k_range: List[int]
    inertias: List[float]
    silhouettes: List[float]
    best_k_silhouette: int
    best_k_elbow: int


def _detect_elbow(inertias: List[float], k_range: List[int]) -> int:
    """Find the elbow via maximum second-derivative of the inertia curve."""
    if len(inertias) < 3:
        return k_range[0]
    d1 = np.diff(inertias)
    d2 = np.diff(d1)
    idx = int(np.argmax(np.abs(d2))) + 1  # +1 because of double-diff offset
    return k_range[min(idx, len(k_range) - 1)]


def select_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    n_init: int = 20,
    random_state: int = 42,
) -> KSelectionResult:
    """
    Evaluate K-Means for k in [k_min, k_max] and return diagnostics.
    """
    k_range = list(range(k_min, k_max + 1))
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        print(f'Fitting KMeans for k={k}... with X shape {X.shape}')
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        logger.info("  k=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, silhouettes[-1])

    best_sil = k_range[int(np.argmax(silhouettes))]
    best_elbow = _detect_elbow(inertias, k_range)

    return KSelectionResult(
        k_range=k_range,
        inertias=inertias,
        silhouettes=silhouettes,
        best_k_silhouette=best_sil,
        best_k_elbow=best_elbow,
    )

def _plot_elbw_silhouette(k_selection: KSelectionResult):
    """Plot inertia and silhouette curves for K selection."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(k_selection.k_range, k_selection.inertias, "o-", color="steelblue", label="Inertia")
    ax2.plot(k_selection.k_range, k_selection.silhouettes, "s--", color="coral", label="Silhouette")
    ax1.axvline(k_selection.best_k_silhouette, ls=":", color="grey", label=f"Best K (Silhouette)={k_selection.best_k_silhouette}")
    ax1.axvline(k_selection.best_k_elbow, ls=":", color="black", label=f"Best K (Elbow)={k_selection.best_k_elbow}")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia", color="steelblue")
    ax2.set_ylabel("Silhouette", color="coral")
    ax1.set_title("Optimal K Selection Diagnostics")
    fig.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  3.  REGIME PROFILING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeProfile:
    regime_id: int
    n_days: int
    mean_daily_return: float      # annualised
    volatility: float             # annualised
    sharpe: float
    avg_correlation: float
    pct_of_sample: float


def _profile_regimes(
    close: pd.DataFrame,
    labels: pd.Series,
    features: pd.DataFrame,
) -> List[RegimeProfile]:
    """Compute summary statistics for each regime cluster."""
    daily_ret = _safe_daily_log_returns(close).mean(axis=1)  # equal-weight basket
    daily_ret = daily_ret.reindex(labels.index)

    corr_cols = [c for c in features.columns if c.startswith("corr_")]
    profiles = []

    for rid in sorted(labels.unique()):
        mask = labels == rid
        n = int(mask.sum())
        r = daily_ret[mask]
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else 0.0
        avg_corr = features.loc[mask, corr_cols].mean().mean() if corr_cols else np.nan

        profiles.append(RegimeProfile(
            regime_id=rid,
            n_days=n,
            mean_daily_return=ann_ret,
            volatility=ann_vol,
            sharpe=sharpe,
            avg_correlation=avg_corr,
            pct_of_sample=n / len(labels),
        ))

    return profiles


# ═══════════════════════════════════════════════════════════════════════════
#  4.  MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    End-to-end regime detection via K-Means on engineered features
    from multi-index closing prices.

    Parameters
    ----------
    close           : DataFrame of closing prices (columns = index names).
    n_regimes       : Number of clusters.  None = auto-select via silhouette.
    use_pca         : Whether to apply PCA before clustering.
    pca_variance    : Cumulative variance to retain if PCA is used.
    return_periods  : Horizons for log returns  (default [1, 5, 21]).
    vol_windows     : Horizons for rolling vol  (default [21, 63]).
    random_state    : Seed for reproducibility.
    """

    def __init__(
        self,
        close: pd.DataFrame,
        n_regimes: Optional[int] = None,
        use_pca: bool = True,
        pca_variance: float = 0.95,
        return_periods: Optional[List[int]] = None,
        vol_windows: Optional[List[int]] = None,
        random_state: int = 42,
    ):
        self.close = close.copy()
        self.n_regimes = n_regimes
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.return_periods = return_periods or [1, 5, 21]
        self.vol_windows = vol_windows or [21, 63]
        self.random_state = random_state
        
        #Populated by .peak_to_trough_labelling()
        self.regime_labels_ptt: Optional[pd.Series] = None

        #Populated by swkmean algorithm
        self.labels_skmeans: Optional[pd.Series] = None

        # Populated by .fit()
        self.features_raw: Optional[pd.DataFrame] = None
        self.features_scaled: Optional[np.ndarray] = None
        self.features_reduced: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.k_selection: Optional[KSelectionResult] = None
        self.kmeans: Optional[KMeans] = None
        self.labels: Optional[pd.Series] = None
        self.profiles: Optional[List[RegimeProfile]] = None
        self.regime_names: Dict[int, str] = {}
        self.rank_map: Optional[Dict[int, int]] = None

        self.train_index: Optional[pd.Index] = None
        self.test_index: Optional[pd.Index] = None
        self.train_labels: Optional[pd.Series] = None
        self.test_labels: Optional[pd.Series] = None
        self.train_index_skmeans: Optional[pd.Index] = None
        self.test_index_skmeans: Optional[pd.Index] = None
        self.train_labels_skmeans: Optional[pd.Series] = None
        self.test_labels_skmeans: Optional[pd.Series] = None
        self.theta_skmeans = None
        self.centroids_skmeans = None
        self.rank_map_skmeans = None

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(self) -> "RegimeDetector":
        """
        Run full pipeline: features → scale → PCA → K-select → cluster → profile.
        
        Full-sample fit for descriptive analysis only.
        For bias-reduced out-of-sample evaluation, use fit_train_test().
   
        """
        # 1. features
        self.features_raw = build_features(
            self.close,
            return_periods=self.return_periods,
            vol_windows=self.vol_windows,
        )

        # 2. standardise
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features_raw)

        print(self.features_raw.shape)
        # 3. optional PCA
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
            self.features_reduced = self.pca.fit_transform(self.features_scaled)
            logger.info(
                "PCA: %d → %d components (%.1f%% variance)",
                self.features_scaled.shape[1],
                self.features_reduced.shape[1],
                self.pca_variance * 100,
            )
        else:
            self.features_reduced = self.features_scaled

        X = self.features_reduced
        print(X.shape)

        print(f"shape of feature_reduced: {X.shape}")
        # 4. select K
        if self.n_regimes is None:
            self.k_selection = select_k(X, k_min=2, k_max=8, random_state=self.random_state)
            self.n_regimes = self.k_selection.best_k_silhouette
            logger.info(
                "Auto-selected K=%d (silhouette=%.4f)",
                self.n_regimes,
                max(self.k_selection.silhouettes),
            )

        # 5. final clustering
        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            n_init=30,
            random_state=self.random_state,
        )
        raw_labels = self.kmeans.fit_predict(X)

        # 6. order regimes by mean return (regime 0 = worst, regime K-1 = best)
        daily_ret = _safe_daily_log_returns(self.close).mean(axis=1)
        daily_ret = daily_ret.reindex(self.features_raw.index)
        mean_ret = pd.Series(raw_labels, index=self.features_raw.index).groupby(raw_labels).apply(
            lambda g: daily_ret.loc[g.index].mean()
        )
        self.rank_map = {old: new for new, old in enumerate(mean_ret.sort_values().index)}
        ordered = np.array([self.rank_map[l] for l in raw_labels])
        
        #print(ordered.shape)

        self.labels = pd.Series(ordered, index=self.features_raw.index, name="regime")

        # 7. profile
        self.profiles = _profile_regimes(self.close, self.labels, self.features_raw)
        self._auto_name_regimes()

        return self
    def peak_to_trough_labelling(self, drawdown_threshold: float = -0.1) -> pd.DataFrame:
        daily_ret = _safe_daily_log_returns(self.close).mean(axis=1)
        cumulative = (1 + daily_ret).cumprod()
        peaks = cumulative.cummax()
        drawdown = (cumulative - peaks) / peaks
        self.regime_labels_ptt = pd.Series(1, index=self.features_raw.index)
        self.regime_labels_ptt[drawdown <= drawdown_threshold] = 0
        return self.regime_labels_ptt

    def fit_train_test(self, split_date: str) -> "RegimeDetector":
        """
        Fit scaler / PCA / KMeans on the training sample only, then
        predict regimes for the test sample using the fitted objects.
        """
        self.features_raw = build_features(
            self.close,
            return_periods=self.return_periods,
            vol_windows=self.vol_windows,
        )

        train_features = self.features_raw.loc[self.features_raw.index < split_date].copy()
        test_features = self.features_raw.loc[self.features_raw.index >= split_date].copy()

        if train_features.empty or test_features.empty:
            raise ValueError("Train/test split produced an empty dataset. Check split_date.")

        self.train_index = train_features.index
        self.test_index = test_features.index

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(train_features)
        X_test_scaled = self.scaler.transform(test_features)

        if self.use_pca:
            self.pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
            X_train = self.pca.fit_transform(X_train_scaled)
            X_test = self.pca.transform(X_test_scaled)
            logger.info(
                "Train/Test PCA: %d → %d components (%.1f%% variance)",
                X_train_scaled.shape[1],
                X_train.shape[1],
                self.pca_variance * 100,
            )
        else:
            X_train = X_train_scaled
            X_test = X_test_scaled

        self.features_scaled = X_train_scaled
        self.features_reduced = X_train

        if self.n_regimes is None:
            self.k_selection = select_k(X_train, k_min=2, k_max=8, random_state=self.random_state)
            self.n_regimes = self.k_selection.best_k_silhouette
            logger.info("Auto-selected K=%d from TRAIN sample", self.n_regimes)

        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            n_init=30,
            random_state=self.random_state,
        )
        train_raw_labels = self.kmeans.fit_predict(X_train)
        test_raw_labels = self.kmeans.predict(X_test)

        daily_ret = _safe_daily_log_returns(self.close).mean(axis=1)
        daily_ret_train = daily_ret.reindex(train_features.index)

        mean_ret_train = pd.Series(train_raw_labels, index=train_features.index).groupby(train_raw_labels).apply(
            lambda g: daily_ret_train.loc[g.index].mean()
        )

        rank_map = {old: new for new, old in enumerate(mean_ret_train.sort_values().index)}

        ordered_train = np.array([rank_map[l] for l in train_raw_labels])
        ordered_test = np.array([rank_map[l] for l in test_raw_labels])

        self.train_labels = pd.Series(ordered_train, index=train_features.index, name="regime")
        self.test_labels = pd.Series(ordered_test, index=test_features.index, name="regime")
        self.labels = pd.concat([self.train_labels, self.test_labels]).sort_index()

        self.profiles = _profile_regimes(self.close, self.labels, self.features_raw)
        self._auto_name_regimes()

        return self
    

    


    
    def detect_regime_skmeans(self,N_S = 5, L = 100, h1 = 50, h2 = 10, epsilon = 1e-6) -> pd.Series:
        # Alternative regime detection using sliced Wasserstein k-means
        import algo_regime.src.sWkmean as ws
        import algo_regime.src.metrics as mt

        N = self.close.shape[0]
       #print(f"shape of close: {self.close.shape}")
        _, _, labels_skmeans = ws.max_mccd_unifortho_sim(N_S, self.close, self.n_regimes, L = L,  epsilon = 1e-6, h1 = h1, h2 = h2, metric = "CVaR")
        transformed_labels = mt.convert_prediction(N, labels_skmeans, h1, h2)
        self.labels_skmeans = pd.Series(transformed_labels.ravel(), index=self.close.index, name="regime_sWkmeans")
        print(self.labels_skmeans.shape)
        return self.labels_skmeans
    # ── naming ───────────────────────────────────────────────────────────
    def detect_regime_skmeans_train_test(
        self,
        split_date: str,
        N_S: int = 5,
        L: int = 100,
        h1: int = 50,
        h2: int = 10,
        epsilon: float = 1e-6,
        metric: str = "CVaR",
    ) -> pd.Series:
        """
        Train sWkmeans on the training sample only, then assign test windows
        to the fitted training centroids using the fixed training projections.

        Parameters
        ----------
        split_date : str
            Chronological train/test split date.
        N_S : int
            Number of repeated clustering runs on the training set.
        L : int
            Number of projection vectors.
        h1 : int
            Lifting window length.
        h2 : int
            Lifting stride.
        epsilon : float
            Convergence tolerance.
        metric : str
            Label ordering helper used inside sWkmeans, e.g. "CVaR".

        Returns
        -------
        pd.Series
            Combined train+test sWkmeans labels indexed by lifted-window dates.
        """
        import algo_regime.src.sWkmean as ws

        split_ts = pd.Timestamp(split_date)

        train_close = self.close.loc[self.close.index < split_ts].copy()
        if train_close.empty:
            raise ValueError("Training sample is empty. Check split_date.")

        # extend test window backwards so lifting on test can use pre-split history
        test_close_ext = get_extended_window(
            close=self.close,
            start_date=split_date,
            end_date=self.close.index.max().strftime("%Y-%m-%d"),
            warmup_bars=h1,
        )
        if test_close_ext.empty:
            raise ValueError("Extended test sample is empty. Check split_date.")

        result = ws.fit_predict_swkmeans_train_test(
            S_train=train_close.values,
            S_test_with_warmup=test_close_ext.values,
            train_price_index=train_close.index,
            test_price_index_with_warmup=test_close_ext.index,
            split_date=split_date,
            K=self.n_regimes,
            L=L,
            epsilon=epsilon,
            h1=h1,
            h2=h2,
            N_S=N_S,
            metric=metric,
            random_state=self.random_state,
        )

        # save useful internals
        self.train_index_skmeans = pd.Index(result["train_index"])
        self.test_index_skmeans = pd.Index(result["test_index"])
        self.theta_skmeans = result["theta"]
        self.centroids_skmeans = result["centroids"]
        self.rank_map_skmeans = result["rank_map"]

        self.train_labels_skmeans = pd.Series(
            result["labels_train"],
            index=self.train_index_skmeans,
            name="regime_skmeans",
        )
        self.test_labels_skmeans = pd.Series(
            result["labels_test"],
            index=self.test_index_skmeans,
            name="regime_skmeans",
        )

        self.labels_skmeans = pd.concat(
            [self.train_labels_skmeans, self.test_labels_skmeans]
        ).sort_index()

        return self.labels_skmeans
    def _auto_name_regimes(self):
        """Give regimes human-readable names based on return/vol profile."""
        if not self.profiles:
            return

        for p in self.profiles:
            ret, vol = p.mean_daily_return, p.volatility
            if ret < -0.02 and vol > 0.15:
                name = "Crisis"
            elif ret < 0 and vol > 0.12:
                name = "Stressed"
            elif abs(ret) < 0.05 and vol < 0.10:
                name = "Quiet / Range-bound"
            elif ret > 0.05 and vol < 0.12:
                name = "Low-vol Rally"
            elif ret > 0 and vol >= 0.12:
                name = "High-vol Rally"
            else:
                name = "Transition"
            self.regime_names[p.regime_id] = name

    # ── reporting ────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame summarising each regime."""
        rows = []
        for p in self.profiles:
            rows.append({
                "Regime": p.regime_id,
                "Name": self.regime_names.get(p.regime_id, ""),
                "Days": p.n_days,
                "% of Sample": f"{p.pct_of_sample:.1%}",
                "Ann. Return": f"{p.mean_daily_return:.2%}",
                "Ann. Vol": f"{p.volatility:.2%}",
                "Sharpe": f"{p.sharpe:.2f}",
                "Avg Corr": f"{p.avg_correlation:.2f}",
            })
        return pd.DataFrame(rows)

    # ── trading signals ──────────────────────────────────────────────────

    def generate_signals(
        self,
        bull_regimes: Optional[List[int]] = None,
        bear_regimes: Optional[List[int]] = None,
    ):
        """
        Produce signal DataFrames for the main KMeans model and, if available,
        for sWkmeans.

        Returns
        -------
        signals : pd.DataFrame
            Columns: regime, regime_ptt, regime_name, weight, weight_ptt
        signals_skmeans : pd.DataFrame or None
            Columns: regime_skmeans, weight_skmeans
        """
        if self.labels is None:
            raise RuntimeError("Call .fit() or .fit_train_test() first.")

        n = self.n_regimes

        if bull_regimes is None and bear_regimes is None:
            weight_map = {}
            for rid in range(n):
                if rid == n - 1:
                    weight_map[rid] = 1.0
                elif rid == 0:
                    weight_map[rid] = 0.0
                else:
                    weight_map[rid] = 0.5
        else:
            bull_regimes = bull_regimes or []
            bear_regimes = bear_regimes or []
            weight_map = {}
            for rid in range(n):
                if rid in bull_regimes:
                    weight_map[rid] = 1.0
                elif rid in bear_regimes:
                    weight_map[rid] = 0.0
                else:
                    weight_map[rid] = 0.5

        # main PCA+KMeans signals
        signals = pd.DataFrame(index=self.labels.index)
        signals["regime"] = self.labels
        signals["regime_ptt"] = (
            self.regime_labels_ptt.reindex(signals.index)
            if self.regime_labels_ptt is not None else np.nan
        )
        signals["regime_name"] = self.labels.map(self.regime_names)
        signals["weight"] = self.labels.map(weight_map)
        signals["weight_ptt"] = (
            signals["regime_ptt"].map(weight_map)
            if self.regime_labels_ptt is not None else np.nan
        )

        # sWkmeans signals
        signals_skmeans = None
        if self.labels_skmeans is not None:
            signals_skmeans = pd.DataFrame(index=self.labels_skmeans.index)
            signals_skmeans["regime_skmeans"] = self.labels_skmeans
            signals_skmeans["weight_skmeans"] = self.labels_skmeans.map(weight_map)

        return signals, signals_skmeans

   # ── back-test helper ─────────────────────────────────────────────────
    def backtest(self, initial_capital=100, signals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Simple long-only back-test: basket return × regime weight.
        Returns DataFrame with columns:
            basket_ret, strategy_ret, cum_basket, cum_strategy, sharpe_basket, sharpe_strategy
        """
        if signals is None:
            signals, _ = self.generate_signals()

        # Use simple (arithmetic) returns so weighting is valid
        simple_ret = (self.close / self.close.shift(1) - 1).mean(axis=1)
        print(f"shape of simple_ret: {simple_ret.shape}")
        simple_ret = simple_ret.reindex(signals.index)

        bt = pd.DataFrame(index=signals.index)
        bt["basket_ret"] = simple_ret

        bt["strategy_ret"] = bt["basket_ret"] * signals["weight"].shift(1)


        if "weight_ptt" in signals.columns:
            bt["strategy_ret_ptt"] = bt["basket_ret"] * signals["weight_ptt"].shift(1)

        bt = bt.dropna(subset=["basket_ret", "strategy_ret"])

        # Cumulative returns via compounding simple returns
        bt["cum_basket"] = initial_capital * (1 + bt["basket_ret"]).cumprod()
        bt["cum_strategy"] = initial_capital * (1 + bt["strategy_ret"]).cumprod()

        if "strategy_ret_ptt" in bt.columns:
            bt["cum_strategy_ptt"] = initial_capital * (1 + bt["strategy_ret_ptt"]).cumprod()

        # Sharpe ratios
        std_basket = bt["basket_ret"].std()
        std_strategy = bt["strategy_ret"].std()
        bt["sharpe_basket"] = (bt["basket_ret"].mean() / std_basket * np.sqrt(252)) if std_basket > 0 else 0.0
        bt["sharpe_strategy"] = (bt["strategy_ret"].mean() / std_strategy * np.sqrt(252)) if std_strategy > 0 else 0.0
       
        #Calmar ratios (using max drawdown from cumulative returns)
        bt["calmar_basket"] = bt["cum_basket"].iloc[-1] / abs(bt["cum_basket"].min()) if bt["cum_basket"].min() < 0 else np.inf
        bt["calmar_strategy"] = bt["cum_strategy"].iloc[-1] / abs(bt["cum_strategy"].min()) if bt["cum_strategy"].min() < 0 else np.inf

        #Annualised Return 
        bt["ann_return_basket"] = (bt["cum_basket"].iloc[-1] / initial_capital) ** (252 / len(bt)) - 1
        bt["ann_return_strategy"] = (bt["cum_strategy"].iloc[-1] / initial_capital) ** (252 / len(bt)) - 1

        #Annualised Volatility
        bt["ann_vol_basket"] = bt["basket_ret"].std() * np.sqrt(252)
        bt["ann_vol_strategy"] = bt["strategy_ret"].std() * np.sqrt(252)

        
        if "strategy_ret_ptt" in bt.columns:
            std_ptt = bt["strategy_ret_ptt"].std()
            bt["sharpe_strategy_ptt"] = (bt["strategy_ret_ptt"].mean() / std_ptt * np.sqrt(252)) if std_ptt > 0 else 0.0
            bt["calmar_strategy_ptt"] = bt["cum_strategy_ptt"].iloc[-1] / abs(bt["cum_strategy_ptt"].min()) if bt["cum_strategy_ptt"].min() < 0 else np.inf
            bt["ann_return_strategy_ptt"] = (bt["cum_strategy_ptt"].iloc[-1] / initial_capital) ** (252 / len(bt)) - 1
            bt["ann_vol_strategy_ptt"] = bt["strategy_ret_ptt"].std() * np.sqrt(252)

        return bt
    
    def backtest_test_only(self, initial_capital=100) -> pd.DataFrame:
        """
        Backtest only on the test sample after fit_train_test().
        """
        if self.test_index is None:
            raise RuntimeError("Call fit_train_test(split_date=...) first.")

        signals, _ = self.generate_signals()
        signals = signals.loc[self.test_index].copy()

        simple_ret = (self.close / self.close.shift(1) - 1).mean(axis=1)
        simple_ret = simple_ret.reindex(signals.index)

        bt = pd.DataFrame(index=signals.index)
        bt["basket_ret"] = simple_ret
        bt["strategy_ret"] = bt["basket_ret"] * signals["weight"].shift(1)

        bt = bt.dropna(subset=["basket_ret", "strategy_ret"])

        bt["cum_basket"] = initial_capital * (1 + bt["basket_ret"]).cumprod()
        bt["cum_strategy"] = initial_capital * (1 + bt["strategy_ret"]).cumprod()

        std_basket = bt["basket_ret"].std()
        std_strategy = bt["strategy_ret"].std()

        bt["sharpe_basket"] = (bt["basket_ret"].mean() / std_basket * np.sqrt(252)) if std_basket > 0 else 0.0
        bt["sharpe_strategy"] = (bt["strategy_ret"].mean() / std_strategy * np.sqrt(252)) if std_strategy > 0 else 0.0

        bt["ann_return_basket"] = (bt["cum_basket"].iloc[-1] / initial_capital) ** (252 / len(bt)) - 1
        bt["ann_return_strategy"] = (bt["cum_strategy"].iloc[-1] / initial_capital) ** (252 / len(bt)) - 1

        bt["ann_vol_basket"] = bt["basket_ret"].std() * np.sqrt(252)
        bt["ann_vol_strategy"] = bt["strategy_ret"].std() * np.sqrt(252)

        return bt

    def backtest_skmeans(self, initial_capital=100, signals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Back-test for sWkmeans regime detection on the available sWkmeans label index.
        """
        if signals is None:
            _, signals = self.generate_signals()

        if signals is None or signals.empty:
            raise RuntimeError("No sWkmeans signals available. Call detect_regime_skmeans_train_test() first.")

        simple_ret = (self.close / self.close.shift(1) - 1).mean(axis=1)
        simple_ret = simple_ret.reindex(signals.index)

        bt = pd.DataFrame(index=signals.index)
        bt["basket_ret"] = simple_ret
        bt["strategy_ret_skmeans"] = bt["basket_ret"] * signals["weight_skmeans"].shift(1)

        bt = bt.dropna(subset=["basket_ret", "strategy_ret_skmeans"])

        bt["cum_basket"] = initial_capital * (1 + bt["basket_ret"]).cumprod()
        bt["cum_strategy_skmeans"] = initial_capital * (1 + bt["strategy_ret_skmeans"]).cumprod()

        std_basket = bt["basket_ret"].std()
        std_skmeans = bt["strategy_ret_skmeans"].std()

        bt["sharpe_basket"] = (bt["basket_ret"].mean() / std_basket * np.sqrt(252)) if std_basket > 0 else 0.0
        bt["sharpe_strategy_skmeans"] = (bt["strategy_ret_skmeans"].mean() / std_skmeans * np.sqrt(252)) if std_skmeans > 0 else 0.0

        bt["ann_return_basket"] = (bt["cum_basket"].iloc[-1] / initial_capital) ** (252 / len(bt)) - 1
        bt["ann_return_strategy_skmeans"] = (bt["cum_strategy_skmeans"].iloc[-1] / initial_capital) ** (252 / len(bt)) - 1

        bt["ann_vol_basket"] = bt["basket_ret"].std() * np.sqrt(252)
        bt["ann_vol_strategy_skmeans"] = bt["strategy_ret_skmeans"].std() * np.sqrt(252)

        return bt
    def backtest_skmeans_test_only(self, initial_capital=100) -> pd.DataFrame:
        """
        Back-test sWkmeans on the test sample only.
        """
        if self.test_index_skmeans is None:
            raise RuntimeError("Call detect_regime_skmeans_train_test() first.")

        _, signals_skmeans = self.generate_signals()
        if signals_skmeans is None:
            raise RuntimeError("No sWkmeans signals available.")

        signals_skmeans = signals_skmeans.loc[self.test_index_skmeans].copy()

        simple_ret = (self.close / self.close.shift(1) - 1).mean(axis=1)
        simple_ret = simple_ret.reindex(signals_skmeans.index)

        bt = pd.DataFrame(index=signals_skmeans.index)
        bt["basket_ret"] = simple_ret
        bt["strategy_ret_skmeans"] = bt["basket_ret"] * signals_skmeans["weight_skmeans"].shift(1)

        bt = bt.dropna(subset=["basket_ret", "strategy_ret_skmeans"])

        bt["cum_basket"] = initial_capital * (1 + bt["basket_ret"]).cumprod()
        bt["cum_strategy_skmeans"] = initial_capital * (1 + bt["strategy_ret_skmeans"]).cumprod()

        std_basket = bt["basket_ret"].std()
        std_skmeans = bt["strategy_ret_skmeans"].std()

        bt["sharpe_basket"] = (bt["basket_ret"].mean() / std_basket * np.sqrt(252)) if std_basket > 0 else 0.0
        bt["sharpe_strategy_skmeans"] = (bt["strategy_ret_skmeans"].mean() / std_skmeans * np.sqrt(252)) if std_skmeans > 0 else 0.0

        bt["ann_return_basket"] = (bt["cum_basket"].iloc[-1] / initial_capital) ** (252 / len(bt)) - 1
        bt["ann_return_strategy_skmeans"] = (bt["cum_strategy_skmeans"].iloc[-1] / initial_capital) ** (252 / len(bt)) - 1

        bt["ann_vol_basket"] = bt["basket_ret"].std() * np.sqrt(252)
        bt["ann_vol_strategy_skmeans"] = bt["strategy_ret_skmeans"].std() * np.sqrt(252)

        return bt
    # ── plotting ─────────────────────────────────────────────────────────

    def plot_k_selection(self):
        """Plot elbow + silhouette diagnostics for K selection."""
        if self.k_selection is None:
            raise RuntimeError("Call .fit() first.")
        return _plot_elbw_silhouette(self.k_selection)

    def plot_regime_profile(self, figsize: Tuple[int, int] = (10,8)):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates 
        """Chart of regime profiles: return vs volatility, coloured by Sharpe."""

        fig, axes = plt.subplots(2, 1, figsize=figsize)
        _REGIME_COLORS = {
            2: ["#D32F2F", "#2E7D32"],                          # red, green
            3: ["#D32F2F", "#F9A825", "#2E7D32"],               # red, amber, green
            4: ["#D32F2F", "#EF6C00", "#9ACD32", "#2E7D32"],   # red, orange, yellow-green, green
            5: ["#D32F2F", "#EF6C00", "#F9A825", "#9ACD32", "#2E7D32"],
        }
        if self.n_regimes in _REGIME_COLORS:
            colours = _REGIME_COLORS[self.n_regimes]
        else:
            # Fallback: linear interpolation from red → green in HSV space
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "bear_bull", ["#D32F2F", "#F9A825", "#2E7D32"], N=self.n_regimes
            )
            colours = [cmap(i / max(self.n_regimes - 1, 1)) for i in range(self.n_regimes)]


        # ── (0,0) Normalised price series coloured by regime ─────────
        ax = axes[0]
        norm_close = self.close.reindex(self.labels.index)
        norm_close = norm_close / norm_close.iloc[0]
        basket = norm_close.mean(axis=1)
        for rid in sorted(self.labels.unique()):
            mask = self.labels == rid
            ax.scatter(
                basket.index[mask], basket[mask],
                c=[colours[rid]], s=4, alpha=0.7,
                label=f"R{rid}: {self.regime_names.get(rid, '')}",
            )
        ax.set_title("Equal-weight Basket — Coloured by Regime")
        ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    

        ax = axes[1]
        for rid in sorted(self.labels.unique()):
            mask = self.labels == rid
            ax.fill_between(
                self.labels.index, 0, 1,
                where=mask, color=colours[rid], alpha=0.6,
                label=f"R{rid}: {self.regime_names.get(rid, '')}",
            )
        ax.set_title("Regime Timeline")
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.tight_layout()
        return fig

    def plot_cluster_pca(self, figsize: Tuple[int, int] = (10, 8)):
        """Scatter of regime clusters in first two PCA components."""
        import matplotlib.pyplot as plt

        if self.features_reduced is None or self.features_reduced.shape[1] < 2:
            raise RuntimeError("PCA not available. Fit with use_pca=True and sufficient variance.")

        fig, ax = plt.subplots(figsize=figsize)
        _REGIME_COLORS = {
            2: ["#D32F2F", "#2E7D32"],                          # red, green
            3: ["#D32F2F", "#F9A825", "#2E7D32"],               # red, amber, green
            4: ["#D32F2F", "#EF6C00", "#9ACD32", "#2E7D32"],   # red, orange, yellow-green, green
            5: ["#D32F2F", "#EF6C00", "#F9A825", "#9ACD32", "#2E7D32"],
        }
        if self.n_regimes in _REGIME_COLORS:
            colours = _REGIME_COLORS[self.n_regimes]
        else:
            # Fallback: linear interpolation from red → green in HSV space
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "bear_bull", ["#D32F2F", "#F9A825", "#2E7D32"], N=self.n_regimes
            )
            colours = [cmap(i / max(self.n_regimes - 1, 1)) for i in range(self.n_regimes)]

        for rid in sorted(self.labels.unique()):
            mask = self.labels.values == rid
            ax.scatter(
                self.features_reduced[mask, 0],
                self.features_reduced[mask, 1],
                c=[colours[rid]], s=8, alpha=0.5,
                label=f"R{rid}: {self.regime_names.get(rid,'')}",
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA — First Two Components")
        ax.legend(fontsize=7)
        plt.tight_layout()
        return fig
    
    def plot_backtest(self, bt: Optional[pd.DataFrame] = None, figsize: Tuple[int, int] = (10, 6)):
        """Cumulative return of strategy vs basket."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if bt is None:
            bt = self.backtest(initial_capital=100)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(bt.index, bt["cum_basket"], label="Buy & Hold", color="grey", alpha=0.8)
        ax.plot(bt.index, bt["cum_strategy"], label="Regime Strategy", color="teal", lw=1.5)

        if "cum_strategy_ptt" in bt.columns:
            ax.plot(bt.index, bt["cum_strategy_ptt"], label="Regime Strategy (PTT)", color="coral", lw=1.5)

        ax.set_title("Log-Cumulative Performance")
        ax.set_xlabel("Date")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()
        return fig
    
    def plot_regimes(self, figsize: Tuple[int, int] = (18, 14)):
        """Four-panel summary chart."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(6, 1, figsize=figsize)
        fig.suptitle("Market Regime Detection — K-Means Clustering", fontsize=15, y=0.98)
        # Semantic colour scale: red (bearish R0) → amber → green (bullish RN-1)
        # Since regimes are ordered by mean return after fit(), this always
        # maps worst → red, best → green, with smooth interpolation between.
        _REGIME_COLORS = {
            2: ["#D32F2F", "#2E7D32"],                          # red, green
            3: ["#D32F2F", "#F9A825", "#2E7D32"],               # red, amber, green
            4: ["#D32F2F", "#EF6C00", "#9ACD32", "#2E7D32"],   # red, orange, yellow-green, green
            5: ["#D32F2F", "#EF6C00", "#F9A825", "#9ACD32", "#2E7D32"],
        }
        if self.n_regimes in _REGIME_COLORS:
            colours = _REGIME_COLORS[self.n_regimes]
        else:
            # Fallback: linear interpolation from red → green in HSV space
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "bear_bull", ["#D32F2F", "#F9A825", "#2E7D32"], N=self.n_regimes
            )
            colours = [cmap(i / max(self.n_regimes - 1, 1)) for i in range(self.n_regimes)]

        # ── (0,0) Normalised price series coloured by regime ─────────
        ax = axes[0]
        norm_close = self.close.reindex(self.labels.index)
        norm_close = norm_close / norm_close.iloc[0]
        basket = norm_close.mean(axis=1)
        for rid in sorted(self.labels.unique()):
            mask = self.labels == rid
            ax.scatter(
                basket.index[mask], basket[mask],
                c=[colours[rid]], s=4, alpha=0.7,
                label=f"R{rid}: {self.regime_names.get(rid, '')}",
            )
        ax.set_title("Equal-weight Basket — Coloured by Regime")
        ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # ── (0,1) Regime timeline ────────────────────────────────────
        ax = axes[1]
        for rid in sorted(self.labels.unique()):
            mask = self.labels == rid
            ax.fill_between(
                self.labels.index, 0, 1,
                where=mask, color=colours[rid], alpha=0.6,
                label=f"R{rid}: {self.regime_names.get(rid, '')}",
            )
        ax.set_title("Regime Timeline")
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # ── (1,0) Elbow + Silhouette ────────────────────────────────
        if self.k_selection is not None:
            ax = axes[2]
            ax2 = ax.twinx()
            ax.plot(self.k_selection.k_range, self.k_selection.inertias, "o-", color="steelblue", label="Inertia")
            ax2.plot(self.k_selection.k_range, self.k_selection.silhouettes, "s--", color="coral", label="Silhouette")
            ax.axvline(self.n_regimes, ls=":", color="grey", label=f"chosen K={self.n_regimes}")
            ax.set_xlabel("K")
            ax.set_ylabel("Inertia", color="steelblue")
            ax2.set_ylabel("Silhouette", color="coral")
            ax.set_title("Optimal K Selection")
            ax.legend(fontsize=7, loc="upper right")
        else:
            axes[2].text(0.5, 0.5, f"K={self.n_regimes} (user-set)", ha="center", va="center")

        # ── (1,1) Regime bar chart: return vs vol ────────────────────
        ax = axes[3]
        rids = [p.regime_id for p in self.profiles]
        rets = [p.mean_daily_return for p in self.profiles]
        vols = [p.volatility for p in self.profiles]
        x = np.arange(len(rids))
        w = 0.35
        ax.bar(x - w / 2, rets, w, color=[colours[r] for r in rids], label="Ann. Return")
        ax.bar(x + w / 2, vols, w, color=[colours[r] for r in rids], alpha=0.5, label="Ann. Vol")
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{r}\n{self.regime_names.get(r,'')}" for r in rids], fontsize=7)
        ax.set_title("Return & Volatility by Regime")
        ax.legend(fontsize=7)
        ax.axhline(0, color="black", lw=0.5)

        # ── (2,0) PCA scatter ────────────────────────────────────────
        ax = axes[4]
        if self.features_reduced is not None and self.features_reduced.shape[1] >= 2:
            for rid in sorted(self.labels.unique()):
                mask = self.labels.values == rid
                ax.scatter(
                    self.features_reduced[mask, 0],
                    self.features_reduced[mask, 1],
                    c=[colours[rid]], s=8, alpha=0.5,
                    label=f"R{rid}",
                )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA — First Two Components")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "PCA not available", ha="center")

        # ── (2,1) Cumulative back-test ───────────────────────────────
        ax = axes[5]
        bt = self.backtest_test_only() if self.test_index is not None else self.backtest()
        ax.plot(bt.index, bt["cum_basket"], label="Buy & Hold", color="grey", alpha=0.8)
        ax.plot(bt.index, bt["cum_strategy"], label="Regime Strategy", color="teal", lw=1.5)
        ax.set_title("Log-Cumulative Performance")
        ax.set_xlabel("Date")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()
        return fig

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "regime_output.csv"):
        """Save labels + signals to CSV."""
        signals = self.generate_signals()
        signals.to_csv(path)
        logger.info("Saved to %s", path)
        return path


# ═══════════════════════════════════════════════════════════════════════════
#  CLI DEMO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import matplotlib
    matplotlib.use("Agg")

    from data.src.data_loader import get_close_prices

    print("=" * 60)
    print(" Regime Detection — European Indices")
    print("=" * 60)

    close = get_close_prices("2010-01-01", "2024-06-01", regions=["europe"])
    print(f"\nClose prices: {close.shape}")
    print(close.tail(), "\n")

    rd = RegimeDetector(close, use_pca=True, pca_variance=0.95)
    rd.fit_train_test(split_date="2020-01-01")

    print("\n── Regime statisical description ──")
    print(rd.summary().to_string(index=False))

    print("\n── Signals (tail) ──")
    signals, signals_skmeans = rd.generate_signals()
    print(signals.tail(10))

    print("\n── Back-test (tail) ──")
    bt = rd.backtest_test_only()
    print(bt.tail(10))


    #fig = rd.plot_regimes()
    #fig.savefig("regime_report.png", dpi=150, bbox_inches="tight")
    #print("\nSaved regime_report.png")
 
