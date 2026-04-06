"""
gmmhmm.py

Market-regime detection on multi-dimensional index time-series using
a Gaussian-Mixture Hidden Markov Model (GMM-HMM).

This version is intentionally aligned with regime_detector.py:
it uses the SAME feature set so the comparison is fair.

Pipeline
    1. Feature engineering   – rolling returns, volatility, correlations,
                               momentum z-scores, RSI
    2. Standardisation       – per-feature z-score (StandardScaler)
    3. Dimensionality reduction (optional PCA)
    4. Regime-number search  – BIC over candidate hidden-state counts
    5. GMM-HMM fit           – infer hidden market regimes
    6. Regime profiling      – mean return / vol / Sharpe per regime
    7. Trading signals       – simple regime-conditioned allocation
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════
#  1. FEATURE ENGINEERING — MATCHES regime_detector.py
# ═══════════════════════════════════════════════════════════════════════════

def _log_returns(close: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """Multi-horizon log returns for each asset."""
    frames = {}
    for p in periods:
        for col in close.columns:
            frames[f"{col}_ret_{p}d"] = np.log(close[col] / close[col].shift(p))
    return pd.DataFrame(frames, index=close.index)


def _rolling_volatility(close: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Annualised rolling volatility of daily log returns for each asset."""
    daily = np.log(close / close.shift(1))
    frames = {}
    for w in windows:
        for col in daily.columns:
            frames[f"{col}_vol_{w}d"] = daily[col].rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(frames, index=close.index)


def _rolling_correlation(close: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Rolling pairwise correlations (upper triangle only)."""
    daily = np.log(close / close.shift(1))
    cols = list(daily.columns)
    frames = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair = f"corr_{cols[i]}_{cols[j]}_{window}d"
            frames[pair] = daily[cols[i]].rolling(window).corr(daily[cols[j]])
    return pd.DataFrame(frames, index=close.index)


def _momentum_zscore(
    close: pd.DataFrame,
    lookback: int = 63,
    ma_window: int = 21,
) -> pd.DataFrame:
    """Z-score of current price relative to rolling mean."""
    frames = {}
    for col in close.columns:
        ma = close[col].rolling(ma_window).mean()
        std = close[col].rolling(lookback).std()
        frames[f"{col}_zscore_{lookback}d"] = (close[col] - ma) / std
    return pd.DataFrame(frames, index=close.index)


def _rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index per asset."""
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
    return_periods: Optional[List[int]] = None,
    vol_windows: Optional[List[int]] = None,
    corr_window: int = 63,
    zscore_lookback: int = 63,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """
    Assemble the feature matrix.

    This is intentionally the same feature set as regime_detector.py.
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
    logger.info("Feature matrix: %d rows × %d cols", *features.shape)
    return features


# ═══════════════════════════════════════════════════════════════════════════
#  2. MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HMMSelectionResult:
    n_range: List[int]
    log_likelihoods: List[float]
    bic_values: List[float]
    best_n_bic: int


@dataclass
class RegimeProfile:
    regime_id: int
    n_days: int
    mean_daily_return: float
    volatility: float
    sharpe: float
    avg_correlation: float
    pct_of_sample: float


def _approx_gmmhmm_n_params(
    n_states: int,
    n_mix: int,
    n_features: int,
    covariance_type: str = "diag",
) -> int:
    """
    Approximate free-parameter count for BIC comparison.
    """
    startprob_params = n_states - 1
    transmat_params = n_states * (n_states - 1)
    mixweight_params = n_states * (n_mix - 1)
    mean_params = n_states * n_mix * n_features

    if covariance_type == "diag":
        cov_params = n_states * n_mix * n_features
    elif covariance_type == "full":
        cov_params = n_states * n_mix * (n_features * (n_features + 1) // 2)
    else:
        raise ValueError("covariance_type must be 'diag' or 'full'.")

    return (
        startprob_params
        + transmat_params
        + mixweight_params
        + mean_params
        + cov_params
    )


def select_hmm_regimes(
    X: np.ndarray,
    n_min: int = 2,
    n_max: int = 8,
    n_mix: int = 2,
    covariance_type: str = "diag",
    n_iter: int = 300,
    random_state: int = 42,
) -> HMMSelectionResult:
    """Choose number of hidden states by BIC."""
    n_range = list(range(n_min, n_max + 1))
    log_likelihoods = []
    bic_values = []

    n_obs, n_features = X.shape

    for n_states in n_range:
        model = GMMHMM(
            n_components=n_states,
            n_mix=n_mix,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        model.fit(X)
        ll = model.score(X)

        k_params = _approx_gmmhmm_n_params(
            n_states=n_states,
            n_mix=n_mix,
            n_features=n_features,
            covariance_type=covariance_type,
        )
        bic = -2.0 * ll + k_params * np.log(n_obs)

        log_likelihoods.append(ll)
        bic_values.append(bic)
        logger.info("states=%d  logL=%.2f  BIC=%.2f", n_states, ll, bic)

    best_n = n_range[int(np.argmin(bic_values))]
    return HMMSelectionResult(
        n_range=n_range,
        log_likelihoods=log_likelihoods,
        bic_values=bic_values,
        best_n_bic=best_n,
    )


def _profile_regimes(
    close: pd.DataFrame,
    labels: pd.Series,
    features: pd.DataFrame,
) -> List[RegimeProfile]:
    """Compute summary statistics for each inferred regime."""
    daily_ret = np.log(close / close.shift(1)).mean(axis=1)
    daily_ret = daily_ret.reindex(labels.index)

    corr_cols = [c for c in features.columns if c.startswith("corr_")]
    profiles = []

    for rid in sorted(labels.unique()):
        mask = labels == rid
        n = int(mask.sum())
        r = daily_ret[mask]
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        sharpe = ann_ret / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else 0.0
        avg_corr = features.loc[mask, corr_cols].mean().mean() if corr_cols else np.nan

        profiles.append(
            RegimeProfile(
                regime_id=int(rid),
                n_days=n,
                mean_daily_return=float(ann_ret),
                volatility=float(ann_vol) if pd.notna(ann_vol) else np.nan,
                sharpe=float(sharpe),
                avg_correlation=float(avg_corr) if pd.notna(avg_corr) else np.nan,
                pct_of_sample=float(n / len(labels)),
            )
        )

    return profiles


# ═══════════════════════════════════════════════════════════════════════════
#  3. MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════

class HMMRegimeDetector:
    """
    End-to-end regime detection via GMM-HMM on engineered features.

    Parameters
    ----------
    close           : DataFrame of closing prices (columns = asset/index names)
    n_regimes       : Number of hidden states. None = select by BIC
    n_mix           : Number of Gaussian mixtures per hidden state
    use_pca         : Whether to apply PCA before HMM fitting
    pca_variance    : Variance explained target if PCA is used
    covariance_type : 'diag' or 'full'
    """

    def __init__(
        self,
        close: pd.DataFrame,
        n_regimes: Optional[int] = None,
        n_mix: int = 2,
        use_pca: bool = True,
        pca_variance: float = 0.95,
        return_periods: Optional[List[int]] = None,
        vol_windows: Optional[List[int]] = None,
        corr_window: int = 63,
        zscore_lookback: int = 63,
        rsi_period: int = 14,
        covariance_type: str = "diag",
        n_iter: int = 300,
        random_state: int = 42,
    ):
        self.close = close.copy()
        self.n_regimes = n_regimes
        self.n_mix = n_mix
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.return_periods = return_periods or [1, 5, 21]
        self.vol_windows = vol_windows or [21, 63]
        self.corr_window = corr_window
        self.zscore_lookback = zscore_lookback
        self.rsi_period = rsi_period
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.features_raw: Optional[pd.DataFrame] = None
        self.features_scaled: Optional[np.ndarray] = None
        self.features_reduced: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.hmm_selection: Optional[HMMSelectionResult] = None
        self.hmm: Optional[GMMHMM] = None
        self.labels: Optional[pd.Series] = None
        self.state_probs: Optional[pd.DataFrame] = None
        self.profiles: Optional[List[RegimeProfile]] = None
        self.regime_names: Dict[int, str] = {}

    def fit(self) -> "HMMRegimeDetector":
        """Run full pipeline: features -> scale -> PCA -> HMM -> profile."""
        self.features_raw = build_features(
            self.close,
            return_periods=self.return_periods,
            vol_windows=self.vol_windows,
            corr_window=self.corr_window,
            zscore_lookback=self.zscore_lookback,
            rsi_period=self.rsi_period,
        )

        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features_raw)

        if self.use_pca:
            self.pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
            self.features_reduced = self.pca.fit_transform(self.features_scaled)
        else:
            self.features_reduced = self.features_scaled

        X = self.features_reduced

        if self.n_regimes is None:
            self.hmm_selection = select_hmm_regimes(
                X,
                n_min=2,
                n_max=8,
                n_mix=self.n_mix,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.n_regimes = self.hmm_selection.best_n_bic

        self.hmm = GMMHMM(
            n_components=self.n_regimes,
            n_mix=self.n_mix,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.hmm.fit(X)

        raw_labels = self.hmm.predict(X)
        raw_probs = self.hmm.predict_proba(X)

        daily_ret = np.log(self.close / self.close.shift(1)).mean(axis=1)
        daily_ret = daily_ret.reindex(self.features_raw.index)
        mean_ret = pd.Series(raw_labels, index=self.features_raw.index).groupby(raw_labels).apply(
            lambda g: daily_ret.loc[g.index].mean()
        )

        rank_map = {old: new for new, old in enumerate(mean_ret.sort_values().index)}
        ordered = np.array([rank_map[l] for l in raw_labels])
        self.labels = pd.Series(ordered, index=self.features_raw.index, name="regime")

        prob_df = pd.DataFrame(
            raw_probs,
            index=self.features_raw.index,
            columns=[f"state_{i}" for i in range(self.n_regimes)],
        )
        ordered_prob_df = pd.DataFrame(index=prob_df.index)
        for old_state, new_state in rank_map.items():
            ordered_prob_df[f"state_{new_state}"] = prob_df[f"state_{old_state}"]
        self.state_probs = ordered_prob_df

        self.profiles = _profile_regimes(self.close, self.labels, self.features_raw)
        self._auto_name_regimes()
        return self

    def _auto_name_regimes(self):
        """Assign simple descriptive names from return-vol profiles."""
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

    def summary(self) -> pd.DataFrame:
        """Return a tidy regime summary."""
        rows = []
        for p in self.profiles:
            rows.append({
                "Regime": p.regime_id,
                "Name": self.regime_names.get(p.regime_id, ""),
                "Days": p.n_days,
                "% of Sample": f"{p.pct_of_sample:.1%}",
                "Ann. Return": f"{p.mean_daily_return:.2%}",
                "Ann. Vol": f"{p.volatility:.2%}" if pd.notna(p.volatility) else "nan",
                "Sharpe": f"{p.sharpe:.2f}",
                "Avg Corr": f"{p.avg_correlation:.2f}" if pd.notna(p.avg_correlation) else "nan",
            })
        return pd.DataFrame(rows)

    def transition_matrix(self) -> pd.DataFrame:
        """Return the fitted transition matrix."""
        if self.hmm is None:
            raise RuntimeError("Call .fit() first.")

        tm = pd.DataFrame(
            self.hmm.transmat_,
            index=[f"R{i}" for i in range(self.n_regimes)],
            columns=[f"R{i}" for i in range(self.n_regimes)],
        )
        return tm

    def generate_signals(
        self,
        bull_regimes: Optional[List[int]] = None,
        bear_regimes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Generate simple regime-conditioned portfolio weights."""
        if self.labels is None:
            raise RuntimeError("Call .fit() first.")

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

        signals = pd.DataFrame(index=self.labels.index)
        signals["regime"] = self.labels
        signals["regime_name"] = self.labels.map(self.regime_names)
        signals["weight"] = self.labels.map(weight_map)
        if self.state_probs is not None:
            signals["confidence"] = self.state_probs.max(axis=1)
        return signals

    def backtest(self, initial_capital: float = 100, signals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Simple long-only basket backtest."""
        if signals is None:
            signals = self.generate_signals()

        # Use simple (arithmetic) returns so weighting is valid
        simple_ret = (self.close / self.close.shift(1) - 1).mean(axis=1)
        simple_ret = simple_ret.reindex(signals.index)

        bt = pd.DataFrame(index=signals.index)
        bt["basket_ret"] = simple_ret
        bt["strategy_ret"] = simple_ret * signals["weight"].shift(1).fillna(0)

        # Cumulative returns via compounding simple returns
        bt["cum_basket"] = initial_capital * (1 + bt["basket_ret"]).cumprod()
        bt["cum_strategy"] = initial_capital * (1 + bt["strategy_ret"]).cumprod()

        # Sharpe ratios (single scalar, not a column repeated across rows)
        std_basket = bt["basket_ret"].std()
        std_strategy = bt["strategy_ret"].std()
        bt["sharpe_basket"] = (bt["basket_ret"].mean() / std_basket * np.sqrt(252)) if std_basket > 0 else 0.0
        bt["sharpe_strategy"] = (bt["strategy_ret"].mean() / std_strategy * np.sqrt(252)) if std_strategy > 0 else 0.0

        #Calmar ratios
        max_drawdown_basket = (bt["cum_basket"].cummax() - bt["cum_basket"]).max() / bt["cum_basket"].cummax().max()
        max_drawdown_strategy = (bt["cum_strategy"].cummax() - bt["cum_strategy"]).max() / bt["cum_strategy"].cummax().max()
        bt["calmar_basket"] = (bt["basket_ret"].mean() * 252) / (max_drawdown_basket) if max_drawdown_basket > 0 else 0.0
        bt["calmar_strategy"] = (bt["strategy_ret"].mean() * 252) / (max_drawdown_strategy) if max_drawdown_strategy > 0 else 0.0

        # Annualised volatility
        bt["ann_vol_basket"] = bt["basket_ret"].std() * np.sqrt(252)
        bt["ann_vol_strategy"] = bt["strategy_ret"].std() * np.sqrt(252)

        #Annualised return
        bt["ann_ret_basket"] = bt["basket_ret"].mean() * 252
        bt["ann_ret_strategy"] = bt["strategy_ret"].mean() * 252
  
        return bt

    def plot_regimes(self, figsize: Tuple[int, int] = (18, 14)):
        """Six-panel summary chart."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if self.labels is None:
            raise RuntimeError("Call .fit() first.")

        fig, axes = plt.subplots(6, 1, figsize=figsize)
        fig.suptitle("Market Regime Detection — GMM-HMM", fontsize=15, y=0.98)

        _REGIME_COLORS = {
            2: ["#D32F2F", "#2E7D32"],
            3: ["#D32F2F", "#F9A825", "#2E7D32"],
            4: ["#D32F2F", "#EF6C00", "#9ACD32", "#2E7D32"],
            5: ["#D32F2F", "#EF6C00", "#F9A825", "#9ACD32", "#2E7D32"],
        }
        if self.n_regimes in _REGIME_COLORS:
            colours = _REGIME_COLORS[self.n_regimes]
        else:
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "bear_bull", ["#D32F2F", "#F9A825", "#2E7D32"], N=self.n_regimes
            )
            colours = [cmap(i / max(self.n_regimes - 1, 1)) for i in range(self.n_regimes)]

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

        ax = axes[2]
        if self.hmm_selection is not None:
            ax2 = ax.twinx()
            ax.plot(self.hmm_selection.n_range, self.hmm_selection.log_likelihoods, "o-", color="steelblue", label="Log-Likelihood")
            ax2.plot(self.hmm_selection.n_range, self.hmm_selection.bic_values, "s--", color="coral", label="BIC")
            ax.axvline(self.n_regimes, ls=":", color="grey", label=f"chosen states={self.n_regimes}")
            ax.set_xlabel("Number of states")
            ax.set_ylabel("Log-Likelihood", color="steelblue")
            ax2.set_ylabel("BIC", color="coral")
            ax.set_title("HMM State Selection")
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.text(0.5, 0.5, f"States={self.n_regimes} (user-set)", ha="center", va="center")
            ax.set_title("HMM State Selection")

        ax = axes[3]
        rids = [p.regime_id for p in self.profiles]
        rets = [p.mean_daily_return for p in self.profiles]
        vols = [p.volatility for p in self.profiles]
        x = np.arange(len(rids))
        w = 0.35
        ax.bar(x - w / 2, rets, w, color=[colours[r] for r in rids], label="Ann. Return")
        ax.bar(x + w / 2, vols, w, color=[colours[r] for r in rids], alpha=0.5, label="Ann. Vol")
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{r}\n{self.regime_names.get(r, '')}" for r in rids], fontsize=7)
        ax.set_title("Return & Volatility by Regime")
        ax.legend(fontsize=7)
        ax.axhline(0, color="black", lw=0.5)

        ax = axes[4]
        if self.features_reduced is not None and self.features_reduced.shape[1] >= 2:
            for rid in sorted(self.labels.unique()):
                mask = self.labels.values == rid
                ax.scatter(
                    self.features_reduced[mask, 0],
                    self.features_reduced[mask, 1],
                    c=[colours[rid]], s=8, alpha=0.5, label=f"R{rid}",
                )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA — First Two Components")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "PCA not available", ha="center", va="center")

        ax = axes[5]
        bt = self.backtest()
        ax.plot(bt.index, bt["cum_basket"], label="Buy & Hold", color="grey", alpha=0.8)
        ax.plot(bt.index, bt["cum_strategy"], label="Regime Strategy", color="teal", lw=1.5)
        ax.set_title("Log-Cumulative Performance")
        ax.set_xlabel("Date")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()
        return fig
