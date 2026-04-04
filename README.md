# Market Regime Detection via PCA and K-Means Clustering

**COMP0040 — Machine Learning** | University College London | 

*Zhidian Jing, Siqi Zhang, Stefaan Fernando, Daniel Greenstein, Oscar Peyron*

---

## Overview

This project applies **Principal Component Analysis (PCA)** and **K-Means clustering** to detect market regimes (bullish/bearish) in multi-dimensional financial time-series. We analyse 9 major stock indices across three regions — America, Europe, and Asia — over the period 2006–2020.

We benchmark our PCA K-Means pipeline against three alternative regime detection methods:

- **GMMHMM** — Hidden Markov Model with Gaussian Mixture emissions
- **Sliced-Wasserstein K-Means** — optimal transport-based clustering
- **Peak-to-Trough** — rule-based labelling using a 20% drawdown threshold

Each method's quality is evaluated through a simple **long/flat trading strategy**: go long during bullish regimes, hold cash otherwise.

## Key Results

| Region   | Method        | Cumulative Return (%) | Sharpe Ratio |
|----------|---------------|----------------------:|-------------:|
| American | PCA K-Means   |                34,808 |         4.37 |
| American | Long Only     |                   293 |         0.51 |
| European | PCA K-Means   |                39,533 |         3.91 |
| European | Long Only     |                   149 |         0.24 |
| Asian    | PCA K-Means   |                23,226 |         3.76 |
| Asian    | Long Only     |                   174 |         0.30 |

## Project Structure

```
├── algo_regime/src/          # Core regime detection pipeline
│   ├── regime_detector.py    # PCA + K-Means regime detection
│   ├── sWkmean.py            # Sliced-Wasserstein K-Means
│   └── metrics.py            # Backtest performance metrics
├── data/src/                 # Data loading and configuration
│   ├── data_loader.py        # Yahoo Finance downloader & feature engineering
│   └── config/               # Index tickers and parameters
├── gmmhmm.py                 # GMMHMM regime detection
├── results.ipynb             # Main notebook: experiments and figures
├── report.tex                # LaTeX report source
├── report.pdf                # Compiled report
└── graphs/                   # Generated figures
    ├── k_selection_*.png     # Elbow/silhouette plots per region
    ├── regime_profile_*.png  # Regime overlays on price series
    ├── cluster_pca_*.png     # PCA cluster visualisations
    └── cumulative_returns_comparison_*.png
```

## Pipeline

1. **Data ingestion** — Download daily OHLCV data for 9 indices via Yahoo Finance
2. **Feature engineering** — Compute log-returns (1d, 5d, 21d), rolling volatility (21d, 63d), pairwise correlations, momentum z-scores, and RSI → 24 features per region
3. **Standardisation & PCA** — Z-score normalise, then reduce to components retaining 95% variance
4. **K-Means clustering** — Select optimal K via silhouette score, cluster, and relabel regimes by mean return
5. **Backtesting** — Long/flat strategy evaluated with cumulative return, Sharpe ratio, annualised return/volatility, and Calmar ratio

## Requirements

```
python >= 3.10
numpy
pandas
scikit-learn
scipy
hmmlearn
matplotlib
yfinance
```

Install dependencies:
```bash
pip install numpy pandas scikit-learn scipy hmmlearn matplotlib yfinance
```

## Usage

Run the full analysis in `results.ipynb`, or use the regime detector directly:

```python
from algo_regime.src.regime_detector import RegimeDetector

detector = RegimeDetector(df_data, use_pca = True)
detector.fit()
bt_data = detector.backtest(initial_capital = 100)
```

## Report

The full write-up is available in [`report.pdf`](report.pdf). To recompile:
```bash
latexmk -pdf report.tex
```