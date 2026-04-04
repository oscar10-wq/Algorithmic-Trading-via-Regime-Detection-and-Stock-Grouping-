"""
tune_hyperparameters.py
-----------------------
Hyperparameter tuning for RegimeDetector (regime_detector.py).

HOW TO USE
----------
1. Put this file in the same folder as regime_detector.py and data_loader.py
2. Run:   python tune_hyperparameters.py
3. It will print a results table and save:
       - tuning_results.csv   ← every combination tried + its scores
       - best_regime_report.png  ← plot of the winning model

WHAT IT TUNES
-------------
The script tries every combination of these settings:

  n_regimes       – how many market regimes (clusters) to detect: 2, 3, 4, 5
  use_pca         – whether to compress features with PCA first: True, False
  pca_variance    – how much variance PCA keeps (only used when use_pca=True):
                    0.90, 0.95
  return_periods  – which time horizons to use for log-return features:
                    [1,5,21]  or  [1,5,21,63]
  vol_windows     – which windows to use for volatility features:
                    [21,63]  or  [10,21,63]

For each combination, it scores the model on THREE things:
  1. Silhouette Score  – how well-separated the clusters are (higher = better)
  2. Sharpe Ratio      – risk-adjusted return of the trading strategy (higher = better)
  3. Combined Score    – silhouette × 0.5 + sharpe × 0.5 (our main ranking metric)

The best combination by combined score is printed and saved.
"""

import itertools
import warnings
import logging
import pandas as pd
import numpy as np

# ── suppress noisy sklearn warnings ──────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)   # silence INFO logs during tuning

# ── import your project files ─────────────────────────────────────────────────
from data_loader import get_close_prices
from regime_detector import RegimeDetector


# =============================================================================
#  STEP 1 — Define the hyperparameter grid
#  Add or remove values here to expand / shrink the search.
# =============================================================================

PARAM_GRID = {
    "n_regimes":      [2, 3, 4, 5],
    "use_pca":        [True, False],
    "pca_variance":   [0.90, 0.95],      # only matters when use_pca=True
    "return_periods": [[1, 5, 21], [1, 5, 21, 63]],
    "vol_windows":    [[21, 63], [10, 21, 63]],
}


# =============================================================================
#  STEP 2 — Load data  (edit dates / regions to match your project)
# =============================================================================

print("Loading price data …")
close = get_close_prices("2010-01-01", "2024-01-01", regions=["europe"])
print(f"  Loaded {close.shape[0]} rows × {close.shape[1]} columns\n")


# =============================================================================
#  STEP 3 — Helper: score one RegimeDetector after fitting
# =============================================================================

def score_model(rd: RegimeDetector) -> dict:
    """
    Fit the model and return a dict of scores.
    Returns None if fitting fails (e.g. too few data points for k).
    """
    try:
        rd.fit()
    except Exception as e:
        return None   # skip broken combinations silently

    # ── Silhouette score (measures cluster quality, range -1 to +1) ──────────
    # We get this from the k_selection object if auto-K was used,
    # otherwise we compute it manually on the reduced features.
    from sklearn.metrics import silhouette_score as _sil

    try:
        sil = _sil(rd.features_reduced, rd.labels.values)
    except Exception:
        sil = np.nan

    # ── Sharpe ratio of the regime-based trading strategy ────────────────────
    bt = rd.backtest()
    strategy_daily = bt["strategy_ret"]
    if strategy_daily.std() > 0:
        sharpe = (strategy_daily.mean() * 252) / (strategy_daily.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # ── Combined score (equal weight between silhouette and sharpe) ──────────
    # We normalise both to be roughly comparable before combining.
    # Silhouette is already in [-1, 1].  Sharpe is typically in [-3, 3].
    # We just average them raw here — you can adjust weights if you prefer.
    combined = 0.5 * sil + 0.5 * (sharpe / 3.0)   # divide sharpe by 3 to scale it

    return {
        "silhouette": round(sil, 4),
        "sharpe":     round(sharpe, 4),
        "combined":   round(combined, 4),
    }


# =============================================================================
#  STEP 4 — Run the grid search
# =============================================================================

# Build every possible combination from the grid
keys   = list(PARAM_GRID.keys())
values = list(PARAM_GRID.values())
all_combinations = list(itertools.product(*values))

print(f"Testing {len(all_combinations)} hyperparameter combinations …\n")

results = []   # we'll collect one row per combination here

for i, combo in enumerate(all_combinations, start=1):
    # Turn the tuple into a dict  e.g. {"n_regimes": 3, "use_pca": True, ...}
    params = dict(zip(keys, combo))

    # If PCA is off, pca_variance doesn't matter — skip duplicates
    if not params["use_pca"] and params["pca_variance"] != PARAM_GRID["pca_variance"][0]:
        continue

    # Print progress
    print(f"[{i}/{len(all_combinations)}]  Testing: {params}", end="  →  ", flush=True)

    # Build and score the model
    rd = RegimeDetector(
        close,
        n_regimes      = params["n_regimes"],
        use_pca        = params["use_pca"],
        pca_variance   = params["pca_variance"],
        return_periods = params["return_periods"],
        vol_windows    = params["vol_windows"],
        random_state   = 42,          # fixed seed → reproducible results
    )

    scores = score_model(rd)

    if scores is None:
        print("FAILED (skipped)")
        continue

    print(f"silhouette={scores['silhouette']:.3f}  sharpe={scores['sharpe']:.3f}  combined={scores['combined']:.3f}")

    # Save this row
    results.append({
        "n_regimes":      params["n_regimes"],
        "use_pca":        params["use_pca"],
        "pca_variance":   params["pca_variance"] if params["use_pca"] else "N/A",
        "return_periods": str(params["return_periods"]),
        "vol_windows":    str(params["vol_windows"]),
        "silhouette":     scores["silhouette"],
        "sharpe":         scores["sharpe"],
        "combined_score": scores["combined"],
    })


# =============================================================================
#  STEP 5 — Find and print the best combination
# =============================================================================

if not results:
    print("\nNo combinations succeeded. Check your data_loader or data.")
else:
    df = pd.DataFrame(results).sort_values("combined_score", ascending=False)

    print("\n" + "=" * 70)
    print("  TUNING COMPLETE — All results (sorted best → worst)")
    print("=" * 70)
    print(df.to_string(index=False))

    # Save full results to CSV
    df.to_csv("tuning_results.csv", index=False)
    print("\n✓ Full results saved to tuning_results.csv")

    # ── Best model ───────────────────────────────────────────────────────────
    best = df.iloc[0]
    print("\n" + "=" * 70)
    print("  BEST HYPERPARAMETERS")
    print("=" * 70)
    print(f"  n_regimes      = {best['n_regimes']}")
    print(f"  use_pca        = {best['use_pca']}")
    print(f"  pca_variance   = {best['pca_variance']}")
    print(f"  return_periods = {best['return_periods']}")
    print(f"  vol_windows    = {best['vol_windows']}")
    print(f"\n  Silhouette     = {best['silhouette']}")
    print(f"  Sharpe Ratio   = {best['sharpe']}")
    print(f"  Combined Score = {best['combined_score']}")
    print("=" * 70)

    # ── Re-fit the best model and save its plot ───────────────────────────────
    print("\nRe-fitting best model and saving plot …")
    import ast
    import matplotlib
    matplotlib.use("Agg")

    best_rd = RegimeDetector(
        close,
        n_regimes      = int(best["n_regimes"]),
        use_pca        = bool(best["use_pca"]),
        pca_variance   = float(best["pca_variance"]) if best["pca_variance"] != "N/A" else 0.95,
        return_periods = ast.literal_eval(best["return_periods"]),
        vol_windows    = ast.literal_eval(best["vol_windows"]),
        random_state   = 42,
    )
    best_rd.fit()

    print("\n── Best Model Regime Summary ──")
    print(best_rd.summary().to_string(index=False))

    fig = best_rd.plot_regimes()
    fig.savefig("best_regime_report.png", dpi=150, bbox_inches="tight")
    print("\n✓ Plot saved to best_regime_report.png")

    print("\nDone! Copy the BEST HYPERPARAMETERS above into your RegimeDetector() call.")
