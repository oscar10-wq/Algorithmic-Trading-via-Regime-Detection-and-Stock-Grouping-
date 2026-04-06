"""
tune_hyperparameters.py
-----------------------
This systematically finds the best hyperparameter settings
for the RegimeDetector model. Rather than guessing which settings work best,
this tests every combination automatically and scores each one so we can
make a data-driven decision about what to use in the final model.
"""

import itertools
import warnings
import logging
import pandas as pd
import numpy as np

# Silencing unnecessary warnings that clutter the output while the grid search runs
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# Importing the project's own modules — data_loader fetches the closing prices,
# RegimeDetector is the main model we are tuning
#from data_loader import get_close_prices
#from regime_detector import RegimeDetector

from data.src.data_loader import get_close_prices
from algo_regime.src.regime_detector import RegimeDetector

REGION = "america"  # Change this to "europe" or "asia" as needed
SPLIT_DATE = "2020-01-01"
START_DATE = "2010-01-01"
END_DATE = "2024-01-01"
# -----------------------------------------------------------------------------
# HYPERPARAMETER GRID
# -----------------------------------------------------------------------------
# This is where I define all the values I want to test for each hyperparameter.
# The script will try every possible combination of these automatically.
#
# n_regimes      - how many market regimes (clusters) to split the data into.
#                  e.g. 3 could represent bear / neutral / bull markets.
#                  Too few and we miss important distinctions, too many and
#                  the regimes become too fragmented to be useful.
#
# use_pca        - whether to apply PCA (dimensionality reduction) before
#                  clustering. The feature matrix can have 30-50 columns,
#                  and K-Means struggles in high dimensions, so PCA can help
#                  by compressing those into a smaller set of components.
#
# pca_variance   - controls how much information PCA keeps. 0.95 means we
#                  keep enough components to explain 95% of the variance.
#                  Only relevant when use_pca is True.
#
# return_periods - the time horizons (in trading days) used to calculate
#                  log return features. [1, 5, 21] covers daily, weekly,
#                  monthly. Adding 63 tests whether quarterly momentum
#                  helps the model detect regimes more accurately.
#
# vol_windows    - the rolling windows used to measure volatility. Adding
#                  a 10-day window tests whether short-term vol spikes
#                  (like a sudden market selloff) are useful signals.
# -----------------------------------------------------------------------------

PARAM_GRID = {
    "n_regimes":      [2, 3, 4, 5],
    "use_pca":        [True, False],
    "pca_variance":   [0.90, 0.95],
    "return_periods": [[1, 5, 21], [1, 5, 21, 63]],
    "vol_windows":    [[21, 63], [10, 21, 63]],
}


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
# Pulling in the European index closing prices for the same date range used
# in the rest of the project so results are directly comparable
# -----------------------------------------------------------------------------

print("Loading price data ...")
close = get_close_prices(START_DATE, END_DATE, regions=[REGION])
print(f"  Loaded {close.shape[0]} rows x {close.shape[1]} columns\n")


# -----------------------------------------------------------------------------
# SCORING FUNCTION
# -----------------------------------------------------------------------------
# For each hyperparameter combination I need a way to objectively compare
# how well the model performs. I decided to score on two things:
#
#   1. Silhouette Score - measures how well-separated the regime clusters are.
#      A score close to 1 means the regimes are clearly distinct from each
#      other. A score near 0 means they overlap badly. Range is -1 to +1.
#
#   2. Sharpe Ratio - measures the risk-adjusted return of the trading strategy
#      that the regimes produce. Higher is better. A Sharpe above 1.0 is
#      generally considered good in finance.
#
#   3. Combined Score - I combine both into a single number (weighted equally)
#      so I can rank all combinations with one metric. This way the winner
#      has to be good at both cluster quality AND trading performance,
#      not just one of them.
# -----------------------------------------------------------------------------
def score_model(rd: RegimeDetector, split_date: str = SPLIT_DATE) -> dict:
    # Fit on training sample only, evaluate on test sample only
    try:
        rd.fit_train_test(split_date=split_date)
    except Exception:
        return None

    from sklearn.metrics import silhouette_score as _sil

    # Silhouette should be computed on the TRAIN representation and TRAIN labels,
    # because clustering is fitted on the training sample.
    try:
        if rd.features_reduced is not None and rd.train_labels is not None:
            sil = _sil(rd.features_reduced, rd.train_labels.values)
        else:
            sil = np.nan
    except Exception:
        sil = np.nan

    # Calculate annualised Sharpe ratio from TEST-ONLY backtest returns
    try:
        bt = rd.backtest_test_only()
        strategy_daily = bt["strategy_ret"]
        if strategy_daily.std() > 0:
            sharpe = (strategy_daily.mean() * 252) / (strategy_daily.std() * np.sqrt(252))
        else:
            sharpe = 0.0
    except Exception:
        sharpe = np.nan

    # Combine the two scores
    sharpe_component = 0.0 if pd.isna(sharpe) else sharpe / 3.0
    sil_component = 0.0 if pd.isna(sil) else sil
    combined = 0.5 * sil_component + 0.5 * sharpe_component

    return {
        "silhouette": round(sil, 4) if not pd.isna(sil) else np.nan,
        "sharpe":     round(sharpe, 4) if not pd.isna(sharpe) else np.nan,
        "combined":   round(combined, 4),
    }

# -----------------------------------------------------------------------------
# GRID SEARCH
# -----------------------------------------------------------------------------
# itertools.product generates every possible combination from the grid above.
# For example if n_regimes=[2,3] and use_pca=[True,False], it produces:
#   (2, True), (2, False), (3, True), (3, False)
# We do this across all 5 hyperparameters at once.
# -----------------------------------------------------------------------------

keys   = list(PARAM_GRID.keys())
values = list(PARAM_GRID.values())
all_combinations = list(itertools.product(*values))

print(f"Testing {len(all_combinations)} hyperparameter combinations ...\n")

# Store each result as a row so we can rank them all at the end
results = []

for i, combo in enumerate(all_combinations, start=1):
    # Convert the tuple back into a named dictionary for readability
    params = dict(zip(keys, combo))

    # Skip duplicate runs — when PCA is off, pca_variance has no effect
    # so we only need to test it once rather than twice
    if not params["use_pca"] and params["pca_variance"] != PARAM_GRID["pca_variance"][0]:
        continue

    print(f"[{i}/{len(all_combinations)}]  Testing: {params}", end="  ->  ", flush=True)

    # Build a fresh RegimeDetector with this combination of hyperparameters.
    # random_state=42 keeps results reproducible — same seed every time
    # means K-Means starts from the same random initialisation each run.
    rd = RegimeDetector(
        close,
        n_regimes      = params["n_regimes"],
        use_pca        = params["use_pca"],
        pca_variance   = params["pca_variance"],
        return_periods = params["return_periods"],
        vol_windows    = params["vol_windows"],
        random_state   = 42,
    )

    scores = score_model(rd)

    # If the model failed, skip and move on to the next combination
    if scores is None:
        print("FAILED (skipped)")
        continue

    print(f"silhouette={scores['silhouette']:.3f}  sharpe={scores['sharpe']:.3f}  combined={scores['combined']:.3f}")

    # Save all the details for this run so we can compare everything at the end
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


# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------
# Sort all combinations by combined score (highest first) and print the table.
# Then re-fit the best model to save its full regime report as a PNG.
# -----------------------------------------------------------------------------

if not results:
    print("\nNo combinations succeeded. Check your data_loader or data.")
else:
    # Sort best to worst by combined score
    df = pd.DataFrame(results).sort_values("combined_score", ascending=False)

    print("\n" + "=" * 70)
    print("  TUNING COMPLETE - All results (sorted best to worst)")
    print("=" * 70)
    print(df.to_string(index=False))

    # Save the full results table to CSV so the team can review it
    df.to_csv(f"tuning_results_{REGION}.csv", index=False)
    print(f"\nFull results saved to tuning_results_{REGION}.csv")

    # The top row is our winner
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

    # Re-fit the best model so we can generate and save its full regime plot.
    # This gives us a visual we can include in the report to justify our choices.
    print("\nRe-fitting best model and saving plot ...")
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

    best_rd.fit_train_test(split_date=SPLIT_DATE)

    print("\n-- Best Model Regime Summary (full-sample descriptive labels) --")
    print(best_rd.summary().to_string(index=False))

    bt = best_rd.backtest_test_only()
    print("\n-- Best Model Test Backtest (tail) --")
    print(bt.tail(10).to_string())

    fig = best_rd.plot_backtest(bt=bt)
    fig.savefig(f"best_regime_backtest_{REGION}.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to best_regime_backtest_{REGION}.png")

    print("\nDone! Copy the BEST HYPERPARAMETERS above into your RegimeDetector() call.")