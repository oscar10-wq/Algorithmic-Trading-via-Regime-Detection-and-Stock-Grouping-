from pathlib import Path

import pandas as pd

from data.src.data_loader import get_close_prices
from gmmhmm import HMMRegimeDetector


OUTPUT_DIR = Path("final_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_backtest_metrics(bt: pd.DataFrame, region: str) -> pd.DataFrame:
    """Extract one-row summary metrics from backtest output."""
    last = bt.iloc[-1]

    metrics = {
        "region": region,
        "ann_ret_basket": float(last["ann_ret_basket"]),
        "ann_ret_strategy": float(last["ann_ret_strategy"]),
        "ann_vol_basket": float(last["ann_vol_basket"]),
        "ann_vol_strategy": float(last["ann_vol_strategy"]),
        "sharpe_basket": float(last["sharpe_basket"]),
        "sharpe_strategy": float(last["sharpe_strategy"]),
        "calmar_basket": float(last["calmar_basket"]),
        "calmar_strategy": float(last["calmar_strategy"]),
        "final_cum_basket": float(last["cum_basket"]),
        "final_cum_strategy": float(last["cum_strategy"]),
    }
    return pd.DataFrame([metrics])


def run_one_region(region: str) -> pd.DataFrame:
    print(f"\n{'=' * 25} {region.upper()} {'=' * 25}")

    close = get_close_prices(
        start_date="2010-01-01",
        end_date="2024-01-01",
        regions=[region],
        interval="1d",
        fill_method="ffill",
    )

    det = HMMRegimeDetector(
        close=close,
        n_regimes=None,   # 保持你现在已经跑通的设置
        n_mix=2,
        use_pca=True,
        pca_variance=0.95,
        covariance_type="diag",
        n_iter=300,
        random_state=42,
    )

    det.fit_train_test(split_date="2020-01-01")

    # 1) regime summary
    summary_df = det.summary()
    summary_path = OUTPUT_DIR / f"gmmhmm_summary_{region}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    # 2) transition matrix
    tm_df = det.transition_matrix()
    tm_path = OUTPUT_DIR / f"gmmhmm_transition_matrix_{region}.csv"
    tm_df.to_csv(tm_path)
    print(f"Saved {tm_path}")

    # 3) test signals
    signals_test = det.generate_signals(dataset="test")
    signals_path = OUTPUT_DIR / f"gmmhmm_signals_{region}.csv"
    signals_test.to_csv(signals_path)
    print(f"Saved {signals_path}")

    # 4) full daily backtest on test period
    bt_test = det.backtest(dataset="test", signals=signals_test)
    bt_path = OUTPUT_DIR / f"gmmhmm_backtest_{region}.csv"
    bt_test.to_csv(bt_path)
    print(f"Saved {bt_path}")

    # 5) one-row metrics summary
    metrics_df = extract_backtest_metrics(bt_test, region)
    metrics_path = OUTPUT_DIR / f"gmmhmm_metrics_{region}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # 6) plot
    fig = det.plot_regimes()
    fig_path = OUTPUT_DIR / f"gmmhmm_regimes_{region}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved {fig_path}")

    return metrics_df


def main():
    all_metrics = []

    for region in ["america", "europe", "asia"]:
        metrics_df = run_one_region(region)
        all_metrics.append(metrics_df)

    comparison_df = pd.concat(all_metrics, ignore_index=True)
    comparison_path = OUTPUT_DIR / "gmmhmm_model_comparison_all_regions.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved {comparison_path}")

    print("\n=== Combined Comparison ===")
    print(comparison_df)


if __name__ == "__main__":
    main()