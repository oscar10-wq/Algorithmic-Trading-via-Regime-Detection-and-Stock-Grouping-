import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.src.data_loader import get_close_prices
from algo_regime.src.regime_detector import RegimeDetector


START_DATE = "2010-01-01"
SPLIT_DATE = "2020-01-01"
END_DATE = "2024-01-01"

REGIONS = ["america", "europe", "asia"]

# PCA + KMeans final parameters
PCA_PARAMS = {
    "america": {
        "n_regimes": 2,
        "use_pca": True,
        "pca_variance": 0.9,
        "return_periods": [1, 5, 21],
        "vol_windows": [10, 21, 63],
    },
    "europe": {
        "n_regimes": 4,
        "use_pca": True,
        "pca_variance": 0.9,
        "return_periods": [1, 5, 21],
        "vol_windows": [21, 63],
    },
    "asia": {
        "n_regimes": 3,
        "use_pca": True,
        "pca_variance": 0.9,
        "return_periods": [1, 5, 21],
        "vol_windows": [10, 21, 63],
    },
}

# SWKMeans parameters
SWK_PARAMS = {
    "america": {
        "n_regimes": 2,
        "N_S": 5,
        "L": 100,
        "h1": 50,
        "h2": 10,
        "epsilon": 1e-6,
        "metric": "CVaR",
    },
    "europe": {
        "n_regimes": 2,
        "N_S": 5,
        "L": 100,
        "h1": 50,
        "h2": 10,
        "epsilon": 1e-6,
        "metric": "CVaR",
    },
    "asia": {
        "n_regimes": 2,
        "N_S": 5,
        "L": 100,
        "h1": 50,
        "h2": 10,
        "epsilon": 1e-6,
        "metric": "CVaR",
    },
}

OUTPUT_DIR = "final_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def summarize_backtest(bt: pd.DataFrame, strategy_col: str, initial_capital: float = 100.0):
    ret = bt[strategy_col].dropna()
    cum_col = "cum_strategy" if strategy_col == "strategy_ret" else "cum_strategy_skmeans"
    cum = bt[cum_col].dropna()

    basket_ret = bt["basket_ret"].dropna()
    basket_cum = bt["cum_basket"].dropna()

    out = {
        "final_capital_basket": basket_cum.iloc[-1] if len(basket_cum) else None,
        "final_capital_strategy": cum.iloc[-1] if len(cum) else None,
        "ann_return_basket": (basket_cum.iloc[-1] / initial_capital) ** (252 / len(basket_cum)) - 1 if len(basket_cum) else None,
        "ann_return_strategy": (cum.iloc[-1] / initial_capital) ** (252 / len(cum)) - 1 if len(cum) else None,
        "ann_vol_basket": basket_ret.std() * (252 ** 0.5) if len(basket_ret) else None,
        "ann_vol_strategy": ret.std() * (252 ** 0.5) if len(ret) else None,
        "sharpe_basket": (basket_ret.mean() / basket_ret.std() * (252 ** 0.5)) if basket_ret.std() > 0 else None,
        "sharpe_strategy": (ret.mean() / ret.std() * (252 ** 0.5)) if ret.std() > 0 else None,
        "n_obs": len(bt),
    }
    return out


def save_backtest_plot(bt: pd.DataFrame, strategy_cum_col: str, title: str, filepath: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bt.index, bt["cum_basket"], label="Buy & Hold")
    ax.plot(bt.index, bt[strategy_cum_col], label="Strategy")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


all_results = []

for region in REGIONS:
    print(f"\n===== Running region: {region} =====")

    close = get_close_prices(START_DATE, END_DATE, regions=[region])

    # -------------------------
    # PCA + KMeans
    # -------------------------
    p = PCA_PARAMS[region]

    rd = RegimeDetector(
        close=close,
        n_regimes=p["n_regimes"],
        use_pca=p["use_pca"],
        pca_variance=p["pca_variance"],
        return_periods=p["return_periods"],
        vol_windows=p["vol_windows"],
        random_state=42,
    )

    rd.fit_train_test(split_date=SPLIT_DATE)

    pca_summary = rd.summary()
    pca_summary.to_csv(os.path.join(OUTPUT_DIR, f"pca_kmeans_summary_{region}.csv"), index=False)

    bt_pca = rd.backtest_test_only()
    pca_metrics = summarize_backtest(bt_pca, strategy_col="strategy_ret")
    pca_metrics_df = pd.DataFrame([pca_metrics])
    pca_metrics_df.to_csv(os.path.join(OUTPUT_DIR, f"pca_kmeans_metrics_{region}.csv"), index=False)

    fig1 = rd.plot_regime_profile()
    fig1.savefig(os.path.join(OUTPUT_DIR, f"pca_kmeans_regimes_{region}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    save_backtest_plot(
        bt_pca,
        strategy_cum_col="cum_strategy",
        title=f"PCA + KMeans Test Backtest ({region})",
        filepath=os.path.join(OUTPUT_DIR, f"pca_kmeans_backtest_{region}.png"),
    )

    all_results.append({
        "region": region,
        "model": "PCA_KMeans",
        **pca_metrics,
    })

    # -------------------------
    # SWKMeans
    # -------------------------
    s = SWK_PARAMS[region]

    rd_sw = RegimeDetector(
        close=close,
        n_regimes=s["n_regimes"],
        use_pca=True,
        pca_variance=0.95,
        random_state=42,
    )

    # keep PCA model available too if you want same object structure
    rd_sw.fit_train_test(split_date=SPLIT_DATE)

    rd_sw.detect_regime_skmeans_train_test(
        split_date=SPLIT_DATE,
        N_S=s["N_S"],
        L=s["L"],
        h1=s["h1"],
        h2=s["h2"],
        epsilon=s["epsilon"],
        metric=s["metric"],
    )

    _, signals_sk = rd_sw.generate_signals()
    if signals_sk is not None:
        signals_sk.to_csv(os.path.join(OUTPUT_DIR, f"swkmeans_signals_{region}.csv"))

    bt_sw = rd_sw.backtest_skmeans_test_only()
    sw_metrics = summarize_backtest(bt_sw, strategy_col="strategy_ret_skmeans")
    sw_metrics_df = pd.DataFrame([sw_metrics])
    sw_metrics_df.to_csv(os.path.join(OUTPUT_DIR, f"swkmeans_metrics_{region}.csv"), index=False)

    save_backtest_plot(
        bt_sw,
        strategy_cum_col="cum_strategy_skmeans",
        title=f"SWKMeans Test Backtest ({region})",
        filepath=os.path.join(OUTPUT_DIR, f"swkmeans_backtest_{region}.png"),
    )

    all_results.append({
        "region": region,
        "model": "SWKMeans",
        **sw_metrics,
    })

comparison_df = pd.DataFrame(all_results)
comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_all_regions.csv"), index=False)

print("\nSaved all outputs to:", OUTPUT_DIR)
print(comparison_df)