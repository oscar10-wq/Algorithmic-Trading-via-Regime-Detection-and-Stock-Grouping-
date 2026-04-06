import pandas as pd
from data.src.data_loader import get_close_prices
from algo_regime.src import sWkmean as ws

START_DATE = "2010-01-01"
SPLIT_DATE = "2020-01-01"
END_DATE = "2024-01-01"
REGION = "america"

close = get_close_prices(START_DATE, END_DATE, regions=[REGION])

train_close = close.loc[close.index < SPLIT_DATE].copy()

warmup_bars = 50
split_ts = pd.Timestamp(SPLIT_DATE)

# first trading day on or after split date
start_loc = close.index.searchsorted(split_ts)
ext_start_loc = max(0, start_loc - warmup_bars)
test_close_ext = close.iloc[ext_start_loc:].copy()

result = ws.fit_predict_swkmeans_train_test(
    S_train=train_close.values,
    S_test_with_warmup=test_close_ext.values,
    train_price_index=train_close.index,
    test_price_index_with_warmup=test_close_ext.index,
    split_date=SPLIT_DATE,
    K=2,
    L=100,
    epsilon=1e-6,
    h1=50,
    h2=10,
    N_S=5,
    metric="CVaR",
    random_state=42,
)

print("Train labels shape:", result["labels_train"].shape)
print("Test labels shape:", result["labels_test"].shape)
print("Train index length:", len(result["train_index"]))
print("Test index length:", len(result["test_index"]))
print("Unique train labels:", sorted(set(result["labels_train"])))
print("Unique test labels:", sorted(set(result["labels_test"])))
print("Requested split date:", split_ts)
print("Actual first test trading day:", close.index[start_loc])

from data.src.data_loader import get_close_prices
from algo_regime.src.regime_detector import RegimeDetector

close = get_close_prices("2010-01-01", "2024-01-01", regions=["america"])

rd = RegimeDetector(
    close,
    n_regimes=2,
    use_pca=True,
    pca_variance=0.95,
    random_state=42,
)

rd.fit_train_test(split_date="2020-01-01")

rd.detect_regime_skmeans_train_test(
    split_date="2020-01-01",
    N_S=5,
    L=100,
    h1=50,
    h2=10,
    epsilon=1e-6,
    metric="CVaR",
)

print(rd.train_labels_skmeans.head())
print(rd.test_labels_skmeans.head())
print(rd.labels_skmeans.tail())

signals, signals_skmeans = rd.generate_signals()

print("\nMain signals tail:")
print(signals.tail())

print("\nSWKMeans signals tail:")
print(signals_skmeans.tail())

bt_sk = rd.backtest_skmeans_test_only()

print("\nSWKMeans test-only backtest tail:")
print(bt_sk.tail())

print("\nMinimum close values:")
print(close.min())

print("\nAny non-positive prices?")
print((close <= 0).sum())
print("\nMissing values in close:")
print(close.isna().sum())