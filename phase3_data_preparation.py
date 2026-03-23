"""
Phase 3: Data Merging & LSTM Sequence Preparation
--------------------------------------------------
Merges price/regime data with sentiment scores, scales features
with no data leakage, and creates sliding-window sequences for
two models:
  Model A — Price only
  Model B — Price + Sentiment
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WINDOW = 60
SPLIT_RATIO = 0.8


def merge_datasets(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join price/regime DataFrame with sentiment DataFrame on Date.
    Both DataFrames must have a normalized 'Date' column.
    """
    price_df = price_df.copy()
    sentiment_df = sentiment_df.copy()
    price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.normalize()
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"]).dt.normalize()

    final_df = pd.merge(price_df, sentiment_df[["Date", "Sentiment"]], on="Date", how="inner")
    final_df = final_df.sort_values("Date").reset_index(drop=True)

    assert final_df.isna().sum().sum() == 0, "NaN values found after merge!"
    print(f"Merged dataset shape: {final_df.shape}")
    print(final_df.head())
    return final_df


def train_test_split_temporal(df: pd.DataFrame, split_ratio: float = SPLIT_RATIO):
    """Time-based (no shuffle) train/test split."""
    idx = int(len(df) * split_ratio)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()


def scale_features(train_df, test_df):
    """
    Fit scalers on training data only to prevent data leakage.

    Returns:
        train_price, test_price   — scaled arrays for Model A
        train_ps, test_ps         — scaled arrays for Model B
        price_scaler, ps_scaler   — fitted scalers for inverse transform
    """
    price_scaler = MinMaxScaler()
    train_price = price_scaler.fit_transform(train_df[["Close"]])
    test_price = price_scaler.transform(test_df[["Close"]])

    ps_scaler = MinMaxScaler()
    train_ps = ps_scaler.fit_transform(train_df[["Close", "Sentiment"]])
    test_ps = ps_scaler.transform(test_df[["Close", "Sentiment"]])

    return train_price, test_price, train_ps, test_ps, price_scaler, ps_scaler


def create_sequences(data: np.ndarray, window: int = WINDOW):
    """
    Build overlapping sliding-window sequences.

    Returns:
        X — shape (n_samples, window, n_features)
        y — shape (n_samples,)  [next Close price, scaled]
    """
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def prepare_all(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, window: int = WINDOW):
    """
    End-to-end Phase 3 pipeline.

    Returns a dict with all arrays and metadata needed for training.
    """
    final_df = merge_datasets(price_df, sentiment_df)
    train_df, test_df = train_test_split_temporal(final_df)
    print(f"Train: {train_df.shape}  |  Test: {test_df.shape}")

    train_price, test_price, train_ps, test_ps, price_scaler, ps_scaler = scale_features(
        train_df, test_df
    )

    X_train_A, y_train_A = create_sequences(train_price, window)
    X_test_A,  y_test_A  = create_sequences(test_price,  window)

    X_train_B, y_train_B = create_sequences(train_ps, window)
    X_test_B,  y_test_B  = create_sequences(test_ps,  window)

    # Regime labels aligned with the test sequences
    regime_test = test_df["Regime"].iloc[window:].values

    print(f"Model A — X_train: {X_train_A.shape}  X_test: {X_test_A.shape}")
    print(f"Model B — X_train: {X_train_B.shape}  X_test: {X_test_B.shape}")

    return {
        "final_df": final_df,
        "X_train_A": X_train_A, "y_train_A": y_train_A,
        "X_test_A":  X_test_A,  "y_test_A":  y_test_A,
        "X_train_B": X_train_B, "y_train_B": y_train_B,
        "X_test_B":  X_test_B,  "y_test_B":  y_test_B,
        "price_scaler": price_scaler,
        "ps_scaler":    ps_scaler,
        "regime_test":  regime_test,
    }
