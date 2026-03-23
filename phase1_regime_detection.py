"""
Phase 1: Data Loading, Preprocessing & Market Regime Detection
---------------------------------------------------------------
Loads DJIA historical price data, computes log returns and
rolling volatility, then labels market regimes (low/high volatility).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load DJIA CSV, clean columns, and sort by date."""
    df = pd.read_csv(filepath)
    df.rename(columns={"Price": "Close"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = df["Close"].astype(str).str.replace(",", "").astype(float)
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close"]]


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns and drop the first NaN row."""
    df = df.copy()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna().reset_index(drop=True)


def compute_volatility(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Compute rolling standard deviation of log returns as volatility proxy."""
    df = df.copy()
    df["Volatility"] = df["Log_Return"].rolling(window=window).std()
    return df.dropna().reset_index(drop=True)


def label_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label market regimes based on median volatility threshold.
      0 → Low volatility  (calm market)
      1 → High volatility (turbulent market)
    """
    df = df.copy()
    vol_threshold = df["Volatility"].median()
    df["Regime"] = np.where(df["Volatility"] < vol_threshold, 0, 1)
    return df


def plot_regimes(df: pd.DataFrame, save_path: str = None):
    """Plot DJIA closing price with high-volatility periods highlighted."""
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Close"], label="DJIA Close Price", linewidth=0.8)
    high_vol = df[df["Regime"] == 1]
    plt.scatter(high_vol["Date"], high_vol["Close"],
                color="red", s=8, label="High Volatility", alpha=0.6)
    plt.title("DJIA Close Price with High-Volatility Regimes")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (USD)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def run_phase1(filepath: str, volatility_window: int = 30) -> pd.DataFrame:
    """Full Phase 1 pipeline. Returns the processed DataFrame."""
    df = load_and_clean(filepath)
    df = compute_log_returns(df)
    df = compute_volatility(df, window=volatility_window)
    df = label_regimes(df)

    print(f"Final dataset shape: {df.shape}")
    print("\nRegime distribution:")
    print(df["Regime"].value_counts())
    print(df.head())
    return df


if __name__ == "__main__":
    df = run_phase1("data/Dow Jones Industrial Average Historical Data.csv")
    plot_regimes(df, save_path="results/regime_plot.png")
