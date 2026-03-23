"""
main.py — End-to-End Pipeline Runner
=====================================
Executes all 5 phases in sequence:
  Phase 1  → Load DJIA price data, detect market regimes
  Phase 2  → VADER sentiment extraction from news headlines
  Phase 3  → Merge datasets, scale features, build LSTM sequences
  Phase 4  → Train Model A (Price) and Model B (Price + Sentiment)
  Phase 5  → Regime-wise evaluation and result plots

Usage:
    python main.py

Dataset files expected in ./data/:
    - Dow Jones Industrial Average Historical Data.csv
    - Combined_News_DJIA.csv
"""

import os
from src.phase1_regime_detection import run_phase1
from src.phase2_vader_sentiment import compute_vader_sentiment
from src.phase3_data_preparation import prepare_all
from src.phase4_5_model_training_evaluation import (
    build_lstm_model,
    train_model,
    evaluate_models,
    plot_predictions,
)

PRICE_CSV   = "data/Dow Jones Industrial Average Historical Data.csv"
NEWS_CSV    = "data/Combined_News_DJIA.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Price Data & Regime Detection")
    print("=" * 60)
    price_df = run_phase1(PRICE_CSV)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: VADER Sentiment Extraction")
    print("=" * 60)
    sentiment_df = compute_vader_sentiment(NEWS_CSV)

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Merge & Sequence Preparation")
    print("=" * 60)
    data = prepare_all(price_df, sentiment_df)

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: LSTM Model Training")
    print("=" * 60)

    model_A = build_lstm_model(
        input_shape=(data["X_train_A"].shape[1], data["X_train_A"].shape[2])
    )
    train_model(model_A, data["X_train_A"], data["y_train_A"], label="Model A (Price-only)")

    model_B = build_lstm_model(
        input_shape=(data["X_train_B"].shape[1], data["X_train_B"].shape[2])
    )
    train_model(model_B, data["X_train_B"], data["y_train_B"], label="Model B (Price + Sentiment)")

    data["model_A"] = model_A
    data["model_B"] = model_B

    # ── Phase 5 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 5: Regime-Wise Evaluation")
    print("=" * 60)
    y_true_A, y_pred_A, y_true_B, y_pred_B = evaluate_models(data)

    plot_predictions(
        y_true_A, y_pred_A, y_true_B, y_pred_B,
        save_path=f"{RESULTS_DIR}/prediction_comparison.png",
    )
    print(f"\nPlot saved to {RESULTS_DIR}/prediction_comparison.png")


if __name__ == "__main__":
    main()
