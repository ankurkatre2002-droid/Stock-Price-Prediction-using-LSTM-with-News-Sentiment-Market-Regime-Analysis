"""
Phase 4 & 5: LSTM Model Training & Regime-Wise Evaluation
----------------------------------------------------------
Builds, trains, and evaluates two LSTM models:
  Model A — Price only
  Model B — Price + VADER Sentiment

Evaluation covers:
  - Overall RMSE
  - Directional Accuracy
  - Regime-wise breakdown (low vs high volatility)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 5


# ─── Model Definition ────────────────────────────────────────────────────────

def build_lstm_model(input_shape: tuple):
    """
    Two-layer stacked LSTM with dropout.
    input_shape: (window_size, n_features)
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, label="Model"):
    """Fit model with early stopping; returns training history."""
    early_stop = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    print(f"{label} training complete.")
    return history


# ─── Inverse Scaling ─────────────────────────────────────────────────────────

def inverse_price_only(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def inverse_price_sentiment(y_scaled, scaler):
    """Standard trick: pad a dummy sentiment column before inverse transform."""
    dummy = np.zeros((len(y_scaled), 1))
    return scaler.inverse_transform(
        np.hstack([y_scaled.reshape(-1, 1), dummy])
    )[:, 0]


# ─── Metrics ─────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_models(data: dict):
    """
    Full evaluation pipeline.
    `data` is the dict returned by phase3_data_preparation.prepare_all().
    """
    # Predictions
    y_pred_A = data["model_A"].predict(data["X_test_A"]).flatten()
    y_pred_B = data["model_B"].predict(data["X_test_B"]).flatten()

    # Inverse transform
    y_true_A = inverse_price_only(data["y_test_A"], data["price_scaler"])
    y_pred_A = inverse_price_only(y_pred_A,         data["price_scaler"])

    y_true_B = inverse_price_sentiment(data["y_test_B"], data["ps_scaler"])
    y_pred_B = inverse_price_sentiment(y_pred_B,         data["ps_scaler"])

    # ── Overall ──
    print("=" * 50)
    print("OVERALL PERFORMANCE")
    print(f"  Model A (Price-only)     RMSE : {rmse(y_true_A, y_pred_A):.4f}")
    print(f"  Model B (Price+Sentiment) RMSE: {rmse(y_true_B, y_pred_B):.4f}")
    print(f"  Model A Directional Accuracy  : {directional_accuracy(y_true_A, y_pred_A):.3f}")
    print(f"  Model B Directional Accuracy  : {directional_accuracy(y_true_B, y_pred_B):.3f}")

    # ── Regime-wise ──
    regime = data["regime_test"]
    low  = regime == 0
    high = regime == 1
    print(f"\nLow-vol samples : {low.sum()} | High-vol samples: {high.sum()}")

    print("\nLOW VOLATILITY REGIME")
    print(f"  Model A RMSE : {rmse(y_true_A[low], y_pred_A[low]):.2f}")
    print(f"  Model B RMSE : {rmse(y_true_B[low], y_pred_B[low]):.2f}")
    print(f"  Model A DA   : {directional_accuracy(y_true_A[low], y_pred_A[low]):.3f}")
    print(f"  Model B DA   : {directional_accuracy(y_true_B[low], y_pred_B[low]):.3f}")

    print("\nHIGH VOLATILITY REGIME")
    print(f"  Model A RMSE : {rmse(y_true_A[high], y_pred_A[high]):.2f}")
    print(f"  Model B RMSE : {rmse(y_true_B[high], y_pred_B[high]):.2f}")
    print(f"  Model A DA   : {directional_accuracy(y_true_A[high], y_pred_A[high]):.3f}")
    print(f"  Model B DA   : {directional_accuracy(y_true_B[high], y_pred_B[high]):.3f}")
    print("=" * 50)

    return y_true_A, y_pred_A, y_true_B, y_pred_B


def plot_predictions(y_true_A, y_pred_A, y_true_B, y_pred_B,
                     save_path: str = None):
    """Side-by-side prediction plots for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, y_true, y_pred, title in zip(
        axes,
        [y_true_A, y_true_B],
        [y_pred_A, y_pred_B],
        ["Model A: Price Only", "Model B: Price + Sentiment"],
    ):
        ax.plot(y_true, label="Actual",    linewidth=1.2)
        ax.plot(y_pred, label="Predicted", linewidth=1.0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Test Timestep")
        ax.set_ylabel("DJIA Close Price")
        ax.legend()

    plt.suptitle("DJIA LSTM Forecast: Model A vs Model B", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
