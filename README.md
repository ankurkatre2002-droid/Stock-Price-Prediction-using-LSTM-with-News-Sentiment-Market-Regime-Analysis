# DJIA Stock Price Forecasting with Sentiment Analysis & Market Regime Detection

> **Research Project** | M.Tech Data Science & AI — VIT Bhopal  
> Predicting Dow Jones Industrial Average (DJIA) closing prices using LSTM networks augmented with VADER/FinBERT news sentiment, evaluated across market volatility regimes.

---

## Research Objective

This project investigates whether incorporating **news sentiment** (VADER & FinBERT) into LSTM-based time-series models improves DJIA forecasting accuracy — and whether that improvement is **regime-dependent** (i.e., more pronounced during high-volatility market periods).

Two models are compared:
| Model | Features |
|-------|----------|
| **Model A** | DJIA closing price only |
| **Model B** | DJIA closing price + daily news sentiment score |

---

## Key Results

| Metric | Model A (Price-only) | Model B (Price + Sentiment) |
|--------|---------------------|----------------------------|
| Overall RMSE | 332.90 | 369.74 |
| Directional Accuracy | 47.9% | 49.5% |
| **Low-Volatility RMSE** | — | — |
| **High-Volatility RMSE** | — | — |

> Sentiment provides a marginal improvement in directional accuracy (+1.6%), with the effect being regime-dependent.

---

## Architecture

```
Phase 1  →  DJIA price data → log returns → rolling volatility → regime labels (0/1)
Phase 2  →  News headlines → VADER cleaning → daily compound sentiment score
Phase 2B →  News headlines → FinBERT (yiyanghkust/finbert-tone) → finance-domain scores
Phase 3  →  Merge datasets → MinMaxScaler (no leakage) → sliding window sequences (60-day)
Phase 4  →  Stacked LSTM (50→50→Dense) × 2 models, EarlyStopping
Phase 5  →  RMSE + Directional Accuracy, broken down by Low/High volatility regime
```

---

## Dataset

| File | Source |
|------|--------|
| `Dow Jones Industrial Average Historical Data.csv` | [Investing.com](https://www.investing.com) |
| `Combined_News_DJIA.csv` | [Kaggle — Daily News for Stock Market Prediction](https://www.kaggle.com/datasets/aaron7sun/stocknews) |

Place both files in the `data/` folder before running.

---

## Project Structure

```
djia-sentiment-lstm/
│
├── data/                          # Raw datasets (not tracked in git)
│   ├── Dow Jones Industrial Average Historical Data.csv
│   └── Combined_News_DJIA.csv
│
├── src/
│   ├── phase1_regime_detection.py      # Price loading, volatility, regime labeling
│   ├── phase2_vader_sentiment.py       # VADER sentiment extraction
│   ├── phase2b_finbert_sentiment.py    # FinBERT sentiment extraction
│   ├── phase3_data_preparation.py      # Merge, scaling, sequence creation
│   └── phase4_5_model_training_evaluation.py  # LSTM training & evaluation
│
├── notebooks/
│   └── ROP_1_VADER_LSTM.ipynb         # Original Colab notebook (reference)
│
├── results/                       # Generated plots and outputs
│   ├── regime_plot.png
│   └── prediction_comparison.png
│
├── main.py                        # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/ankurkatre2002-droid/djia-sentiment-lstm.git
cd djia-sentiment-lstm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add datasets to data/

# 4. Run the full pipeline
python main.py
```

---

## Technical Stack

- **Python 3.10+**
- **TensorFlow / Keras** — LSTM model
- **NLTK VADER** — rule-based sentiment analysis
- **HuggingFace Transformers** — FinBERT (`yiyanghkust/finbert-tone`)
- **scikit-learn** — MinMaxScaler, RMSE
- **Pandas / NumPy / Matplotlib**

---

## LSTM Model

```python
Sequential([
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
# Optimizer: Adam (lr=0.001) | Loss: MSE | EarlyStopping (patience=5)
```

---

## Author

**Ankur Katre** | M.Tech Data Science & AI, VIT Bhopal  
[LinkedIn](https://www.linkedin.com/in/ankur-katre-aa7a41372) · [GitHub](https://github.com/ankurkatre2002-droid)

---
