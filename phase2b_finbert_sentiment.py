"""
Phase 2B: FinBERT Sentiment Feature Engineering
------------------------------------------------
Uses the yiyanghkust/finbert-tone model to compute
finance-domain sentiment scores per day from news headlines.

Note: Requires a GPU for reasonable runtime.
      Install with: pip install transformers torch
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "yiyanghkust/finbert-tone"


def load_finbert():
    """Load FinBERT tokenizer and model in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def decode_bytes(text) -> str:
    """Decode byte-string literals common in the DJIA news dataset."""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="ignore")
    return str(text)


def clean_text_finbert(text: str) -> str:
    """Light cleaning suitable for FinBERT (preserves casing & punctuation)."""
    if not isinstance(text, str):
        return ""
    return text.replace("\n", " ").strip()


def finbert_sentiment_score(text: str, tokenizer, model) -> float:
    """
    Score a single text snippet with FinBERT.
    Returns: positive_prob - negative_prob  (range roughly -1 to +1)
    Label order from yiyanghkust/finbert-tone: [negative, neutral, positive]
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    return float(probs[2] - probs[0])  # positive - negative


def compute_finbert_sentiment(news_csv_path: str) -> pd.DataFrame:
    """
    Load news CSV and compute average daily FinBERT sentiment score.

    Returns a DataFrame with columns: ['Date', 'Sentiment_FinBERT']
    """
    news_df = pd.read_csv(news_csv_path)
    headline_cols = [col for col in news_df.columns if col.startswith("Top")]

    news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.normalize()

    # Decode byte strings
    for col in headline_cols:
        news_df[col] = news_df[col].apply(decode_bytes).apply(clean_text_finbert)

    tokenizer, model = load_finbert()

    def daily_score(row):
        scores = [
            finbert_sentiment_score(t, tokenizer, model)
            for t in row
            if isinstance(t, str) and t.strip()
        ]
        return np.mean(scores) if scores else np.nan

    print("Computing FinBERT scores (this may take several minutes)...")
    news_df["Sentiment_FinBERT"] = news_df[headline_cols].apply(daily_score, axis=1)

    result = news_df[["Date", "Sentiment_FinBERT"]].copy()
    print(result["Sentiment_FinBERT"].describe())
    return result


if __name__ == "__main__":
    finbert_df = compute_finbert_sentiment("data/Combined_News_DJIA.csv")
    print(finbert_df.head())
