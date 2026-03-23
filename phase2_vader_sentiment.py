"""
Phase 2: VADER Sentiment Feature Engineering
---------------------------------------------
Loads the Combined News DJIA dataset, cleans headlines,
and computes a daily average VADER compound sentiment score.
"""

import re
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)


def clean_text(text: str) -> str:
    """Lowercase, strip URLs, punctuation, and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_vader_sentiment(news_csv_path: str) -> pd.DataFrame:
    """
    Load news CSV, clean each headline column, and compute
    average daily VADER compound score.

    Returns a DataFrame with columns: ['Date', 'Sentiment']
    """
    news_df = pd.read_csv(news_csv_path)
    headline_cols = [col for col in news_df.columns if col.startswith("Top")]

    # Clean each headline
    for col in headline_cols:
        news_df[col] = news_df[col].astype(str).apply(clean_text)

    sia = SentimentIntensityAnalyzer()

    def headline_sentiment(text: str) -> float:
        return sia.polarity_scores(text)["compound"]

    # Average sentiment across all 25 headlines per day
    news_df["Sentiment"] = news_df[headline_cols].apply(
        lambda row: np.mean(
            [headline_sentiment(t) for t in row if t.strip() != ""]
        ),
        axis=1,
    )

    news_df = news_df[["Date", "Sentiment"]].copy()
    news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.normalize()

    print("Sentiment statistics:")
    print(news_df["Sentiment"].describe())
    return news_df


if __name__ == "__main__":
    sentiment_df = compute_vader_sentiment("data/Combined_News_DJIA.csv")
    print(sentiment_df.head())
