"""
sentiment.py
------------
Runs FinBERT sentiment scoring on earnings call transcripts.
FinBERT is a BERT model fine-tuned on financial text — much more
accurate than general-purpose sentiment models for this use case.

Usage:
    from src.sentiment import score_transcript
    result = score_transcript("Revenue grew strongly this quarter...")
"""

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd
from tqdm import tqdm


def load_finbert():
    """Load FinBERT model from HuggingFace."""
    model_name = "ProsusAI/finbert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp


def chunk_text(text, max_words=400):
    """
    FinBERT has a 512 token limit. We split long transcripts
    into chunks and average the sentiment scores.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


def score_transcript(text, nlp=None):
    """
    Score a single transcript. Returns:
    - positive_score: float 0-1
    - negative_score: float 0-1
    - neutral_score: float 0-1
    - label: dominant sentiment
    """
    if nlp is None:
        nlp = load_finbert()

    chunks = chunk_text(text)
    results = nlp(chunks)

    scores = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        scores[r["label"]] += r["score"]

    total = sum(scores.values())
    for k in scores:
        scores[k] /= total

    scores["label"] = max(scores, key=lambda x: scores[x] if x != "label" else -1)
    return scores


def score_all_transcripts(transcripts_df, nlp=None):
    """
    Score a DataFrame of transcripts.
    Expects columns: company, date, transcript_text
    Returns DataFrame with sentiment scores added.
    """
    if nlp is None:
        nlp = load_finbert()

    results = []
    for _, row in tqdm(transcripts_df.iterrows(), total=len(transcripts_df)):
        scores = score_transcript(row["transcript_text"], nlp)
        results.append({
            "company": row["company"],
            "date": row["date"],
            "positive": round(scores["positive"], 4),
            "negative": round(scores["negative"], 4),
            "neutral": round(scores["neutral"], 4),
            "label": scores["label"]
        })

    return pd.DataFrame(results)
