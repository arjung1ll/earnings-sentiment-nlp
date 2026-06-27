import os
import pandas as pd
import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from datetime import timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Earnings call dates ───────────────────────────────────────────────────────
# Date = the day the earnings call was held

EARNINGS_DATES = {
    "Q123":  "2023-04-27",
    "Q223":  "2023-07-27",
    "Q323":  "2023-10-25",
    "Q424":  "2023-02-13",  # FY2023 / Q423
    "Q124":  "2024-04-30",
    "Q224":  "2024-07-31",
    "Q324":  "2024-10-23",
    "Q425":  "2025-02-13",  # FY2024 / Q424
    "Q125":  "2025-04-29",
    "Q225":  "2025-07-30",
    "Q325":  "2025-10-22",
    "Q426":  "2026-02-13",  # FY2025 / Q425
    "Q126":  "2026-04-28",
}

TICKER = "BARC.L"
DATA_DIR = os.path.join("data", "raw")

# ── Load FinBERT ──────────────────────────────────────────────────────────────

print("Loading FinBERT...")
model_name = "ProsusAI/finbert"
tokenizer  = BertTokenizer.from_pretrained(model_name)
model      = BertForSequenceClassification.from_pretrained(model_name)
nlp        = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("FinBERT ready\n")

# ── Helper functions ──────────────────────────────────────────────────────────

def load_transcript(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, max_words=400):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def score_transcript(text):
    chunks  = chunk_text(text)
    scores  = {"positive": 0, "negative": 0, "neutral": 0}
    for chunk in chunks:
        result = nlp(chunk[:512])
        scores[result[0]["label"]] += result[0]["score"]
    total = sum(scores.values())
    return {k: round(v / total, 4) for k, v in scores.items()}

def get_returns(ticker, earnings_date, windows=[1, 3, 5]):
    date  = pd.to_datetime(earnings_date)
    start = (date - timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (date + timedelta(days=10)).strftime("%Y-%m-%d")
    stock = yf.download(ticker, start=start, end=end, progress=False)
    if stock.empty:
        return {}
    close = stock["Close"].dropna()
    days  = close.index.tolist()
    idx   = min(range(len(days)), key=lambda i: abs((days[i] - date).days))
    base  = float(close.iloc[idx].iloc[0])
    returns = {}
    for w in windows:
        if idx + w < len(close):
            future = float(close.iloc[idx + w].iloc[0])
            returns[f"return_{w}d"] = round((future - base) / base * 100, 4)
    return returns

# ── Main pipeline ─────────────────────────────────────────────────────────────

results = []

for quarter, date in tqdm(EARNINGS_DATES.items()):
    speech_file = os.path.join(DATA_DIR, f"Barclays_{quarter}_EarningsCall_Speech.txt")
    qa_file     = os.path.join(DATA_DIR, f"Barclays_{quarter}_EarningsCall_QA.txt")

    if not os.path.exists(speech_file):
        print(f"Missing: {speech_file}")
        continue

    print(f"\nProcessing {quarter} ({date})...")

    # Score speech
    speech_scores = score_transcript(load_transcript(speech_file))

    # Score Q&A if available
    qa_scores = {}
    if os.path.exists(qa_file):
        qa_scores = score_transcript(load_transcript(qa_file))

    # Get stock returns
    returns = get_returns(TICKER, date)

    results.append({
        "quarter":             quarter,
        "date":                date,
        "speech_positive":     speech_scores.get("positive", None),
        "speech_negative":     speech_scores.get("negative", None),
        "speech_neutral":      speech_scores.get("neutral",  None),
        "qa_positive":         qa_scores.get("positive",     None),
        "qa_negative":         qa_scores.get("negative",     None),
        "qa_neutral":          qa_scores.get("neutral",      None),
        **returns
    })

df = pd.DataFrame(results)
print("\n\nResults:")
print(df.to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/all_quarters_sentiment_returns.csv", index=False)
print("\nSaved to data/processed/all_quarters_sentiment_returns.csv")

# ── Plot: Sentiment vs 5-day return ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df["speech_positive"], df["return_5d"],
           color="#2c7be5", s=80, zorder=5)

for _, row in df.iterrows():
    ax.annotate(row["quarter"],
                (row["speech_positive"], row["return_5d"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("Speech Positive Sentiment Score", fontsize=12)
ax.set_ylabel("5-Day Stock Return (%)", fontsize=12)
ax.set_title("Barclays Earnings Call Sentiment vs 5-Day Stock Return\n(Q1 2023 to Q1 2026)",
             fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/sentiment_vs_returns.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to outputs/sentiment_vs_returns.png")