import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta

EARNINGS_DATES = {
    "Q123":  "2023-04-27",
    "Q223":  "2023-07-27",
    "Q323":  "2023-10-25",
    "Q424":  "2023-02-13",
    "Q124":  "2024-04-30",
    "Q224":  "2024-07-31",
    "Q324":  "2024-10-23",
    "Q425":  "2025-02-13",
    "Q125":  "2025-04-29",
    "Q225":  "2025-07-30",
    "Q325":  "2025-10-22",
    "Q426":  "2026-02-13",
    "Q126":  "2026-04-28",
}

TICKER = "BARC.L"
SENTIMENT_CSV = os.path.join("data", "processed", "all_quarters_sentiment_returns.csv")

# ── Load sentiment data ───────────────────────────────────────────────────────

df = pd.read_csv(SENTIMENT_CSV)
df = df.set_index("quarter")

# ── Helper: get price data ────────────────────────────────────────────────────

def get_price_data(ticker, earnings_date):
    date  = pd.to_datetime(earnings_date)
    start = (date - timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (date + timedelta(days=10)).strftime("%Y-%m-%d")
    stock = yf.download(ticker, start=start, end=end, progress=False)
    if stock.empty:
        return None, None
    close = stock["Close"].dropna()
    days  = close.index.tolist()
    idx   = min(range(len(days)), key=lambda i: abs((days[i] - date).days))
    base  = float(close.iloc[idx].iloc[0])
    returns = {}
    for w in [1, 3, 5]:
        if idx + w < len(close):
            future = float(close.iloc[idx + w].iloc[0])
            returns[w] = round((future - base) / base * 100, 2)
    return close, returns

# ── Generate one chart per quarter ────────────────────────────────────────────

os.makedirs("outputs/quarterly_charts", exist_ok=True)
colours = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}

for quarter, date in EARNINGS_DATES.items():
    print(f"Plotting {quarter}...")

    close, returns = get_price_data(TICKER, date)
    if close is None:
        print(f"  No price data for {quarter}")
        continue

    if quarter not in df.index:
        print(f"  No sentiment data for {quarter}")
        continue

    row = df.loc[quarter]
    ret5        = returns.get(5, 0)
    line_colour = "#2ecc71" if ret5 >= 0 else "#e74c3c"

    # Speech sentiment values
    speech_vals = [
        row["speech_negative"],
        row["speech_neutral"],
        row["speech_positive"]
    ]
    # Q&A sentiment values
    qa_vals = [
        row["qa_negative"],
        row["qa_neutral"],
        row["qa_positive"]
    ]
    sentiments  = ["Negative", "Neutral", "Positive"]
    bar_colours = [colours["negative"], colours["neutral"], colours["positive"]]

    # ── Layout: 1 row, 3 panels ───────────────────────────────────────────────
    fig, (ax_price, ax_speech, ax_qa) = plt.subplots(
        1, 3, figsize=(16, 4.5),
        gridspec_kw={"width_ratios": [3, 1.5, 1.5]}
    )
    fig.suptitle(
        f"Barclays (BARC.L)  —  {quarter} Earnings Call  |  {date}",
        fontsize=13, fontweight="bold", y=1.02
    )

    # ── Panel 1: Price chart ──────────────────────────────────────────────────
    ax_price.plot(close.index, close.values, marker="o", markersize=4,
                  color=line_colour, linewidth=2)
    ax_price.axvline(x=pd.to_datetime(date), color="#e74c3c",
                     linestyle="--", linewidth=1.5, label="Earnings call", zorder=5)
    ax_price.axvspan(pd.to_datetime(date), close.index[-1],
                     alpha=0.06, color=line_colour)

    ret_text = "\n".join([
        f"+1d: {returns.get(1, 0):+.2f}%",
        f"+3d: {returns.get(3, 0):+.2f}%",
        f"+5d: {returns.get(5, 0):+.2f}%",
    ])
    ax_price.text(0.02, 0.97, ret_text, transform=ax_price.transAxes,
                  fontsize=9, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor="lightgrey", alpha=0.9))

    ax_price.set_title("Stock Price", fontsize=10, fontweight="bold")
    ax_price.set_ylabel("Price (pence)")
    ax_price.legend(fontsize=8)
    ax_price.spines[["top", "right"]].set_visible(False)
    ax_price.tick_params(axis="x", labelsize=7, rotation=45)

    # ── Panel 2: Speech sentiment ─────────────────────────────────────────────
    bars = ax_speech.bar(sentiments, [v * 100 for v in speech_vals],
                         color=bar_colours, edgecolor="white", linewidth=1.2)
    ax_speech.set_title("Prepared Speech\n(FinBERT)", fontsize=10, fontweight="bold")
    ax_speech.set_ylabel("% of chunks")
    ax_speech.set_ylim(0, 100)
    ax_speech.spines[["top", "right"]].set_visible(False)
    ax_speech.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars, speech_vals):
        ax_speech.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 1,
                       f"{val*100:.1f}%", ha="center", fontsize=8, fontweight="bold")

    # ── Panel 3: Q&A sentiment ────────────────────────────────────────────────
    bars = ax_qa.bar(sentiments, [v * 100 for v in qa_vals],
                     color=bar_colours, edgecolor="white", linewidth=1.2)
    ax_qa.set_title("Q&A Session\n(FinBERT)", fontsize=10, fontweight="bold")
    ax_qa.set_ylabel("% of chunks")
    ax_qa.set_ylim(0, 100)
    ax_qa.spines[["top", "right"]].set_visible(False)
    ax_qa.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars, qa_vals):
        ax_qa.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 1,
                   f"{val*100:.1f}%", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    path = f"outputs/quarterly_charts/BARC_{quarter}_combined.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

print("\nAll combined charts done.")