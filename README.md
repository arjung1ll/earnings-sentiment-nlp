# FTSE 100 Earnings Call Sentiment Analyser

**Can the words a CEO uses predict short-term stock returns?**

This project applies NLP to FTSE 100 earnings call transcripts to measure management sentiment and test whether that sentiment correlates with stock price movements in the 1, 3, and 5 days following each call.

Built as an independent project alongside BSc Economics, Finance & Data Science at Imperial College London.

---

## The idea

Every quarter, FTSE 100 executives host earnings calls with analysts. The *language* they use — confident vs cautious, specific vs vague — carries signals about the company's outlook. This project quantifies that language using FinBERT (a BERT model fine-tuned on financial text) and asks: does sentiment predict returns?

---

## Project structure

```
earnings-sentiment-nlp/
│
├── data/
│   ├── raw/              # Raw transcript text files
│   └── processed/        # Cleaned, tokenised transcripts
│
├── notebooks/
│   ├── 01_data_collection.ipynb      # Scraping & collecting transcripts
│   ├── 02_sentiment_analysis.ipynb   # Running FinBERT on transcripts
│   ├── 03_stock_returns.ipynb        # Pulling price data via yfinance
│   └── 04_correlation_analysis.ipynb # Sentiment vs return analysis
│
├── src/
│   ├── scraper.py        # Transcript collection logic
│   ├── sentiment.py      # FinBERT sentiment scoring
│   ├── returns.py        # Stock return calculations
│   └── visualise.py      # Plotting functions
│
├── outputs/              # Charts and results
├── requirements.txt
└── README.md
```

---

## Methodology

**Step 1 — Data collection**
Collect earnings call transcripts for FTSE 100 companies from Motley Fool UK and similar sources. Target: 50+ transcripts across 10–15 companies, 2022–2025.

**Step 2 — Sentiment scoring**
Run each transcript through [FinBERT](https://huggingface.co/ProsusAI/finbert), a transformer model trained specifically on financial language. Each transcript gets a score: positive, negative, or neutral, plus a confidence score.

**Step 3 — Stock return data**
Use `yfinance` to pull daily closing prices. Calculate cumulative returns for the 1, 3, and 5 trading days after each earnings call date.

**Step 4 — Correlation analysis**
Test whether higher positive sentiment scores predict higher subsequent returns. Run OLS regression (sentiment score → return) and visualise the relationship.

**Step 5 — Results & visualisation**
Plot sentiment scores over time per company, scatter plots of sentiment vs returns, and a summary heatmap across sectors.

---

## Tech stack

| Tool | Purpose |
|------|---------|
| `Python 3.11` | Core language |
| `FinBERT` (HuggingFace) | Financial sentiment model |
| `yfinance` | Stock price data |
| `pandas` | Data manipulation |
| `scikit-learn` | Regression analysis |
| `matplotlib / seaborn` | Visualisation |
| `BeautifulSoup` | Web scraping |
| `Jupyter` | Notebooks |

---

## Sample companies

Initial analysis covers a cross-section of FTSE 100 sectors:

- **Banks:** Barclays, HSBC, Lloyds
- **Energy:** BP, Shell
- **Consumer:** Unilever, Tesco
- **Tech/Media:** RELX, Sage Group
- **Mining:** Rio Tinto, Glencore

---

## Status

🟡 In progress — Summer 2026

- [x] Project structure set up
- [ ] Transcript collection (target: 50 calls)
- [ ] FinBERT sentiment pipeline
- [ ] Stock return data collection
- [ ] Correlation analysis
- [ ] Final visualisations

---

## Author

**Arjun Gill** — BSc Economics, Finance & Data Science, Imperial College London

[LinkedIn](https://www.linkedin.com/in/arjungill1074/)
