import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# ── Individual quarter plots ──────────────────────────────────────────────────

os.makedirs("outputs/quarterly_charts", exist_ok=True)

for quarter, date in EARNINGS_DATES.items():
    print(f"Plotting {quarter}...")
    close, returns = get_price_data(TICKER, date)
    if close is None:
        print(f"  No data for {quarter}")
        continue

    fig, ax = plt.subplots(figsize=(9, 4))

    # Colour the line green or red based on 5-day return
    ret5 = returns.get(5, 0)
    line_colour = "#2ecc71" if ret5 >= 0 else "#e74c3c"

    ax.plot(close.index, close.values, marker="o", markersize=4,
            color=line_colour, linewidth=2)
    ax.axvline(x=pd.to_datetime(date), color="#e74c3c",
               linestyle="--", linewidth=1.5, label="Earnings call", zorder=5)

    # Shade the post-earnings region
    ax.axvspan(pd.to_datetime(date), close.index[-1],
               alpha=0.06, color=line_colour)

    # Return annotations
    ret_text = "\n".join([
        f"  +1d: {returns.get(1, 'N/A'):+.2f}%" if isinstance(returns.get(1), float) else "  +1d: N/A",
        f"  +3d: {returns.get(3, 'N/A'):+.2f}%" if isinstance(returns.get(3), float) else "  +3d: N/A",
        f"  +5d: {returns.get(5, 'N/A'):+.2f}%" if isinstance(returns.get(5), float) else "  +5d: N/A",
    ])
    ax.text(0.02, 0.97, ret_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgrey", alpha=0.9))

    ax.set_title(f"Barclays (BARC.L) — {quarter} Earnings Call  |  {date}",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Price (pence)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()

    path = f"outputs/quarterly_charts/BARC_{quarter}_price.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Summary grid: all quarters on one page ────────────────────────────────────

print("\nBuilding summary grid...")
quarters = list(EARNINGS_DATES.keys())
n        = len(quarters)
ncols    = 3
nrows    = (n + ncols - 1) // ncols

fig = plt.figure(figsize=(18, nrows * 4))
fig.suptitle("Barclays (BARC.L) — Price Around Earnings Calls: Q123 to Q126",
             fontsize=15, fontweight="bold", y=1.01)

for i, (quarter, date) in enumerate(EARNINGS_DATES.items()):
    ax = fig.add_subplot(nrows, ncols, i + 1)
    close, returns = get_price_data(TICKER, date)
    if close is None:
        ax.set_title(f"{quarter} — no data")
        continue

    ret5        = returns.get(5, 0)
    line_colour = "#2ecc71" if ret5 >= 0 else "#e74c3c"

    ax.plot(close.index, close.values, marker="o", markersize=3,
            color=line_colour, linewidth=1.5)
    ax.axvline(x=pd.to_datetime(date), color="#e74c3c",
               linestyle="--", linewidth=1.2, zorder=5)
    ax.axvspan(pd.to_datetime(date), close.index[-1],
               alpha=0.06, color=line_colour)

    ret_text = (
        f"+1d: {returns.get(1, 'N/A'):+.1f}%  "
        f"+3d: {returns.get(3, 'N/A'):+.1f}%  "
        f"+5d: {returns.get(5, 'N/A'):+.1f}%"
        if all(isinstance(returns.get(w), float) for w in [1, 3, 5])
        else ""
    )
    ax.set_title(f"{quarter}  |  {date}\n{ret_text}", fontsize=8, fontweight="bold")
    ax.set_ylabel("Price (p)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
grid_path = "outputs/all_quarters_price_grid.png"
plt.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Grid saved: {grid_path}")
print("\nAll done.")