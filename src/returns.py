"""
returns.py
----------
Pulls FTSE 100 stock price data and calculates post-earnings returns.

Usage:
    from src.returns import get_post_earnings_returns
    df = get_post_earnings_returns("BARC.L", "2024-02-15")
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


FTSE100_TICKERS = {
    "Barclays": "BARC.L",
    "HSBC": "HSBA.L",
    "Lloyds": "LLOY.L",
    "BP": "BP.L",
    "Shell": "SHEL.L",
    "Unilever": "ULVR.L",
    "Tesco": "TSCO.L",
    "RELX": "REL.L",
    "Sage Group": "SGE.L",
    "Rio Tinto": "RIO.L",
    "Glencore": "GLEN.L",
}


def get_post_earnings_returns(ticker, earnings_date, windows=[1, 3, 5]):
    """
    Calculate stock returns for N trading days after an earnings call.

    Args:
        ticker: Yahoo Finance ticker (e.g. 'BARC.L')
        earnings_date: date string 'YYYY-MM-DD'
        windows: list of day windows to calculate returns for

    Returns:
        dict with return for each window
    """
    date = pd.to_datetime(earnings_date)
    start = date - timedelta(days=2)
    end = date + timedelta(days=max(windows) + 5)

    stock = yf.download(ticker, start=start, end=end, progress=False)

    if stock.empty:
        return None

    stock = stock["Close"].dropna()
    trading_days = stock.index.tolist()

    try:
        earnings_idx = min(
            range(len(trading_days)),
            key=lambda i: abs((trading_days[i] - date).days)
        )
    except ValueError:
        return None

    returns = {"ticker": ticker, "earnings_date": earnings_date}
    base_price = stock.iloc[earnings_idx]

    for w in windows:
        target_idx = earnings_idx + w
        if target_idx < len(stock):
            future_price = stock.iloc[target_idx]
            ret = (future_price - base_price) / base_price
            returns[f"return_{w}d"] = round(float(ret), 4)
        else:
            returns[f"return_{w}d"] = None

    return returns


def build_returns_table(earnings_events):
    """
    Build a full returns table from a list of earnings events.

    Args:
        earnings_events: list of dicts with keys: company, ticker, date

    Returns:
        DataFrame with returns for each event
    """
    rows = []
    for event in earnings_events:
        result = get_post_earnings_returns(event["ticker"], event["date"])
        if result:
            result["company"] = event["company"]
            rows.append(result)

    return pd.DataFrame(rows)
