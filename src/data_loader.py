# src/data_loader.py

import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker="AAPL", start="2015-01-01", end="2024-01-01", save=True):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    if save:
        os.makedirs("data/raw", exist_ok=True)
        file_path = f"data/raw/{ticker}.csv"
        df.to_csv(file_path)
        print(f"✅ Data saved to {file_path}")
    
    return df
