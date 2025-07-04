from datetime import datetime
import yfinance as yf
import os

def fetch_data(ticker="AAPL", start="1980-01-01", end="2024-01-01", save=True):
    df = yf.download(ticker, start=start, end=end)
    
    if save:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(f"data/raw/{ticker}.csv")
        print(f"✅ Data saved to data/raw/{ticker}.csv")
    
    return df
