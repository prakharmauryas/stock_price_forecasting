from src.data_loader import fetch_data

# Fetch stock data for AAPL (Apple Inc.)
df = fetch_data(ticker="AAPL", start="2015-01-01", end="2024-01-01")
print(df.head())
