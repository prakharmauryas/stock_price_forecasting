import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(path, lookback=60):
    """
    Loads stock price data from a CSV, cleans and scales the 'Close' column,
    and prepares windowed sequences for LSTM.

    Parameters:
    - path (str): path to CSV file (e.g., data/raw/AAPL.csv)
    - lookback (int): number of time steps per sample (default: 60)

    Returns:
    - X (np.ndarray): shape (samples, lookback, 1)
    - y (np.ndarray): shape (samples, 1)
    - scaler (MinMaxScaler): fitted scaler for inverse transform
    """

    # Load CSV
    df = pd.read_csv(path)

    # Drop any rows that contain the word 'AAPL' (likely a duplicate header row)
    df = df[~df.astype(str).apply(lambda row: row.str.contains('AAPL')).any(axis=1)]

    # Convert 'Close' column to numeric and drop NaNs
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    # Extract and reshape closing prices
    close_prices = df['Close'].values.reshape(-1, 1)

    # Scale prices
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    # Create sliding windows of length = lookback
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler
