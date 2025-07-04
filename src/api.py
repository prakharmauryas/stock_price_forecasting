import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

LOOKBACK = 200
MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.save"
DATA_DIR = "data/raw"

def load_model_and_scaler():
    """Load the trained LSTM model and scaler."""
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def clean_and_prepare_input(scaler, ticker="AAPL", lookback=LOOKBACK):
    """Prepare the latest input sequence for prediction."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file for {ticker} not found.")

    df = pd.read_csv(path)
    df = df[~df.astype(str).apply(lambda row: row.str.contains(ticker)).any(axis=1)]
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    if len(df) < lookback:
        raise ValueError(f"Not enough data for ticker {ticker}. Need at least {lookback} rows.")

    close_prices = df['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close_prices)

    input_seq = scaled[-lookback:].reshape(1, lookback, 1)
    return df, input_seq

def predict_next_price(model, scaler, X_input):
    """Predict a single next price."""
    pred_scaled = model.predict(X_input)
    return scaler.inverse_transform(pred_scaled)[0][0]

def predict_multi_day(model, scaler, input_seq, days=7):
    """Predict multiple future prices using rolling forecast."""
    forecast = []
    current_input = input_seq.copy()

    for _ in range(days):
        pred_scaled = model.predict(current_input)
        forecast.append(pred_scaled[0][0])

        # Roll and append the new prediction
        next_step = np.append(current_input[0][1:], [[pred_scaled[0][0]]], axis=0)
        current_input = next_step.reshape(1, LOOKBACK, 1)

    return scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
