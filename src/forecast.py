import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

LOOKBACK = 60
MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.save"
DATA_PATH = "data/raw/AAPL.csv"

def load_latest_sequence(path, lookback, scaler):
    df = pd.read_csv(path)
    df = df[~df.astype(str).apply(lambda row: row.str.contains('AAPL')).any(axis=1)]
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    close_prices = df['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close_prices)
    
    if len(scaled) < lookback:
        raise ValueError("Not enough data for prediction window")
    
    return scaled[-lookback:].reshape(1, lookback, 1)

def forecast_next():
    print("🔁 Loading model and scaler...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("📈 Preparing input...")
    X_input = load_latest_sequence(DATA_PATH, LOOKBACK, scaler)

    print("🔮 Predicting next price...")
    pred_scaled = model.predict(X_input)
    pred = scaler.inverse_transform(pred_scaled)

    print(f"📅 Next predicted closing price: ${pred[0][0]:.2f}")

if __name__ == "__main__":
    forecast_next()
