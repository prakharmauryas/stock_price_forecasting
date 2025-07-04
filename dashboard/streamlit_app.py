import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

LOOKBACK = 60
MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.save"
DATA_PATH = "data/raw/AAPL.csv"

# Load model and scaler
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# Prepare input sequence
def prepare_input(path, lookback, scaler):
    df = pd.read_csv(path)
    df = df[~df.astype(str).apply(lambda row: row.str.contains('AAPL')).any(axis=1)]
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    close_prices = df['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close_prices)

    if len(scaled) < lookback:
        raise ValueError("Not enough data for prediction.")

    input_seq = scaled[-lookback:].reshape(1, lookback, 1)
    return df, input_seq

# Streamlit UI
st.title("📈 Stock Price Forecaster (LSTM)")
st.write("This app uses an LSTM model to forecast the next closing price for Apple (AAPL).")

model, scaler = load_resources()

# Load data & prepare
df, X_input = prepare_input(DATA_PATH, LOOKBACK, scaler)

# Predict
pred_scaled = model.predict(X_input)
predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

# Display latest data
st.subheader("🔍 Recent 60 Days")
st.line_chart(df['Close'].tail(LOOKBACK))

# Show prediction
st.subheader("🔮 Predicted Next Closing Price")
st.success(f"${predicted_price:.2f}")

# Show full chart with prediction
st.subheader("📊 Forecast Visualization")

extended_close = df['Close'].tail(LOOKBACK).tolist()
extended_close.append(predicted_price)

plt.figure(figsize=(10, 4))
plt.plot(range(LOOKBACK), extended_close[:-1], label="Past 60 Days")
plt.plot(LOOKBACK, predicted_price, 'ro', label="Next Prediction")
plt.title("Stock Price Forecast")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
st.pyplot(plt)
