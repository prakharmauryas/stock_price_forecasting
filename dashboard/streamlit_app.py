import sys
import os
import streamlit as st
import matplotlib.pyplot as plt

# Allow importing from the src/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api import (
    load_model_and_scaler,
    clean_and_prepare_input,
    predict_multi_day
)
from src.data_loader import fetch_data

LOOKBACK = 200  # Days of past data to look at

# Streamlit config
st.set_page_config(page_title="Stock Price Forecast", layout="centered")
st.title("📈 Stock Price Forecaster (LSTM)")
st.write("Forecast the next **N closing prices** using an LSTM model trained on historical stock data.")

# Load model & scaler (cached)
@st.cache_resource
def load_resources():
    return load_model_and_scaler()

model, scaler = load_resources()

# Select stock ticker
ticker = st.selectbox("📊 Choose Stock Ticker", ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"])

# Forecast range (1 to 30 days)
N_DAYS = st.slider("📅 Select number of days to forecast", min_value=1, max_value=30, value=7)

# Cache and fetch historical data
@st.cache_data
def get_data(ticker):
    return fetch_data(ticker=ticker, save=True)

get_data(ticker)

# Prepare input
try:
    df, X_input = clean_and_prepare_input(scaler=scaler, ticker=ticker, lookback=LOOKBACK)

    # Run multi-day prediction
    future_prices = predict_multi_day(model, scaler, X_input[-LOOKBACK:], days=N_DAYS)

    # 📈 Show past data
    st.subheader(f"🔍 Recent {LOOKBACK} Days - {ticker}")
    st.line_chart(df['Close'].tail(LOOKBACK))

    # 🔮 Show future prices
    st.subheader(f"🔮 Predicted Closing Prices for Next {N_DAYS} Days")
    for i, price in enumerate(future_prices, 1):
        st.markdown(f"- Day {i}: **${price:.2f}**")

    # 📊 Plot forecast
    st.subheader("📊 Forecast Visualization")
    historical = df['Close'].tail(LOOKBACK).tolist()
    x_future = list(range(len(historical), len(historical) + N_DAYS))

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(historical)), historical, label="Historical")
    plt.plot(x_future, future_prices, 'ro--', label="Forecast")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.title(f"{ticker} Stock Price Forecast")
    plt.legend()
    st.pyplot(plt)

except Exception as e:
    st.error(f"🚫 Error occurred: {e}")
