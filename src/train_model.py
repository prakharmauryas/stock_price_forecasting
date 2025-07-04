import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import joblib

LOOKBACK = 200
DATA_PATH = "data/raw/AAPL.csv"
MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.save"

def load_data(path):
    df = pd.read_csv(path)
    df = df[~df.astype(str).apply(lambda row: row.str.contains('AAPL')).any(axis=1)]
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    return df[['Close']].values

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

print("✅ Starting LSTM training...")

# Load and scale data
raw_close = load_data(DATA_PATH)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(raw_close)

# Create sequences
X, y = create_sequences(scaled_data, LOOKBACK)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32, callbacks=[early_stop], verbose=1)

# Save
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ Model trained and saved.")
