import numpy as np
import os
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.feature_engineering import load_and_preprocess

print("✅ Starting LSTM training...")

# Load preprocessed data
X, y, scaler = load_and_preprocess("data/raw/AAPL.csv", lookback=60)
print(f"✅ Loaded and preprocessed data. Total samples: {len(X)}")

# Reshape X to 3D shape [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
print(f"✅ Reshaped X to: {X.shape}")

# Train-test split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
print("✅ Model architecture built.")

model.compile(optimizer='adam', loss='mean_squared_error')
print("✅ Model compiled.")

# Set up early stopping
early_stop = EarlyStopping(monitor='loss', patience=3)

# Train the model
print("🚀 Starting training...")
model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[early_stop], verbose=1)

# Save model and scaler
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/scaler.save")
print("✅ Model saved to models/lstm_model.h5")
print("✅ Scaler saved to models/scaler.save")
print("🎉 LSTM model training complete!")
