Perfect! Below are your final `README.md` and `requirements.txt` files that you can directly include in your GitHub repo for the stock price forecasting project.

---

### ✅ `README.md`

```markdown
# 📈 Stock Price Forecasting using LSTM (Streamlit App)

This project is a **Streamlit web application** that uses a **trained LSTM (Long Short-Term Memory)** model to forecast future stock prices based on historical closing price data. You can forecast stock prices for companies like **AAPL, MSFT, GOOG, TSLA, etc.** using a time-series deep learning approach.

---

## 🚀 Features

- Predict **next N days** of stock prices using LSTM
- Choose from popular tickers (AAPL, MSFT, TSLA, GOOG)
- Visualize the last 200 days + forecast
- Clean, interactive Streamlit dashboard
- Train your own LSTM with any ticker

---

## 📂 Project Structure

```

stock\_price\_forecasting/
├── dashboard/
│   └── streamlit\_app.py        # Streamlit UI
├── data/
│   └── raw/                    # Raw CSVs saved from yfinance
├── models/
│   ├── lstm\_model.h5           # Trained Keras LSTM model
│   └── scaler.save             # Trained MinMaxScaler
├── src/
│   ├── **init**.py
│   ├── api.py                  # Model loading & prediction logic
│   ├── data\_loader.py          # Fetch data from Yahoo Finance
│   └── train\_model.py          # LSTM training script
├── requirements.txt
└── README.md

````

---

## 🧪 How to Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/prakharmauryas/stock_price_forecasting.git
cd stock_price_forecasting
````

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate     # On Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run dashboard/streamlit_app.py
```

---

## 🧠 Model Training

To retrain the LSTM model using updated data:

```bash
python -m src.train_model
```

This will:

* Fetch and preprocess data
* Train an LSTM model
* Save it to `models/lstm_model.h5` and `models/scaler.save`

---

## 📦 Dependencies

* `tensorflow`
* `streamlit`
* `pandas`, `numpy`, `matplotlib`
* `scikit-learn`
* `yfinance`
* `joblib`

---

## 🙌 Acknowledgements

* [DeepLearning.AI](https://deeplearning.ai/) Time Series Forecasting inspiration
* [Yahoo Finance API](https://pypi.org/project/yfinance/)
* [Streamlit](https://streamlit.io/)

---

## 📬 Contact

**Prakhar Maurya**
📧 [prakhar.myself@gmail.com](mailto:prakhar.myself@gmail.com)
📌 Feel free to contribute, fork, or open issues!

````

---



