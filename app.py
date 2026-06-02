from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import json
import os
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime, timezone

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
SCRAPER_API_KEY = os.environ.get("SCRAPER_API_KEY", "")
app = Flask(__name__)
CORS(app)

# ==============================
# USER / CONTACT FILES (server.py)
# ==============================
USER_FILE    = "users.json"
CONTACT_FILE = "contacts.json"

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(CONTACT_FILE):
    with open(CONTACT_FILE, "w") as f:
        json.dump([], f)

# ==============================
# MODEL DIRECTORY (app.py)
# ==============================
MODEL_DIR = "models"

# ==============================
# LOAD .PKL MODELS
# ==============================
models = {}

print("Loading .pkl models...")
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl"):
        name = file.split("_")[0].lower()
        with open(os.path.join(MODEL_DIR, file), "rb") as f:
            models[name] = pickle.load(f)
        print(f"  ✓ {name} model loaded")

print("All PKL models:", list(models.keys()))

# ==============================
# LOAD LSTM MODELS
# ==============================
lstm_models = {}

print("Loading LSTM models...")
for file in os.listdir(MODEL_DIR):
    if file.endswith(".h5"):
        name = file.split("_")[0].lower()
        lstm_models[name] = load_model(os.path.join(MODEL_DIR, file))
        print(f"  ✓ LSTM {name} loaded")

print("All LSTM models:", list(lstm_models.keys()))

# ==============================
# SCALER
# ==============================
scaler = MinMaxScaler(feature_range=(0, 1))

# ==============================
# LSTM PREDICTION FUNCTION
# ==============================
def lstm_predict(df, model):
    try:
        close_data = df[['Close']]
        scaled = scaler.fit_transform(close_data)

        if len(scaled) < 60:
            return None

        last_60 = scaled[-60:]
        X_input = np.array([last_60])

        pred       = model.predict(X_input)
        pred_price = scaler.inverse_transform(pred)
        return float(pred_price[0][0])

    except Exception as e:
        print("LSTM error:", e)
        return None

# ==============================
# YAHOO SESSION
# ==============================
_yahoo_session = None
_yahoo_crumb   = None


SCRAPER_API_KEY = "f3cda1f628b1ca4ef74e41619820bc7c"   # ← paste your key

def fetch_yahoo_price(symbol):
    target_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1mo"
    proxy_url  = f"http://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url={target_url}"

    r = requests.get(proxy_url, timeout=60)   # 60s timeout — scraper can be slow
    r.raise_for_status()

    closes = [
        c for c in
        r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        if c is not None
    ]

    if len(closes) < 2:
        raise ValueError(f"Not enough data for {symbol}")

    current = round(float(closes[-1]), 2)
    prev    = round(float(closes[-2]), 2)
    change  = round(current - prev, 2)
    pct     = round((change / prev) * 100, 2) if prev else 0.0
    return current, change, pct

def fetch_yahoo_historical(symbol, period):
    range_map = {
        "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
        "1y": "1y", "2y": "2y", "5y": "5y",
    }
    yf_range   = range_map.get(period, "6mo")
    target_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={yf_range}"
    proxy_url  = f"http://api.scraperapi.com?api_key={f3cda1f628b1ca4ef74e41619820bc7c}&url={target_url}"

    r = requests.get(proxy_url, timeout=60)
    r.raise_for_status()

    # ... rest of the function stays exactly the same

    return records

# ──────────────────────────────────────────────────────────────
# STOCK LISTS
# ──────────────────────────────────────────────────────────────
TICKER_STOCKS = [
    ("NIFTY 50",   "^NSEI"),
    ("SENSEX",     "^BSESN"),
    ("TCS",        "TCS.NS"),
    ("RELIANCE",   "RELIANCE.NS"),
    ("INFY",       "INFY.NS"),
    ("HDFC BANK",  "HDFCBANK.NS"),
    ("ICICI BANK", "ICICIBANK.NS"),
    ("WIPRO",      "WIPRO.NS"),
    ("BHARTI",     "BHARTIARTL.NS"),
    ("ITC",        "ITC.NS"),
    ("BAJFINANCE", "BAJFINANCE.NS"),
    ("SBIN",       "SBIN.NS"),
]

NIFTY50_SYMBOLS = [
    ("RELIANCE",   "RELIANCE.NS"),
    ("TCS",        "TCS.NS"),
    ("HDFCBANK",   "HDFCBANK.NS"),
    ("INFY",       "INFY.NS"),
    ("ICICIBANK",  "ICICIBANK.NS"),
    ("HINDUNILVR", "HINDUNILVR.NS"),
    ("ITC",        "ITC.NS"),
    ("SBIN",       "SBIN.NS"),
    ("BHARTIARTL", "BHARTIARTL.NS"),
    ("KOTAKBANK",  "KOTAKBANK.NS"),
    ("LT",         "LT.NS"),
    ("BAJFINANCE", "BAJFINANCE.NS"),
    ("WIPRO",      "WIPRO.NS"),
    ("HCLTECH",    "HCLTECH.NS"),
    ("AXISBANK",   "AXISBANK.NS"),
    ("ASIANPAINT", "ASIANPAINT.NS"),
    ("MARUTI",     "MARUTI.NS"),
    ("SUNPHARMA",  "SUNPHARMA.NS"),
    ("TATAMOTORS", "TATAMOTORS.NS"),
    ("NTPC",       "NTPC.NS"),
    ("POWERGRID",  "POWERGRID.NS"),
    ("ULTRACEMCO", "ULTRACEMCO.NS"),
    ("TITAN",      "TITAN.NS"),
    ("NESTLEIND",  "NESTLEIND.NS"),
    ("TECHM",      "TECHM.NS"),
]

HISTORICAL_TICKERS = {
    "NIFTY":    "^NSEI",
    "SENSEX":   "^BSESN",
    "TCS":      "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "INFY":     "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK":"ICICIBANK.NS",
    "WIPRO":    "WIPRO.NS",
    "SBIN":     "SBIN.NS",
    "MARUTI":   "MARUTI.NS",
}

# ──────────────────────────────────────────────────────────────
# PAGE ROUTES  (from server.py)
# ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login-page')
def login_page():
    return render_template('login.html')

@app.route('/register-page')
def register_page():
    return render_template('register.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/holiday')
def holiday():
    return render_template('holiday.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/press_release')
def press_release():
    return render_template('press_release.html')

# ───────── PATTERN PAGES ─────────

@app.route('/reversal_chart_pattern')
def reversal_chart_pattern():
    return render_template('reversal_chart_pattern.html')

@app.route('/sidewase_chart_pattern')
def sideways_chart_pattern():
    return render_template('sidewase_chart_pattern.html')

@app.route('/single_candlestick')
def single_candlestick():
    return render_template('single_candlestick.html')

@app.route('/double_candelestic')
def double_candlestick():
    return render_template('double_candelestic.html')

@app.route('/triple_candelestic')
def triple_candlestick():
    return render_template('triple_candelestic.html')

@app.route('/continuation_chart_pattern')
def continuation_chart_pattern():
    return render_template('continuation_chart_pattern.html')

# ──────────────────────────────────────────────────────────────
# AUTH ROUTES  (from server.py)
# ──────────────────────────────────────────────────────────────

@app.route('/register', methods=['POST'])
def register():
    data     = request.get_json()
    username = data.get("username")
    password = data.get("password")

    with open(USER_FILE, "r") as f:
        users = json.load(f)

    for user in users:
        if user["username"] == username:
            return jsonify({"success": False, "message": "User already exists"})

    users.append({"username": username, "password": password})

    with open(USER_FILE, "w") as f:
        json.dump(users, f)

    return jsonify({"success": True, "message": "Registered successfully"})


@app.route('/login', methods=['POST'])
def login():
    data     = request.get_json()
    username = data.get("username")
    password = data.get("password")

    with open(USER_FILE, "r") as f:
        users = json.load(f)

    for user in users:
        if user["username"] == username and user["password"] == password:
            return jsonify({"success": True, "message": "Login successful"})

    return jsonify({"success": False, "message": "Invalid credentials"})


@app.route('/contact-submit', methods=['POST'])
def contact_submit():
    data = request.get_json()

    with open(CONTACT_FILE, "r") as f:
        contacts = json.load(f)

    contacts.append(data)

    with open(CONTACT_FILE, "w") as f:
        json.dump(contacts, f)

    return jsonify({"success": True, "message": "Message saved successfully"})

# ──────────────────────────────────────────────────────────────
# API ROUTES  (from app.py)
# ──────────────────────────────────────────────────────────────

@app.route("/test")
def test():
    return jsonify({"success": True, "message": "API is working 🚀"})


# ── Manual Prediction ──────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()

        company    = body.get("company")
        open_price = float(body.get("open"))
        high_price = float(body.get("high"))
        low_price  = float(body.get("low"))
        volume     = float(body.get("volume"))

        model_key_map = {
            "TCS":      "tcs",
            "Reliance": "reliance",
            "NIFTY":    "nifty50",
            "SENSEX":   "sensex",
        }

        key = model_key_map.get(company)
        if not key:
            return jsonify({"success": False, "error": f"Unknown company: {company}"}), 400
        if key not in models:
            return jsonify({"success": False, "error": "Model not found"}), 500

        model      = models[key]
        input_data = np.array([[open_price, high_price, low_price, volume]])
        prediction = model.predict(input_data)[0]

        return jsonify({
            "success":        True,
            "company":        company,
            "predicted_close": float(prediction),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Live + LSTM Prediction ─────────────────────────────────────
@app.route("/predict_live", methods=["GET"])
def predict_live():
    try:
        company = request.args.get("company")

        ticker_map = {
            "TCS":      "TCS.NS",
            "Reliance": "RELIANCE.NS",
            "NIFTY":    "^NSEI",
            "SENSEX":   "^BSESN",
        }
        model_key_map = {
            "TCS":      "tcs",
            "Reliance": "reliance",
            "NIFTY":    "nifty50",
            "SENSEX":   "sensex",
        }

        ticker = ticker_map.get(company)
        if not ticker:
            return jsonify({"success": False, "error": "Unknown company"}), 400

        stock   = yf.Ticker(ticker)
        df_live = stock.history(period="5d")

        if df_live.empty:
            return jsonify({"success": False, "error": "Market closed or no data"}), 404

        latest     = df_live.iloc[-1]
        open_price = latest["Open"]
        high_price = latest["High"]
        low_price  = latest["Low"]
        volume     = latest["Volume"]

        key        = model_key_map[company]
        model      = models[key]
        input_data = np.array([[open_price, high_price, low_price, volume]])
        price_pred = float(model.predict(input_data)[0])

        df_lstm   = yf.download(ticker, period="6mo")
        lstm_pred = None
        trend     = None

        if key in lstm_models:
            lstm_model = lstm_models[key]
            lstm_pred  = lstm_predict(df_lstm, lstm_model)
            if lstm_pred:
                today_close = df_lstm["Close"].iloc[-1]
                trend = "UP" if lstm_pred > today_close else "DOWN"

        return jsonify({
            "success": True,
            "company": company,
            "live_data": {
                "open":   float(open_price),
                "high":   float(high_price),
                "low":    float(low_price),
                "volume": float(volume),
            },
            "predicted_close": price_pred,
            "lstm_prediction": lstm_pred,
            "trend":           trend,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Live Ticker ────────────────────────────────────────────────
@app.route("/live_ticker")
def live_ticker():
    results = []
    for name, symbol in TICKER_STOCKS:
        try:
            price, change, pct = fetch_yahoo_price(symbol)
            results.append({"name": name, "price": price, "change": change, "percent": pct})
            time.sleep(0.1)
        except Exception as e:
            print(f"[ticker] Skipping {symbol}: {e}")
            continue

    if not results:
        return jsonify({"success": False, "error": "Yahoo Finance unavailable. Try again in a minute."}), 500

    return jsonify({"success": True, "data": results})


# ── Top Gainers ────────────────────────────────────────────────
@app.route("/top_gainers")
def top_gainers():
    try:
        gainers = []
        for name, symbol in NIFTY50_SYMBOLS:
            try:
                price, change, pct = fetch_yahoo_price(symbol)
                if pct > 0:
                    gainers.append({"name": name, "price": price, "change": change, "percent": pct})
                time.sleep(0.08)
            except Exception as e:
                print(f"[gainers] Skipping {symbol}: {e}")
                continue

        gainers.sort(key=lambda x: x["percent"], reverse=True)
        top5 = gainers[:5]

        if not top5:
            return jsonify({"success": False, "error": "No gainers data. Market may be closed."}), 500

        return jsonify({"success": True, "data": top5})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Historical Data ────────────────────────────────────────────
@app.route("/historical")
def historical():
    try:
        symbol_key = request.args.get("symbol", "NIFTY")
        period     = request.args.get("period", "6mo")

        ticker_sym = HISTORICAL_TICKERS.get(symbol_key)
        if not ticker_sym:
            return jsonify({"success": False, "error": f"Unknown symbol: {symbol_key}"}), 400

        records = fetch_yahoo_historical(ticker_sym, period)

        if not records:
            return jsonify({"success": False, "error": "No data returned — market may be closed or symbol invalid"}), 404

        closes      = [r["close"]  for r in records]
        first_close = closes[0]
        last_close  = closes[-1]
        pct_change  = round(((last_close - first_close) / first_close) * 100, 2) if first_close else 0
        period_high = round(max(r["high"]   for r in records), 2)
        period_low  = round(min(r["low"]    for r in records), 2)
        avg_vol     = int(sum(r["volume"]   for r in records) / len(records))

        return jsonify({
            "success": True,
            "symbol":  symbol_key,
            "period":  period,
            "data":    records,
            "summary": {
                "start_price": first_close,
                "end_price":   last_close,
                "pct_change":  pct_change,
                "period_high": period_high,
                "period_low":  period_low,
                "avg_volume":  avg_vol,
                "data_points": len(records),
            },
        })

    except Exception as e:
        import traceback
        print("[historical] Error:", traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
   
    app.run(host="0.0.0.0", port=8080, debug=True)
