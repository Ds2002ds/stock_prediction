from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np
import yfinance as yf
import requests
import time

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# ==============================
# MODEL DIRECTORY
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
# SCALER (temporary)
# ==============================
scaler = MinMaxScaler(feature_range=(0,1))

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

       pred = model.predict(X_input)
       pred_price = scaler.inverse_transform(pred)

       return float(pred_price[0][0])

   except Exception as e:
       print("LSTM error:", e)
       return None

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
   return jsonify({"status": "ok", "message": "Stock Prediction API Running 🚀"})


@app.route("/test")
def test():
   return jsonify({"success": True, "message": "API is working 🚀"})


# ==============================
# MANUAL PREDICTION (UNCHANGED)
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
   try:
       body = request.get_json()

       company = body.get("company")
       open_price = float(body.get("open"))
       high_price = float(body.get("high"))
       low_price  = float(body.get("low"))
       volume     = float(body.get("volume"))

       model_key_map = {
           "TCS": "tcs",
           "Reliance": "reliance",
           "NIFTY": "nifty50",
           "SENSEX": "sensex"
       }

       key = model_key_map.get(company)

       if not key:
           return jsonify({"success": False, "error": f"Unknown company: {company}"}), 400

       if key not in models:
           return jsonify({"success": False, "error": f"Model not found"}), 500

       model = models[key]

       input_data = np.array([[open_price, high_price, low_price, volume]])
       prediction = model.predict(input_data)[0]

       return jsonify({
           "success": True,
           "company": company,
           "predicted_close": float(prediction)
       })

   except Exception as e:
       return jsonify({"success": False, "error": str(e)}), 500


# ==============================
# LIVE + LSTM PREDICTION
# ==============================
@app.route("/predict_live", methods=["GET"])
def predict_live():
   try:
       company = request.args.get("company")

       ticker_map = {
           "TCS": "TCS.NS",
           "Reliance": "RELIANCE.NS",
           "NIFTY": "^NSEI",
           "SENSEX": "^BSESN"
       }

       model_key_map = {
           "TCS": "tcs",
           "Reliance": "reliance",
           "NIFTY": "nifty50",
           "SENSEX": "sensex"
       }

       ticker = ticker_map.get(company)

       if not ticker:
           return jsonify({"success": False, "error": f"Unknown company"}), 400

       # ------------------------
       # LIVE DATA (1 DAY)
       # ------------------------
       stock = yf.Ticker(ticker)
       df_live = stock.history(period="5d")

       if df_live.empty:
           return jsonify({"success": False, "error": "Market closed or no data"}), 404

       latest = df_live.iloc[-1]

       open_price = latest["Open"]
       high_price = latest["High"]
       low_price  = latest["Low"]
       volume     = latest["Volume"]

       # ------------------------
       # ENSEMBLE PREDICTION
       # ------------------------
       key = model_key_map[company]
       model = models[key]

       input_data = np.array([[open_price, high_price, low_price, volume]])
       price_pred = float(model.predict(input_data)[0])

       # ------------------------
       # LSTM DATA (6 MONTHS)
       # ------------------------
       df_lstm = yf.download(ticker, period="6mo")

       lstm_pred = None
       trend = None

       if key in lstm_models:
           lstm_model = lstm_models[key]

           lstm_pred = lstm_predict(df_lstm, lstm_model)

           if lstm_pred:
               today_close = df_lstm["Close"].iloc[-1]
               trend = "UP" if lstm_pred > today_close else "DOWN"

       # ------------------------
       # FINAL RESPONSE
       # ------------------------
       return jsonify({
           "success": True,
           "company": company,

           "live_data": {
               "open": float(open_price),
               "high": float(high_price),
               "low": float(low_price),
               "volume": float(volume)
           },

           "predicted_close": price_pred,   # OLD MODEL
           "lstm_prediction": lstm_pred,    # NEW MODEL
           "trend": trend                  # PATTERN
       })

   except Exception as e:
       return jsonify({"success": False, "error": str(e)}), 500


# ==============================
# LIVE TICKER — Yahoo crumb session
# ==============================

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

_yahoo_session = None
_yahoo_crumb   = None

def init_yahoo_session():
    global _yahoo_session, _yahoo_crumb
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        s.get("https://fc.yahoo.com", timeout=5)
        s.get("https://finance.yahoo.com", timeout=8)
        r = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=8)
        if r.status_code == 200 and r.text and len(r.text) < 50:
            _yahoo_crumb = r.text.strip()
            print(f"[ticker] Yahoo session ready, crumb: {_yahoo_crumb}")
        else:
            print(f"[ticker] Crumb failed ({r.status_code}), will try without crumb")
    except Exception as e:
        print(f"[ticker] Session init error: {e}")
    _yahoo_session = s

def fetch_yahoo_price(symbol):
    global _yahoo_session, _yahoo_crumb
    if not _yahoo_session:
        init_yahoo_session()

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1mo"
    if _yahoo_crumb:
        url += f"&crumb={_yahoo_crumb}"

    r = _yahoo_session.get(url, timeout=10)

    # Refresh session and retry once on auth failure
    if r.status_code in (401, 403):
        init_yahoo_session()
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1mo"
        if _yahoo_crumb:
            url += f"&crumb={_yahoo_crumb}"
        r = _yahoo_session.get(url, timeout=10)

    r.raise_for_status()
    closes = [c for c in r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"] if c is not None]

    if len(closes) < 2:
        raise ValueError(f"Not enough data for {symbol}")

    current = round(float(closes[-1]), 2)
    prev    = round(float(closes[-2]), 2)
    change  = round(current - prev, 2)
    pct     = round((change / prev) * 100, 2) if prev else 0.0
    return current, change, pct


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

#TOPGAINER
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

@app.route("/top_gainers")
def top_gainers():
    try:
        gainers = []
        for name, symbol in NIFTY50_SYMBOLS:
            try:
                price, change, pct = fetch_yahoo_price(symbol)
                if pct > 0:
                    gainers.append({
                        "name":    name,
                        "price":   price,
                        "change":  change,
                        "percent": pct,
                    })
                time.sleep(0.08)
            except Exception as e:
                print(f"[gainers] Skipping {symbol}: {e}")
                continue

        # Sort by % gain descending, return top 5
        gainers.sort(key=lambda x: x["percent"], reverse=True)
        top5 = gainers[:5]

        if not top5:
            return jsonify({"success": False, "error": "No gainers data. Market may be closed."}), 500

        return jsonify({"success": True, "data": top5})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
# ==============================
# HISTORICAL DATA
# ==============================

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


def fetch_yahoo_historical(symbol, period):
    """Fetch full OHLCV history using the same Yahoo crumb session as live_ticker."""
    global _yahoo_session, _yahoo_crumb
    if not _yahoo_session:
        init_yahoo_session()

    # Map period string to Yahoo Finance range param
    range_map = {
        "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
        "1y":  "1y",  "2y":  "2y",  "5y":  "5y",
    }
    yf_range = range_map.get(period, "6mo")

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={yf_range}"
    if _yahoo_crumb:
        url += f"&crumb={_yahoo_crumb}"

    r = _yahoo_session.get(url, timeout=15)

    if r.status_code in (401, 403):
        init_yahoo_session()
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={yf_range}"
        if _yahoo_crumb:
            url += f"&crumb={_yahoo_crumb}"
        r = _yahoo_session.get(url, timeout=15)

    r.raise_for_status()
    chart = r.json()["chart"]["result"][0]

    timestamps = chart.get("timestamp", [])
    quote      = chart["indicators"]["quote"][0]

    opens   = quote.get("open",   [])
    highs   = quote.get("high",   [])
    lows    = quote.get("low",    [])
    closes  = quote.get("close",  [])
    volumes = quote.get("volume", [])

    records = []
    from datetime import datetime, timezone
    for i, ts in enumerate(timestamps):
        try:
            c = closes[i]
            if c is None or c != c:   # skip None / NaN
                continue
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            records.append({
                "date":   dt,
                "open":   round(float(opens[i]   or c), 2),
                "high":   round(float(highs[i]   or c), 2),
                "low":    round(float(lows[i]    or c), 2),
                "close":  round(float(c),              2),
                "volume": int(volumes[i] or 0),
            })
        except Exception:
            continue

    return records


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
            "success":  True,
            "symbol":   symbol_key,
            "period":   period,
            "data":     records,
            "summary": {
                "start_price":  first_close,
                "end_price":    last_close,
                "pct_change":   pct_change,
                "period_high":  period_high,
                "period_low":   period_low,
                "avg_volume":   avg_vol,
                "data_points":  len(records),
            }
        })

    except Exception as e:
        import traceback
        print("[historical] Error:", traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np
import yfinance as yf
import requests
import time

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# ==============================
# MODEL DIRECTORY
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
# SCALER (temporary)
# ==============================
scaler = MinMaxScaler(feature_range=(0,1))

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

       pred = model.predict(X_input)
       pred_price = scaler.inverse_transform(pred)

       return float(pred_price[0][0])

   except Exception as e:
       print("LSTM error:", e)
       return None

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
   return jsonify({"status": "ok", "message": "Stock Prediction API Running 🚀"})


@app.route("/test")
def test():
   return jsonify({"success": True, "message": "API is working 🚀"})


# ==============================
# MANUAL PREDICTION (UNCHANGED)
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
   try:
       body = request.get_json()

       company = body.get("company")
       open_price = float(body.get("open"))
       high_price = float(body.get("high"))
       low_price  = float(body.get("low"))
       volume     = float(body.get("volume"))

       model_key_map = {
           "TCS": "tcs",
           "Reliance": "reliance",
           "NIFTY": "nifty50",
           "SENSEX": "sensex"
       }

       key = model_key_map.get(company)

       if not key:
           return jsonify({"success": False, "error": f"Unknown company: {company}"}), 400

       if key not in models:
           return jsonify({"success": False, "error": f"Model not found"}), 500

       model = models[key]

       input_data = np.array([[open_price, high_price, low_price, volume]])
       prediction = model.predict(input_data)[0]

       return jsonify({
           "success": True,
           "company": company,
           "predicted_close": float(prediction)
       })

   except Exception as e:
       return jsonify({"success": False, "error": str(e)}), 500


# ==============================
# LIVE + LSTM PREDICTION
# ==============================
@app.route("/predict_live", methods=["GET"])
def predict_live():
   try:
       company = request.args.get("company")

       ticker_map = {
           "TCS": "TCS.NS",
           "Reliance": "RELIANCE.NS",
           "NIFTY": "^NSEI",
           "SENSEX": "^BSESN"
       }

       model_key_map = {
           "TCS": "tcs",
           "Reliance": "reliance",
           "NIFTY": "nifty50",
           "SENSEX": "sensex"
       }

       ticker = ticker_map.get(company)

       if not ticker:
           return jsonify({"success": False, "error": f"Unknown company"}), 400

       # ------------------------
       # LIVE DATA (1 DAY)
       # ------------------------
       stock = yf.Ticker(ticker)
       df_live = stock.history(period="1d")

       if df_live.empty:
           return jsonify({"success": False, "error": "Market closed or no data"}), 404

       latest = df_live.iloc[-1]

       open_price = latest["Open"]
       high_price = latest["High"]
       low_price  = latest["Low"]
       volume     = latest["Volume"]

       # ------------------------
       # ENSEMBLE PREDICTION
       # ------------------------
       key = model_key_map[company]
       model = models[key]

       input_data = np.array([[open_price, high_price, low_price, volume]])
       price_pred = float(model.predict(input_data)[0])

       # ------------------------
       # LSTM DATA (6 MONTHS)
       # ------------------------
       df_lstm = yf.download(ticker, period="6mo")

       lstm_pred = None
       trend = None

       if key in lstm_models:
           lstm_model = lstm_models[key]

           lstm_pred = lstm_predict(df_lstm, lstm_model)

           if lstm_pred:
               today_close = df_lstm["Close"].iloc[-1]
               trend = "UP" if lstm_pred > today_close else "DOWN"

       # ------------------------
       # FINAL RESPONSE
       # ------------------------
       return jsonify({
           "success": True,
           "company": company,

           "live_data": {
               "open": float(open_price),
               "high": float(high_price),
               "low": float(low_price),
               "volume": float(volume)
           },

           "predicted_close": price_pred,   # OLD MODEL
           "lstm_prediction": lstm_pred,    # NEW MODEL
           "trend": trend                  # PATTERN
       })

   except Exception as e:
       return jsonify({"success": False, "error": str(e)}), 500


# ==============================
# LIVE TICKER — Yahoo crumb session
# ==============================

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

_yahoo_session = None
_yahoo_crumb   = None

def init_yahoo_session():
    global _yahoo_session, _yahoo_crumb
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        s.get("https://fc.yahoo.com", timeout=5)
        s.get("https://finance.yahoo.com", timeout=8)
        r = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=8)
        if r.status_code == 200 and r.text and len(r.text) < 50:
            _yahoo_crumb = r.text.strip()
            print(f"[ticker] Yahoo session ready, crumb: {_yahoo_crumb}")
        else:
            print(f"[ticker] Crumb failed ({r.status_code}), will try without crumb")
    except Exception as e:
        print(f"[ticker] Session init error: {e}")
    _yahoo_session = s

def fetch_yahoo_price(symbol):
    global _yahoo_session, _yahoo_crumb
    if not _yahoo_session:
        init_yahoo_session()

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1mo"
    if _yahoo_crumb:
        url += f"&crumb={_yahoo_crumb}"

    r = _yahoo_session.get(url, timeout=10)

    # Refresh session and retry once on auth failure
    if r.status_code in (401, 403):
        init_yahoo_session()
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1mo"
        if _yahoo_crumb:
            url += f"&crumb={_yahoo_crumb}"
        r = _yahoo_session.get(url, timeout=10)

    r.raise_for_status()
    closes = [c for c in r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"] if c is not None]

    if len(closes) < 2:
        raise ValueError(f"Not enough data for {symbol}")

    current = round(float(closes[-1]), 2)
    prev    = round(float(closes[-2]), 2)
    change  = round(current - prev, 2)
    pct     = round((change / prev) * 100, 2) if prev else 0.0
    return current, change, pct


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

#TOPGAINER
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

@app.route("/top_gainers")
def top_gainers():
    try:
        gainers = []
        for name, symbol in NIFTY50_SYMBOLS:
            try:
                price, change, pct = fetch_yahoo_price(symbol)
                if pct > 0:
                    gainers.append({
                        "name":    name,
                        "price":   price,
                        "change":  change,
                        "percent": pct,
                    })
                time.sleep(0.08)
            except Exception as e:
                print(f"[gainers] Skipping {symbol}: {e}")
                continue

        # Sort by % gain descending, return top 5
        gainers.sort(key=lambda x: x["percent"], reverse=True)
        top5 = gainers[:5]

        if not top5:
            return jsonify({"success": False, "error": "No gainers data. Market may be closed."}), 500

        return jsonify({"success": True, "data": top5})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
   init_yahoo_session()   # warm up Yahoo session at startup
   app.run(host="0.0.0.0", port=5000, debug=True)
# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
   init_yahoo_session()   # warm up Yahoo session at startup
   app.run(host="0.0.0.0", port=5000, debug=True)
